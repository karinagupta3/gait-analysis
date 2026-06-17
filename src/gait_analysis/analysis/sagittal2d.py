"""Single-camera 2D SAGITTAL gait angles (screening mode).

Monocular 3D depth is too unreliable for metric OpenSim scaling (see
docs/05 + commit history), so for the one-phone path we measure sagittal-plane
joint flexion directly in the image plane of a SIDE-VIEW video. This needs no
depth, no model scaling, and no OpenSim -- it is robust precisely because it only
uses the two axes a single camera measures well.

SCOPE / HONESTY: side-view only; sagittal plane only (knee/hip/ankle flexion).
This is a screening estimate, not a diagnostic, and not 3D. Frontal/transverse
angles are NOT available from one camera. Perspective and out-of-plane motion add
error; report ranges, not false precision.
"""
from __future__ import annotations

import numpy as np

from ..pose.mediapipe3d import BLAZEPOSE_33

_IDX = {name: i for i, name in enumerate(BLAZEPOSE_33)}


def _angle_at(b, a, c):
    """Interior angle (degrees) at vertex b for points a-b-c. Each is (T,2)."""
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _pick_side(vis: np.ndarray) -> str:
    """Choose the limb facing the camera = the side with higher mean visibility."""
    r = np.nanmean([vis[:, _IDX[f"right_{j}"]] for j in ("hip", "knee", "ankle")])
    l = np.nanmean([vis[:, _IDX[f"left_{j}"]] for j in ("hip", "knee", "ankle")])
    return "right" if r >= l else "left"


# Sagittal chain that must be confidently detected AND inside the frame for a frame
# to be usable. This is the key quality gate: as the subject ENTERS or LEAVES the
# frame, MediaPipe still returns a (wrong, low-confidence) pose with scattered
# markers — those frames must be rejected from both the angles and the overlay.
_CHAIN = ("hip", "knee", "ankle")


def valid_frame_mask(
    image_landmarks: np.ndarray,   # (T,33,2) normalised 0..1
    visibility: np.ndarray,        # (T,33)
    side: str | None = None,
    min_visibility: float = 0.5,
    edge_margin: float = 0.03,
) -> np.ndarray:
    """Boolean (T,) mask of frames usable for sagittal analysis.

    A frame is valid only when the camera-facing hip-knee-ankle chain is (a) detected
    above ``min_visibility``, (b) fully inside the frame interior (not clipped at an
    edge), and (c) finite. This rejects the entry/exit frames that produce the
    scattered "markers are off" poses.
    """
    side = side or _pick_side(visibility)
    idxs = [_IDX[f"{side}_{j}"] for j in _CHAIN]
    xy = image_landmarks[:, idxs, :].astype(float)          # (T, 3, 2)
    vis_ok = np.min(visibility[:, idxs], axis=1) >= min_visibility
    finite = np.all(np.isfinite(xy), axis=(1, 2))
    safe = np.where(np.isfinite(xy), xy, 0.5)               # avoid NaN in the bounds test
    inside = np.all((safe >= edge_margin) & (safe <= 1.0 - edge_margin), axis=(1, 2))
    return vis_ok & finite & inside


def smooth_along_time(arr: np.ndarray, win: int = 5) -> np.ndarray:
    """Centered moving average over axis 0, ignoring NaN (reduces marker jitter).

    Returns a float copy; positions that are NaN across the whole window stay NaN.
    """
    arr = np.asarray(arr, dtype=float)
    T = arr.shape[0]
    if win < 2 or T < win:
        return arr.copy()
    half = win // 2
    out = np.empty_like(arr)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        seg = arr[lo:hi]
        with np.errstate(invalid="ignore"):
            out[t] = np.nanmean(seg, axis=0) if np.isfinite(seg).any() else arr[t]
    return out


def compute_sagittal_angles(
    image_landmarks: np.ndarray,   # (T,33,2) normalised 0..1
    visibility: np.ndarray,        # (T,33)
    width: int,
    height: int,
    min_visibility: float = 0.5,
    side: str | None = None,
) -> dict:
    """Per-frame sagittal flexion angles (knee, hip, ankle) for the camera-facing side."""
    px = image_landmarks.astype(float) * np.array([width, height])  # to pixels (true geometry)
    side = side or _pick_side(visibility)
    p = lambda j: px[:, _IDX[f"{side}_{j}"]]
    v = lambda j: visibility[:, _IDX[f"{side}_{j}"]]

    hip, knee, ankle = p("hip"), p("knee"), p("ankle")
    foot = p("foot_index")

    # Flexion = deviation from the straight (anatomically-neutral) configuration.
    knee_flex = 180.0 - _angle_at(knee, hip, ankle)          # 0 straight, + flexed
    # Hip flexion ~ thigh angle from the downward vertical (sagittal image plane).
    # Robust to trunk-marker noise; image y points DOWN, so downward vertical = +y.
    thigh = knee - hip
    hip_flex = np.degrees(np.arctan2(thigh[:, 0], thigh[:, 1]))  # signed: +anterior swing
    ankle_dorsi = _angle_at(ankle, knee, foot) - 90.0        # ~0 neutral, + dorsiflexion

    # Reject frames where the body is clipped at an edge or low-confidence (entry/exit
    # of the walk) — these produce the scattered, wrong poses. This is stricter than a
    # per-joint visibility test: it also requires the chain to be fully inside the frame.
    good = valid_frame_mask(image_landmarks, visibility, side, min_visibility)
    for arr in (knee_flex, hip_flex, ankle_dorsi):
        arr[~good] = np.nan

    def summ(a, lo_conf=False):
        a = a[np.isfinite(a)]
        if a.size < 5:
            return None
        # Robust 2nd/98th percentiles reject single-frame outliers (perspective,
        # partial occlusion) that otherwise inflate ROM in markerless data.
        lo, hi = float(np.percentile(a, 2)), float(np.percentile(a, 98))
        return {"min": lo, "max": hi, "rom": hi - lo, "low_confidence": lo_conf}

    return {
        "side": side,
        "frames_used": int(good.sum()),
        "frames_total": int(len(good)),
        "knee_flexion": summ(knee_flex),
        "hip_flexion": summ(hip_flex),
        "ankle_dorsiflexion": summ(ankle_dorsi, lo_conf=True),  # ankle from 2 foot pts = noisy
        "_series": {"knee_flexion": knee_flex, "hip_flexion": hip_flex, "ankle_dorsiflexion": ankle_dorsi},
    }
