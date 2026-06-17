"""Rich single-camera 2D gait metrics (screening mode).

Builds on sagittal2d's image-plane angles, but extracts the clinically useful
quantities a screening report should show — for BOTH legs where visible:

  * sagittal knee + hip flexion, smoothed with a peak-preserving filter
  * gait events (heel strike ~ peak hip flexion, toe-off ~ peak hip extension;
    the standard kinematic "Zeni" approximation, since we have no force plate)
  * per-STRIDE peak knee flexion and hip flex/ext, reported as mean +/- SD across
    strides (far more robust + clinically meaningful than one min/max over the clip)
  * temporal metrics: cadence, stride time, step time, stance % / swing %
  * left/right symmetry (experimental — the far leg is noisier in one camera)
  * trunk lean and a "view quality" check (how lateral the walk is)

HONESTY: single-camera, sagittal plane only. The camera-facing leg is reliable;
the far leg and any out-of-plane motion are low-confidence. Ankle/foot angles are
NOT reported here — the 2D toe marker is too noisy to trust (it produced nonsense
in the old report). This is a screening estimate, not a diagnosis.
"""
from __future__ import annotations

import numpy as np

from ..pose.mediapipe3d import BLAZEPOSE_33
from .sagittal2d import _angle_at, valid_frame_mask

_IDX = {name: i for i, name in enumerate(BLAZEPOSE_33)}


def _savgol(y: np.ndarray, fps: float) -> np.ndarray:
    """Peak-preserving smoothing (Savitzky-Golay). Falls back to the raw signal
    when too short. Window ~1/3 s, clamped odd and < length."""
    finite = np.isfinite(y)
    if finite.sum() < 7:
        return y
    win = int(max(5, round(fps / 3)))
    if win % 2 == 0:
        win += 1
    if win >= finite.sum():
        return y
    try:
        from scipy.signal import savgol_filter
        out = y.copy()
        # interpolate interior NaNs so the filter is continuous, then restore NaNs
        idx = np.arange(len(y))
        good = np.isfinite(y)
        filled = np.interp(idx, idx[good], y[good])
        sm = savgol_filter(filled, win, 3)
        out[good] = sm[good]
        return out
    except Exception:
        return y


def _side_angles(px, vis, side, valid):
    """Knee + hip sagittal flexion (deg) for one leg, masked to valid frames."""
    p = lambda j: px[:, _IDX[f"{side}_{j}"]]
    v = lambda j: vis[:, _IDX[f"{side}_{j}"]]
    hip, knee, ankle = p("hip"), p("knee"), p("ankle")
    knee_flex = 180.0 - _angle_at(knee, hip, ankle)            # 0 straight, + flexed
    thigh = knee - hip
    hip_flex = np.degrees(np.arctan2(thigh[:, 0], thigh[:, 1]))  # + anterior swing
    legvis = np.minimum.reduce([v("hip"), v("knee"), v("ankle")])
    mask = valid & (legvis >= 0.5)
    knee_flex = np.where(mask, knee_flex, np.nan)
    hip_flex = np.where(mask, hip_flex, np.nan)
    return knee_flex, hip_flex, int(mask.sum())


def _events(hip_flex: np.ndarray, fps: float):
    """Heel strikes ~ hip-flexion maxima, toe-offs ~ minima (Zeni kinematic proxy)."""
    from scipy.signal import find_peaks
    sep = max(1, int(0.4 * fps))                  # >=0.4 s between like events
    y = hip_flex.copy()
    bad = ~np.isfinite(y)
    if bad.all():
        return np.array([], int), np.array([], int)
    y[bad] = np.nanmean(y)                          # flat-fill gaps so peaks don't break
    hs, _ = find_peaks(y, distance=sep, prominence=3)
    to, _ = find_peaks(-y, distance=sep, prominence=3)
    return hs, to


def _per_stride_peaks(knee_flex, hip_flex, hs):
    """Mean +/- SD of per-stride peak knee flexion and hip flex/ext across strides."""
    if len(hs) < 2:
        return None
    kpk, hflex, hext = [], [], []
    for a, b in zip(hs[:-1], hs[1:]):
        kseg = knee_flex[a:b + 1]
        hseg = hip_flex[a:b + 1]
        if np.isfinite(kseg).any():
            kpk.append(np.nanmax(kseg))
        if np.isfinite(hseg).any():
            hflex.append(np.nanmax(hseg))
            hext.append(np.nanmin(hseg))
    if not kpk:
        return None
    return {
        "knee_peak_mean": float(np.mean(kpk)), "knee_peak_sd": float(np.std(kpk)),
        "hip_flex_mean": float(np.mean(hflex)) if hflex else None,
        "hip_ext_mean": float(np.mean(hext)) if hext else None,
        "n_strides": len(kpk),
    }


def _temporal(hs, to, fps):
    """Cadence, stride/step time, stance%/swing% from one leg's events."""
    out = {"n_strides": max(0, len(hs) - 1)}
    if len(hs) >= 2:
        stride_frames = np.diff(hs)
        stride_t = float(np.median(stride_frames) / fps)
        out["stride_time_s"] = round(stride_t, 2)
        out["step_time_s"] = round(stride_t / 2, 2)
        out["cadence_spm"] = round(120.0 / stride_t, 0)   # 2 steps per stride
        # stance = HS -> next TO; stance% of stride
        pcts = []
        for a, b in zip(hs[:-1], hs[1:]):
            tos = to[(to > a) & (to < b)]
            if tos.size:
                pcts.append((tos[0] - a) / (b - a) * 100.0)
        if pcts:
            out["stance_pct"] = round(float(np.mean(pcts)), 0)
            out["swing_pct"] = round(100.0 - float(np.mean(pcts)), 0)
    return out


def _trunk_lean(px, vis, valid):
    """Mean forward/back trunk lean (deg from vertical) over valid frames."""
    sh = (px[:, _IDX["left_shoulder"]] + px[:, _IDX["right_shoulder"]]) / 2
    hipm = (px[:, _IDX["left_hip"]] + px[:, _IDX["right_hip"]]) / 2
    v = np.minimum(
        np.minimum(vis[:, _IDX["left_shoulder"]], vis[:, _IDX["right_shoulder"]]),
        np.minimum(vis[:, _IDX["left_hip"]], vis[:, _IDX["right_hip"]]))
    trunk = sh - hipm                       # vector hip->shoulder (image y is down)
    ang = np.degrees(np.arctan2(trunk[:, 0], -trunk[:, 1]))  # 0 = upright
    ang = np.where(valid & (v >= 0.5), ang, np.nan)
    if np.isfinite(ang).sum() < 5:
        return None
    return round(float(np.nanmedian(ang)), 1)


def _view_quality(px, vis, valid, primary):
    """How lateral the walk is, scale- and tracking-shot-invariant: in a true side
    view the left and right hips sit (nearly) one behind the other, so their
    horizontal separation is small relative to the thigh length. Walking toward/away
    from the camera spreads the hips apart -> larger ratio -> foreshortened angles.
    """
    lh, rh = px[:, _IDX["left_hip"]], px[:, _IDX["right_hip"]]
    hip = px[:, _IDX[f"{primary}_hip"]]
    knee = px[:, _IDX[f"{primary}_knee"]]
    inter = np.abs(lh[:, 0] - rh[:, 0])
    thigh = np.linalg.norm(knee - hip, axis=1)
    m = valid & np.isfinite(inter) & np.isfinite(thigh) & (thigh > 1)
    if m.sum() < 5:
        return {"hip_sep_ratio": None, "label": "too few frames"}
    ratio = float(np.median(inter[m]) / np.median(thigh[m]))
    if ratio <= 0.35:
        label = "good side view"
    elif ratio <= 0.6:
        label = "partly oblique — interpret ranges with some caution"
    else:
        label = "walking toward/away from camera — angles foreshortened, low confidence"
    return {"hip_sep_ratio": round(ratio, 2), "label": label}


def _sym(a, b):
    if a is None or b is None or max(abs(a), abs(b)) < 1e-6:
        return None
    return round(100.0 * (1 - abs(a - b) / max(abs(a), abs(b))), 0)


def compute_gait_metrics(image_landmarks, visibility, width, height, fps) -> dict:
    """Full 2D screening metrics for both legs. See module docstring."""
    fps = float(fps) or 30.0
    px = image_landmarks.astype(float) * np.array([width, height])
    vis = visibility.astype(float)

    # Frame validity per side (camera-facing chain in frame + plausible).
    sides = {}
    side_quality = {}
    for side in ("right", "left"):
        valid = valid_frame_mask(image_landmarks, vis, side=side)
        kf, hf, nvalid = _side_angles(px, vis, side, valid)
        kf, hf = _savgol(kf, fps), _savgol(hf, fps)
        hs, to = _events(hf, fps)
        peaks = _per_stride_peaks(kf, hf, hs)
        temporal = _temporal(hs, to, fps)
        sides[side] = {
            "valid_frames": nvalid,
            "peaks": peaks,
            "temporal": temporal,
            "_series": {"knee": kf, "hip": hf},
            "_events": {"hs": hs.tolist(), "to": to.tolist()},
        }
        side_quality[side] = nvalid

    primary = "right" if side_quality["right"] >= side_quality["left"] else "left"
    valid_primary = valid_frame_mask(image_landmarks, vis, side=primary)

    # Symmetry between the two legs' per-stride knee peaks + stride time.
    rp, lp = sides["right"]["peaks"], sides["left"]["peaks"]
    rt, lt = sides["right"]["temporal"], sides["left"]["temporal"]
    symmetry = None
    if rp and lp:
        symmetry = {
            "knee_peak_pct": _sym(rp.get("knee_peak_mean"), lp.get("knee_peak_mean")),
            "stride_time_pct": _sym(rt.get("stride_time_s"), lt.get("stride_time_s")),
        }

    # Overall cadence: prefer the primary leg, else whichever leg has it.
    cadence = sides[primary]["temporal"].get("cadence_spm")
    if cadence is None:
        for s in ("right", "left"):
            if sides[s]["temporal"].get("cadence_spm"):
                cadence = sides[s]["temporal"]["cadence_spm"]
                break

    total = len(image_landmarks)
    return {
        "fps": round(fps, 1),
        "frames_total": int(total),
        "frames_used": int(max(side_quality.values())),
        "primary_side": primary,
        "cadence_spm": cadence,
        "view_quality": _view_quality(px, vis, valid_primary, primary),
        "trunk_lean_deg": _trunk_lean(px, vis, valid_primary),
        "sides": sides,
        "symmetry": symmetry,
    }
