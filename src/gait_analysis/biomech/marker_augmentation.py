"""Marker augmentation: sparse video keypoints -> 43 dense ANATOMICAL OpenSim markers.

This is OpenCap's load-bearing accuracy step (Uhlrich, Falisse, Delp et al., PLOS
Comput Biol 2023). Driving OpenSim inverse kinematics from raw video keypoints --
which is what our old quick path did -- gives joint angles that are ~3.4 deg worse
on average and up to ~32.6 deg worse on poorly-constrained DOFs like LUMBAR/trunk
extension. That trunk error is exactly the "hunched over" posture we saw in the 3D
viewer. The fix OpenCap uses is a small LSTM that maps the sparse keypoints to the
43 "_study" anatomical markers the LaiUhlrich2022 model is built around, and then
runs IK on THOSE markers.

We reuse the exact Stanford LSTM (Apache-2.0) bundled with Pose2Sim
(MarkerAugmenter/LSTM/v0.3_{lower,upper}/model.onnx + mean.npy/std.npy). The math
here faithfully replicates Pose2Sim.markerAugmentation.augment_markers_all:
  1. take the feature keypoints in OpenSim frame (Y up), one (x,y,z) per marker,
  2. subtract the hip-midpoint root marker, divide by subject height,
  3. append [height, mass], standardize with the model's mean/std,
  4. run the LSTM over the whole clip as one (1, T, F) sequence,
  5. un-standardize: out*height + hip-root -> 3D anatomical marker trajectories.

INPUT keypoints come from MediaPipe BlazePose-33. BlazePose lacks a distinct small
toe, so RSmallToe/LSmallToe are synthesized lateral to the big toe (low IK weight,
so the approximation is acceptable). Neck = shoulder midpoint, Hip = hip midpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..pose.mediapipe3d import BLAZEPOSE_33

# --- feature marker sets the LSTM was trained on (order matters!) -------------
# (mirrors Pose2Sim.markerAugmentation.getOpenPoseMarkers_* exactly)
_LOWER_FEATURES = [
    "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
    "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
    "RBigToe", "LBigToe",
]
_LOWER_RESPONSE = [
    "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study",
    "r_knee_study", "r_mknee_study", "r_ankle_study", "r_mankle_study",
    "r_toe_study", "r_5meta_study", "r_calc_study", "L_knee_study",
    "L_mknee_study", "L_ankle_study", "L_mankle_study", "L_toe_study",
    "L_calc_study", "L_5meta_study", "r_shoulder_study", "L_shoulder_study",
    "C7_study", "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
    "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", "r_sh1_study",
    "r_sh2_study", "r_sh3_study", "L_sh1_study", "L_sh2_study", "L_sh3_study",
    "RHJC_study", "LHJC_study",
]
_UPPER_FEATURES = ["Neck", "RShoulder", "LShoulder", "RElbow", "LElbow",
                   "RWrist", "LWrist"]
_UPPER_RESPONSE = ["r_lelbow_study", "r_melbow_study", "r_lwrist_study",
                   "r_mwrist_study", "L_lelbow_study", "L_melbow_study",
                   "L_lwrist_study", "L_mwrist_study"]

# Raw keypoints (named to match the LaiUhlrich2022 markerset) that IK uses
# DIRECTLY in addition to the augmented markers (withHands_LSTM IK setup).
_PASSTHROUGH = {
    "Nose": "nose",
    "RThumb": "right_thumb", "RIndex": "right_index", "RPinky": "right_pinky",
    "LThumb": "left_thumb", "LIndex": "left_index", "LPinky": "left_pinky",
}

# BlazePose index for each feature marker we can read directly.
_BP = {n: i for i, n in enumerate(BLAZEPOSE_33)}
_DIRECT = {
    "RShoulder": "right_shoulder", "LShoulder": "left_shoulder",
    "RHip": "right_hip", "LHip": "left_hip",
    "RKnee": "right_knee", "LKnee": "left_knee",
    "RAnkle": "right_ankle", "LAnkle": "left_ankle",
    "RHeel": "right_heel", "LHeel": "left_heel",
    "RElbow": "right_elbow", "LElbow": "left_elbow",
    "RWrist": "right_wrist", "LWrist": "left_wrist",
    "RBigToe": "right_foot_index", "LBigToe": "left_foot_index",
}

_AUG_DIR = None


def _augmenter_dir() -> Path:
    """Locate the bundled Stanford LSTM (ships inside Pose2Sim)."""
    global _AUG_DIR
    if _AUG_DIR is None:
        import Pose2Sim
        _AUG_DIR = Path(Pose2Sim.__file__).resolve().parent / "MarkerAugmenter" / "LSTM"
    return _AUG_DIR


def _build_feature_dict(world: np.ndarray) -> dict[str, np.ndarray]:
    """world: (T,33,3) in OpenSim frame -> {feature_name: (T,3)} incl. synthesized."""
    feats: dict[str, np.ndarray] = {}
    for name, bp in _DIRECT.items():
        feats[name] = world[:, _BP[bp], :]
    rsh, lsh = feats["RShoulder"], feats["LShoulder"]
    rhip, lhip = feats["RHip"], feats["LHip"]
    feats["Neck"] = 0.5 * (rsh + lsh)
    feats["Hip"] = 0.5 * (rhip + lhip)
    # Small toe: BlazePose has no 5th-toe keypoint. Place it lateral to the big
    # toe along the foot's lateral axis (forward = toe-heel, up = +Y).
    up = np.array([0.0, 1.0, 0.0])
    for side, sign in (("R", 1.0), ("L", -1.0)):
        big = feats[f"{side}BigToe"]
        heel = feats[f"{side}Heel"]
        fwd = big - heel
        fwd_n = fwd / (np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9)
        lateral = np.cross(fwd_n, up)            # points to the subject's right
        lateral /= (np.linalg.norm(lateral, axis=1, keepdims=True) + 1e-9)
        width = 0.4 * np.linalg.norm(fwd, axis=1, keepdims=True)
        # +right for the right foot's small toe, +left for the left foot's.
        feats[f"{side}SmallToe"] = big + sign * (-1.0) * lateral * width
    return feats


def _run_lstm(model_dir: Path, feature_names: list[str],
              feats: dict[str, np.ndarray], hip_root: np.ndarray,
              height_m: float, mass_kg: float) -> np.ndarray:
    """Replicates Pose2Sim's per-augmenter ONNX inference. Returns (T, n_resp*3)."""
    import onnxruntime as ort

    T = hip_root.shape[0]
    # (T, n_feat*3): per marker, hip-rooted then height-normalised.
    cols = []
    for name in feature_names:
        cols.append(feats[name] - hip_root)          # (T,3)
    X = np.concatenate(cols, axis=1) / height_m       # (T, n_feat*3)
    X = np.concatenate([X, np.full((T, 1), height_m), np.full((T, 1), mass_kg)], axis=1)

    mean = np.load(model_dir / "mean.npy", allow_pickle=True)
    std = np.load(model_dir / "std.npy", allow_pickle=True)
    X = (X - mean) / std
    X = X.reshape(1, T, X.shape[1]).astype(np.float32)

    sess = ort.InferenceSession(str(model_dir / "model.onnx"))
    out = sess.run(["output_0"], {"inputs": X})[0]    # (1, T, n_resp*3)
    out = out.reshape(out.shape[1], out.shape[2])      # (T, n_resp*3)
    # un-normalise: * height, then add hip root back per marker.
    out = out * height_m
    out = out + np.tile(hip_root, (1, out.shape[1] // 3))
    return out


def augment(world: np.ndarray, height_m: float, mass_kg: float
            ) -> tuple[list[str], np.ndarray]:
    """world: (T,33,3) in OpenSim frame (Y up). -> (marker_names, positions (T,M,3)).

    Output markers = 43 augmented anatomical "_study" markers + passthrough
    keypoints (Nose + 6 hand markers) the LaiUhlrich2022 IK setup consumes.
    """
    world = np.asarray(world, dtype=float)
    if world.ndim != 3 or world.shape[1:] != (33, 3):
        raise ValueError(f"expected (T,33,3) world landmarks, got {world.shape}")
    feats = _build_feature_dict(world)
    hip_root = feats["Hip"]
    aug = _augmenter_dir()

    lower = _run_lstm(aug / "v0.3_lower", _LOWER_FEATURES, feats, hip_root,
                      height_m, mass_kg)
    upper = _run_lstm(aug / "v0.3_upper", _UPPER_FEATURES, feats, hip_root,
                      height_m, mass_kg)

    names: list[str] = []
    chunks: list[np.ndarray] = []
    for i, n in enumerate(_LOWER_RESPONSE):
        names.append(n)
        chunks.append(lower[:, 3 * i:3 * i + 3])
    for i, n in enumerate(_UPPER_RESPONSE):
        names.append(n)
        chunks.append(upper[:, 3 * i:3 * i + 3])
    for marker, bp in _PASSTHROUGH.items():
        names.append(marker)
        chunks.append(world[:, _BP[bp], :])

    positions = np.stack(
        [np.column_stack([chunks[m][:, a] for a in range(3)]) for m in range(len(chunks))],
        axis=1,
    )  # (T, M, 3)
    return names, positions


def estimate_height_m(world: np.ndarray) -> float:
    """Rough subject height from the most-upright frames (heel-to-head span)."""
    head = world[:, _BP["nose"], :]
    rheel = world[:, _BP["right_heel"], :]
    lheel = world[:, _BP["left_heel"], :]
    foot_y = np.minimum(rheel[:, 1], lheel[:, 1])
    span = head[:, 1] - foot_y                  # nose-to-heel vertical
    span = span[np.isfinite(span)]
    if span.size == 0:
        return 1.70
    # nose sits ~0.95 of stature; take the tall (upright) percentile.
    stature = np.percentile(span, 90) / 0.93
    return float(np.clip(stature, 1.3, 2.1))
