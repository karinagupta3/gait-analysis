"""Shared constants and lightweight config.

We standardize on the COCO-17 keypoint layout that RTMPose emits by default.
Foot keypoints (heel/big toe) would come from a Halpe-26 model later; for Phase 1
gait-event detection we use the ankle keypoints relative to the pelvis (Zeni 2008),
which only needs COCO-17.
"""

from __future__ import annotations

from dataclasses import dataclass

# COCO-17 keypoint indices (RTMPose default output order).
COCO17 = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


@dataclass(frozen=True)
class PoseConfig:
    """Defaults for RTMPose extraction."""
    # rtmlib model size: 'lightweight' | 'balanced' | 'performance'
    mode: str = "balanced"
    backend: str = "onnxruntime"
    # 'cpu' is the safe default on Apple Silicon. CoreML acceleration is possible
    # via onnxruntime's CoreMLExecutionProvider but is environment-dependent.
    device: str = "cpu"
    # Minimum mean keypoint confidence to accept a person detection in a frame.
    min_person_score: float = 0.3


@dataclass(frozen=True)
class GaitConfig:
    """Defaults for spatiotemporal analysis."""
    fps: float = 60.0
    # Minimum separation between successive heel strikes, as a fraction of fps.
    # ~0.4 s is a conservative floor for human stride timing.
    min_event_sep_s: float = 0.4
    # Keypoint confidence below this is treated as missing (NaN) per-frame.
    min_keypoint_score: float = 0.3
