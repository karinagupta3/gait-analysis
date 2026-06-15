"""Mapping from video keypoints to OpenSim IK markers, with per-marker weights.

OpenSim Inverse Kinematics minimizes the weighted sum of squared distances between
EXPERIMENTAL markers (our .trc, named like BlazePose keypoints) and the MODEL's
virtual markers of the same name. So two things must line up:
  1. the OpenSim model must carry virtual markers named exactly like these keys, and
  2. each marker gets an IK weight reflecting how much we trust it.

Weights encode the honesty rules from docs/04: sagittal, large-amplitude landmarks
(hips/knees/ankles/shoulders) are trusted most; distal/noisy points less; face and
hands are excluded (weight 0 / omitted). This is the single-camera (quick-mode)
marker set built on BlazePose-33; the accurate-mode (RTMPose/Halpe) set can extend it.
"""

from __future__ import annotations

# IK weights over BlazePose-33 keypoint names (see pose/mediapipe3d.BLAZEPOSE_33).
# 0 == do not use as an IK marker. Higher == trusted more in the fit.
IK_MARKER_WEIGHTS: dict[str, float] = {
    # Pelvis / trunk anchors -- highest weight, drive the root.
    "left_hip": 2.0, "right_hip": 2.0,
    "left_shoulder": 1.5, "right_shoulder": 1.5,
    # Lower limb -- the clinical core, sagittal-reliable.
    "left_knee": 1.5, "right_knee": 1.5,
    "left_ankle": 1.5, "right_ankle": 1.5,
    "left_heel": 1.0, "right_heel": 1.0,
    "left_foot_index": 1.0, "right_foot_index": 1.0,
    # Upper limb -- useful for arm-swing/elbow, lower weight (noisier).
    "left_elbow": 0.5, "right_elbow": 0.5,
    "left_wrist": 0.5, "right_wrist": 0.5,
    # Head -- single low-weight anchor for trunk/lumbar orientation only.
    "nose": 0.3,
    # Excluded (weight 0, omitted from IK): eyes, ears, mouth, pinky, index, thumb.
}


def active_markers() -> list[str]:
    """Marker names actually used by IK (weight > 0), in a stable order."""
    return [m for m, w in IK_MARKER_WEIGHTS.items() if w > 0]


def validate_against_trc_markers(trc_markers: list[str]) -> list[str]:
    """Return active IK markers that are MISSING from a TRC's marker list.

    Use this before running IK so a name mismatch fails loudly instead of
    silently dropping a marker.
    """
    trc = set(trc_markers)
    return [m for m in active_markers() if m not in trc]
