"""Where to put our named markers on the LaiUhlrich2022 OpenSim model (Track B).

The base model ships with its own anatomical marker set, not markers named like our
BlazePose keypoints. So we INJECT virtual markers named like our keypoints. We place
each at the relevant **joint centre** (resolved from the model) because video keypoints
already approximate joint centres -- this avoids inventing skin-marker offsets.

This spec is the reviewable artifact; exact locations are refined by validating Track B
against Track A (docs/05). Body and joint names below follow the Rajagopal/LaiUhlrich
convention; `build_marked_model.py` resolves a joint's location on `body` and falls back
to the body origin + `offset` if the joint can't be resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .markerset import active_markers


@dataclass(frozen=True)
class Placement:
    marker: str                       # our keypoint name (an active IK marker)
    body: str                         # OpenSim body the marker rigidly attaches to
    at_joint: str | None = None       # place at this joint's centre (in `body` frame)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # extra local offset (m)
    note: str = ""


# Rajagopal/LaiUhlrich bodies: pelvis, femur_r/l, tibia_r/l, talus_r/l, calcn_r/l,
# toes_r/l, torso, humerus_r/l, ulna_r/l (and radius/hand). Joints: hip_r/l, knee_r/l,
# ankle_r/l, subtalar_r/l, mtp_r/l, acromial_r/l, elbow_r/l, back.
PLACEMENTS: list[Placement] = [
    # Pelvis / trunk
    Placement("left_hip",  "pelvis", "hip_l"),
    Placement("right_hip", "pelvis", "hip_r"),
    Placement("left_shoulder",  "torso", "acromial_l"),
    Placement("right_shoulder", "torso", "acromial_r"),
    Placement("nose", "torso", offset=(0.0, 0.5, 0.0), note="approx head; low IK weight"),
    # Lower limb -- joint centres
    Placement("left_knee",  "femur_l", "knee_l"),
    Placement("right_knee", "femur_r", "knee_r"),
    Placement("left_ankle",  "tibia_l", "ankle_l"),
    Placement("right_ankle", "tibia_r", "ankle_r"),
    Placement("left_heel",  "calcn_l", note="calcaneus origin ~ heel"),
    Placement("right_heel", "calcn_r", note="calcaneus origin ~ heel"),
    Placement("left_foot_index",  "toes_l", note="toe segment origin"),
    Placement("right_foot_index", "toes_r", note="toe segment origin"),
    # Upper limb
    Placement("left_elbow",  "humerus_l", "elbow_l"),
    Placement("right_elbow", "humerus_r", "elbow_r"),
    Placement("left_wrist",  "ulna_l", note="distal forearm ~ wrist"),
    Placement("right_wrist", "ulna_r", note="distal forearm ~ wrist"),
]


def placements() -> list[Placement]:
    return list(PLACEMENTS)


def validate() -> dict:
    """Check the spec exactly covers the active IK markers. Returns {missing, extra}."""
    placed = {p.marker for p in PLACEMENTS}
    active = set(active_markers())
    return {"missing": sorted(active - placed), "extra": sorted(placed - active)}
