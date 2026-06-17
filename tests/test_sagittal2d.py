"""Tests for the 2D sagittal screening frame-validity gate and angle masking.

The gate is the fix for "markers are off": it must keep clean full-body frames and
reject (a) edge-clipped frames, (b) low-confidence frames, and (c) the anatomically
implausible frames MediaPipe hallucinates when the legs are out of frame.
"""

import numpy as np

from gait_analysis.analysis.sagittal2d import (
    compute_sagittal_angles,
    smooth_along_time,
    valid_frame_mask,
)
from gait_analysis.pose.mediapipe3d import BLAZEPOSE_33

_IDX = {n: i for i, n in enumerate(BLAZEPOSE_33)}


def _pose(shoulder=0.30, hip=0.52, knee=0.72, ankle=0.92, x=0.50, vis=0.9):
    """One frame (1,33,2)+(1,33) with a plausible upright right+left leg at column x."""
    il = np.full((1, 33, 2), 0.5, np.float32)
    v = np.full((1, 33), vis, np.float32)
    for side in ("left", "right"):
        for joint, y in (("shoulder", shoulder), ("hip", hip), ("knee", knee), ("ankle", ankle)):
            j = _IDX[f"{side}_{joint}"]
            il[0, j] = (x, y)
    return il, v


def test_gate_keeps_plausible_pose():
    il, v = _pose()
    assert valid_frame_mask(il, v)[0] == True   # noqa: E712


def test_gate_rejects_edge_clipped():
    il, v = _pose(x=0.01)                         # leg at the very left edge
    assert valid_frame_mask(il, v)[0] == False   # noqa: E712


def test_gate_rejects_low_confidence():
    il, v = _pose(vis=0.2)
    assert valid_frame_mask(il, v)[0] == False   # noqa: E712


def test_gate_rejects_hallucinated_bunched_leg():
    # Legs out of frame -> MediaPipe collapses knee/ankle up near the hip (high vis,
    # in-frame coords) -> visibility+bounds alone would pass; geometry must reject it.
    il, v = _pose(hip=0.50, knee=0.52, ankle=0.54)   # thigh/shank collapsed
    assert valid_frame_mask(il, v)[0] == False        # noqa: E712


def test_gate_rejects_inverted_order():
    il, v = _pose(hip=0.80, knee=0.60, ankle=0.40)   # ankle above knee above hip (upside down)
    assert valid_frame_mask(il, v)[0] == False        # noqa: E712


def test_angles_ignore_invalid_frames():
    # 20 good frames + 10 edge-clipped frames -> only 20 used, angles still computed.
    good = [_pose(x=0.5) for _ in range(20)]
    bad = [_pose(x=0.01) for _ in range(10)]
    il = np.concatenate([g[0] for g in good] + [b[0] for b in bad], axis=0)
    v = np.concatenate([g[1] for g in good] + [b[1] for b in bad], axis=0)
    out = compute_sagittal_angles(il, v, 1280, 720)
    assert out["frames_used"] == 20
    assert out["frames_total"] == 30


def test_smooth_preserves_shape_and_reduces_jitter():
    rng = np.random.default_rng(0)
    base = np.linspace(0, 1, 50)[:, None] + np.zeros((50, 3))
    noisy = base + rng.normal(0, 0.05, (50, 3))
    sm = smooth_along_time(noisy, win=5)
    assert sm.shape == noisy.shape
    # Smoothed signal is closer to the underlying ramp than the noisy one.
    assert np.mean(np.abs(sm - base)) < np.mean(np.abs(noisy - base))
