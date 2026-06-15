"""Synthetic-signal tests for gait-event detection (no video/network needed)."""

import numpy as np

from gait_analysis.analysis.spatiotemporal import compute_parameters
from gait_analysis.config import COCO17, GaitConfig


def _synthetic_walk(fps=60.0, duration_s=5.0, stride_period_s=1.0, amp_px=50.0, vel_px_s=20.0):
    """Build a clean synthetic gait sequence with known cadence/symmetry.

    One heel strike per stride per foot; feet half a cycle out of phase.
    stride_period 1.0 s -> each foot ~1 HS/s -> both feet ~2 HS/s -> ~120 steps/min.
    """
    n = int(fps * duration_s)
    t = np.arange(n) / fps
    pelvis_x = vel_px_s * t
    f = 1.0 / stride_period_s

    kpts = np.zeros((n, 17, 2), dtype=np.float32)
    # Hips define the pelvis midpoint.
    kpts[:, COCO17["left_hip"], 0] = pelvis_x
    kpts[:, COCO17["right_hip"], 0] = pelvis_x
    # Ankles oscillate fore/aft about the pelvis, antiphase between sides.
    kpts[:, COCO17["left_ankle"], 0] = pelvis_x + amp_px * np.sin(2 * np.pi * f * t)
    kpts[:, COCO17["right_ankle"], 0] = pelvis_x + amp_px * np.sin(2 * np.pi * f * t + np.pi)
    scores = np.ones((n, 17), dtype=np.float32)
    return kpts, scores


def test_cadence_and_symmetry():
    kpts, scores = _synthetic_walk()
    params = compute_parameters(kpts, scores, GaitConfig(fps=60.0))

    # ~120 steps/min for a 1 s stride period, both feet.
    assert 110.0 <= params["cadence_steps_per_min"] <= 130.0

    # Symmetric synthetic gait -> ratio near 1.0.
    assert abs(params["stride_time_symmetry_ratio_LR"] - 1.0) <= 0.05

    # Roughly equal heel-strike counts per side.
    nl = params["n_heel_strikes"]["left"]
    nr = params["n_heel_strikes"]["right"]
    assert abs(nl - nr) <= 1
    assert nl >= 3


def test_handles_missing_detections():
    kpts, scores = _synthetic_walk()
    # Drop confidence on a few frames -> treated as NaN and interpolated.
    scores[10:15, :] = 0.0
    params = compute_parameters(kpts, scores, GaitConfig(fps=60.0))
    assert np.isfinite(params["cadence_steps_per_min"])
