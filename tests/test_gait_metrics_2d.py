"""Tests for the rich 2D gait-metrics helpers (events, temporal, per-stride peaks).

These exercise the core logic on synthetic angle series so they run without a video
or mediapipe.
"""

import numpy as np

from gait_analysis.analysis import gait_metrics_2d as G


def test_events_detected_on_periodic_signal():
    fps = 30.0
    t = np.arange(300)
    hip = 20 * np.sin(2 * np.pi * t / 30.0)   # 1 s period (30 frames)
    hs, to = G._events(hip, fps)
    assert len(hs) >= 8                         # ~10 cycles over 300 frames
    assert len(to) >= 8


def test_temporal_metrics_plausible():
    fps = 30.0
    t = np.arange(300)
    hip = 20 * np.sin(2 * np.pi * t / 30.0)
    hs, to = G._events(hip, fps)
    temp = G._temporal(hs, to, fps)
    assert abs(temp["stride_time_s"] - 1.0) < 0.2
    assert 100 < temp["cadence_spm"] < 140
    assert temp["n_strides"] >= 8


def test_per_stride_peaks_mean_and_count():
    knee = np.concatenate([np.linspace(0, 60, 15) for _ in range(5)])
    hip = np.zeros_like(knee)
    hs = np.array([0, 15, 30, 45, 60])
    pk = G._per_stride_peaks(knee, hip, hs)
    assert pk["n_strides"] == 4
    assert 55 < pk["knee_peak_mean"] <= 60
    assert pk["knee_peak_sd"] < 5


def test_symmetry_index():
    assert G._sym(60.0, 60.0) == 100
    assert G._sym(60.0, 30.0) == 50
    assert G._sym(None, 60.0) is None


def test_savgol_preserves_peak_better_than_flat():
    fps = 60.0
    x = np.linspace(0, 4 * np.pi, 240)
    clean = 30 + 30 * np.sin(x)            # peak ~60
    noisy = clean + np.random.default_rng(0).normal(0, 3, clean.shape)
    sm = G._savgol(noisy, fps)
    # smoothed peak stays close to the true 60 (not flattened away)
    assert 52 < np.nanmax(sm) < 66
