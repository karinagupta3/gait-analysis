"""Tests for rep-based movement metrics (squat / sit-to-stand) helpers."""

import numpy as np

from gait_analysis.analysis import movement_2d as M


def test_detect_reps_counts_cycles():
    fps = 30.0
    t = np.arange(300)
    knee = 40 + 40 * np.sin(2 * np.pi * t / 60.0)   # 2 s period -> ~5 cycles
    bottoms, tops = M._detect_reps(knee, fps)
    assert 4 <= len(bottoms) <= 6


def test_detect_reps_ignores_small_wobble():
    fps = 30.0
    t = np.arange(300)
    knee = 10 + 2 * np.sin(2 * np.pi * t / 30.0)    # tiny excursion, not real reps
    bottoms, _ = M._detect_reps(knee, fps)
    assert len(bottoms) == 0                          # prominence gate rejects wobble


def test_symmetry_index():
    assert M._sym(100.0, 100.0) == 100
    assert M._sym(100.0, 50.0) == 50
    assert M._sym(None, 5.0) is None


def test_sts_extra_5x_30s_and_power():
    fps = 30.0
    bottoms = [0, 60, 120, 180, 240, 300, 360]      # seated
    tops = [30, 90, 150, 210, 270, 330]             # 6 stands
    out = M._sts_extra(tops, bottoms, fps, n_frames=900, height_cm=170, weight_kg=70)
    assert out["stands"] == 6
    assert out["sts_5x_time_s"] == 9.0              # time to 5th stand (frame 270)
    assert out["sts_30s_count"] == 6                # 30 s clip
    assert 2.5 < out["power_wkg"] < 5.0             # plausible leg power
    assert out["power_w"] > 0


def test_sts_extra_no_power_without_anthropometrics():
    out = M._sts_extra([30, 90, 150, 210, 270], [0, 60, 120, 180, 240, 300],
                       30.0, n_frames=330, height_cm=None, weight_kg=None)
    assert "power_wkg" not in out
    assert out["sts_5x_time_s"] == 9.0


def test_helpers_handle_empty():
    assert M._mean([]) is None
    assert M._sd([]) is None
    assert M._nanmax(np.array([np.nan, np.nan])) is None
