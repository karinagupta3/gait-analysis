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


def test_helpers_handle_empty():
    assert M._mean([]) is None
    assert M._sd([]) is None
    assert M._nanmax(np.array([np.nan, np.nan])) is None
