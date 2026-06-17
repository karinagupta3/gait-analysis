"""Tests for Timed Up & Go (TUG) timing from a synthetic hip trajectory."""

import numpy as np

from gait_analysis.analysis import tug_2d as T


def _synthetic_tug(n=300, fps=30.0):
    il = np.full((n, 33, 2), 0.5, np.float32)
    vis = np.full((n, 33), 0.9, np.float32)
    hy = np.empty(n)
    hx = np.empty(n)
    for i in range(n):
        if i < 30:          y, x = 0.65, 0.5                       # seated
        elif i < 60:        y, x = 0.65 - (i - 30) / 30 * 0.20, 0.5  # rise
        elif i < 150:       y, x = 0.45, 0.5 - (i - 60) / 90 * 0.25  # walk away
        elif i < 160:       y, x = 0.45, 0.25                      # turn
        elif i < 260:       y, x = 0.45, 0.25 + (i - 160) / 100 * 0.25  # walk back
        else:               y, x = 0.45 + (i - 260) / 40 * 0.20, 0.5    # sit
        hy[i], hx[i] = y, x
    for idx in (23, 24):
        il[:, idx, 1] = hy; il[:, idx, 0] = hx
    for idx in (27, 28):
        il[:, idx, 1] = hy + 0.25; il[:, idx, 0] = hx
    return il, vis


def test_tug_total_time_and_event_order():
    il, vis = _synthetic_tug()
    m = T.compute_tug_metrics(il, vis, 1280, 720, 30.0)
    assert m["total_time_s"] is not None
    assert 7.5 <= m["total_time_s"] <= 10.0          # ~10 s clip, trimmed to motion
    ev = m["_events"]
    assert ev["start"] < ev["stand"] < ev["turn"] < ev["end"]


def test_tug_unusable_clip_returns_none():
    il = np.full((20, 33, 2), 0.5, np.float32)        # too short / no motion
    vis = np.full((20, 33), 0.9, np.float32)
    m = T.compute_tug_metrics(il, vis, 1280, 720, 30.0)
    assert m["total_time_s"] is None
