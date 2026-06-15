"""Tests for metric spatiotemporal from a 3D .trc (synthetic walking with known geometry)."""

import numpy as np

from gait_analysis.analysis.spatiotemporal_3d import compute, read_trc
from gait_analysis.biomech.blazepose_to_trc import write_trc


def _make_walk_trc(path, fs=100.0, dur=6.0, speed=1.2, stride_T=1.05, amp=0.35, width=0.2):
    t = np.arange(0, dur, 1 / fs)
    w = 2 * np.pi / stride_T * t
    px = speed * t
    names = ["RHip", "LHip", "RHeel", "LHeel"]
    pos = np.zeros((len(t), 4, 3))
    # axis 0 = forward, axis 1 = vertical (up), axis 2 = lateral
    pos[:, 0] = np.column_stack([px, np.full_like(t, 0.95), np.zeros_like(t)])           # RHip
    pos[:, 1] = np.column_stack([px, np.full_like(t, 0.95), np.zeros_like(t)])           # LHip
    pos[:, 2] = np.column_stack([px + amp * np.sin(w), np.full_like(t, 0.05),
                                 np.full_like(t, width / 2)])                            # RHeel
    pos[:, 3] = np.column_stack([px + amp * np.sin(w + np.pi), np.full_like(t, 0.05),
                                 np.full_like(t, -width / 2)])                           # LHeel
    write_trc(path, names, pos, t)


def test_read_trc_roundtrip(tmp_path):
    p = tmp_path / "w.trc"
    _make_walk_trc(p)
    times, names, pos = read_trc(p)
    assert names == ["RHip", "LHip", "RHeel", "LHeel"]
    assert pos.shape[1:] == (4, 3) and len(times) == pos.shape[0]


def test_spatiotemporal_values(tmp_path):
    p = tmp_path / "w.trc"
    _make_walk_trc(p, speed=1.2, stride_T=1.05, width=0.2)
    r = compute(p)
    assert r["available"] is True
    assert abs(r["speed_m_s"] - 1.2) < 0.1                       # pelvis travel
    assert abs(r["stride_length_m"]["r"] - 1.26) < 0.15          # v * stride_T
    assert abs(r["step_width_m"] - 0.2) < 0.05                   # lateral foot separation
    assert 80 < r["cadence_steps_min"] < 140                     # ~114
    assert 0.9 < r["stride_length_symmetry"] <= 1.0              # L/R near symmetric


def test_missing_markers_graceful(tmp_path):
    p = tmp_path / "bad.trc"
    t = np.arange(0, 2, 0.01)
    write_trc(p, ["Head", "Neck"], np.zeros((len(t), 2, 3)), t)
    r = compute(p)
    assert r["available"] is False and "needs" in r["_note"]
