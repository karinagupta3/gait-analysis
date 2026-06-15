"""Tests for concurrent-validity comparison of two .mot files."""

import numpy as np

from gait_analysis.analysis.validate import _plane, compare


def _write_mot(path, n=200, noise=0.0, bias=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2.0, n)
    w = 2 * np.pi * 1.0 * t
    cols = ["time", "hip_flexion_r", "knee_angle_r", "hip_adduction_r"]
    base = np.column_stack([
        t,
        20 * np.sin(w), 30 + 30 * np.sin(w), 6 * np.sin(w),
    ])
    if noise or bias:
        base[:, 1:] += rng.normal(0, noise, base[:, 1:].shape) + bias
    lines = ["m", "nColumns=4", "inDegrees=yes", "endheader", "\t".join(cols)]
    for r in base:
        lines.append("\t".join(f"{v:.5f}" for v in r))
    path.write_text("\n".join(lines) + "\n")


def test_plane_classification():
    assert _plane("knee_angle_r") == "sagittal"
    assert _plane("hip_flexion_l") == "sagittal"
    assert _plane("hip_adduction_r") == "frontal"
    assert _plane("hip_rotation_r") == "transverse"


def test_close_signals_pass(tmp_path):
    _write_mot(tmp_path / "ref.mot")
    _write_mot(tmp_path / "test.mot", noise=0.5, bias=0.5, seed=1)
    res = compare(tmp_path / "ref.mot", tmp_path / "test.mot")
    assert res["summary"]["pass"] is True
    assert res["summary"]["sagittal_rmse_mean"] < 5.0
    # high correlation on the sagittal knee
    assert res["per_coordinate"]["knee_angle_r"]["r"] > 0.9


def test_large_offset_fails(tmp_path):
    _write_mot(tmp_path / "ref.mot")
    _write_mot(tmp_path / "test.mot", bias=20.0, seed=2)   # 20 deg off
    res = compare(tmp_path / "ref.mot", tmp_path / "test.mot")
    assert res["summary"]["pass"] is False
    assert res["per_coordinate"]["knee_angle_r"]["rmse"] > 5.0
