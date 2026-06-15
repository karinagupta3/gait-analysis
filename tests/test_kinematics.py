"""Tests for OpenSim .mot parsing, ROM, symmetry, and plotting (offline)."""

import numpy as np

from gait_analysis.analysis.kinematics import (
    compute_rom, plot_coordinates, read_storage, summarize, symmetry,
)


def _write_synthetic_mot(path, n=101):
    t = np.linspace(0, 1.0, n)
    hip_r = 30 * np.sin(2 * np.pi * t)          # range ~60
    hip_l = 15 * np.sin(2 * np.pi * t + 0.2)    # range ~30 -> asymmetric
    knee_r = 30 + 30 * np.sin(2 * np.pi * t)
    knee_l = 30 + 30 * np.sin(2 * np.pi * t)
    pelvis_tilt = -12 + 2 * np.sin(2 * np.pi * t)

    cols = ["time", "hip_flexion_r", "hip_flexion_l",
            "knee_angle_r", "knee_angle_l", "pelvis_tilt"]
    data = np.column_stack([t, hip_r, hip_l, knee_r, knee_l, pelvis_tilt])

    lines = [
        "synthetic",
        "version=1",
        f"nRows={n}",
        f"nColumns={len(cols)}",
        "inDegrees=yes",
        "endheader",
        "\t".join(cols),
    ]
    for row in data:
        lines.append("\t".join(f"{v:.6f}" for v in row))
    path.write_text("\n".join(lines) + "\n")


def test_read_and_rom(tmp_path):
    mot = tmp_path / "ik.mot"
    _write_synthetic_mot(mot)
    time, coords, meta = read_storage(mot)

    assert len(time) == 101
    assert meta["inDegrees"] == "yes"
    assert set(coords) == {"hip_flexion_r", "hip_flexion_l",
                           "knee_angle_r", "knee_angle_l", "pelvis_tilt"}

    rom = compute_rom(coords)
    assert abs(rom["hip_flexion_r"]["range"] - 60) < 1.0
    assert abs(rom["knee_angle_r"]["range"] - 60) < 1.0


def test_symmetry_flags_asymmetry(tmp_path):
    mot = tmp_path / "ik.mot"
    _write_synthetic_mot(mot)
    _, coords, _ = read_storage(mot)
    sym = symmetry(coords)
    # knee is symmetric (~1.0); hip is deliberately asymmetric (~0.5).
    assert abs(sym["knee_angle"] - 1.0) < 0.05
    assert sym["hip_flexion"] < 0.7


def test_plot_writes_png(tmp_path):
    mot = tmp_path / "ik.mot"
    _write_synthetic_mot(mot)
    time, coords, meta = read_storage(mot)
    out = plot_coordinates(time, coords, tmp_path / "plot.png")
    assert out.exists() and out.stat().st_size > 0

    summary = summarize(time, coords, meta)
    assert summary["n_coordinates"] == 5
    assert summary["in_degrees"] is True
