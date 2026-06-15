"""Test the HTML report generator (offline)."""

import numpy as np

from gait_analysis.analysis.report import build_html_report


def _write_mot(path, n=200):
    t = np.linspace(0, 2.0, n)
    f = 1.0
    cols = ["time", "hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l",
            "ankle_angle_r", "ankle_angle_l", "pelvis_tilt"]
    data = np.column_stack([
        t,
        20 * np.sin(2 * np.pi * f * t), 20 * np.sin(2 * np.pi * f * t),
        20 + 15 * np.sin(2 * np.pi * f * t), 30 + 30 * np.sin(2 * np.pi * f * t),
        10 * np.sin(2 * np.pi * f * t), 10 * np.sin(2 * np.pi * f * t),
        -10 + 2 * np.sin(2 * np.pi * f * t),
    ])
    lines = ["synthetic", "nColumns=8", "inDegrees=yes", "endheader", "\t".join(cols)]
    for row in data:
        lines.append("\t".join(f"{v:.5f}" for v in row))
    path.write_text("\n".join(lines) + "\n")


def test_build_html_report(tmp_path):
    mot = tmp_path / "ik.mot"
    _write_mot(mot)
    out = build_html_report(mot, tmp_path / "report.html", gait_speed_m_s=1.2, subject="Demo")
    assert out.exists()
    text = out.read_text()
    assert "Clinical signature flags" in text
    assert "Joint-angle curves" in text
    assert "Range of motion" in text
    assert "Limitations" in text
    assert "data:image/png;base64," in text          # embedded plot
    assert "Data confidence" in text
    assert "Demo" in text                             # subject metadata
