"""Tests for gait-cycle normalization + the ensemble walking report."""

import numpy as np

from gait_analysis.analysis import normative
from gait_analysis.analysis.gait_cycle import cycle_normalize, ensemble


def _gait_coords(strides=6, fs=100.0):
    t = np.arange(0, strides / 0.95, 1 / fs)
    w = 2 * np.pi * 0.95 * t
    return t, {
        "hip_flexion_r": 12 + 20 * np.sin(w),
        "hip_flexion_l": 12 + 20 * np.sin(w + np.pi),
        "knee_angle_r": 8 + 26 * (0.5 + 0.5 * np.sin(w)),
        "knee_angle_l": 8 + 26 * (0.5 + 0.5 * np.sin(w + np.pi)),
        "ankle_angle_r": 6 * np.sin(2 * w),
        "ankle_angle_l": 6 * np.sin(2 * w + np.pi),
    }


def test_cycle_normalize_shapes():
    t, coords = _gait_coords(strides=6)
    norm, pct = cycle_normalize(t, coords)
    assert pct[0] == 0 and pct[-1] == 100 and len(pct) == 101
    assert "knee_angle" in norm and "r" in norm["knee_angle"]
    cyc = norm["knee_angle"]["r"]
    assert cyc.ndim == 2 and cyc.shape[1] == 101 and cyc.shape[0] >= 3  # several strides
    mean, sd = ensemble(cyc)
    assert mean.shape == (101,) and np.all(sd >= 0)


def test_too_few_cycles_yields_nothing():
    t = np.linspace(0, 1, 100)
    coords = {"hip_flexion_r": 12 + 20 * np.sin(2 * np.pi * 0.5 * t)}  # <2 strides
    norm, _ = cycle_normalize(t, coords)
    assert norm == {} or "hip_flexion" not in norm


def test_normative_band():
    m, sd = normative.band("knee_angle")
    assert m.shape == (101,) and sd.shape == (101,)
    assert m.max() > 40   # swing-phase knee flexion peak present
    assert normative.band("nonexistent_coord") is None


def test_gait_report_uses_cycle_plot(tmp_path):
    from gait_analysis.analysis.report import build_html_report
    t, coords = _gait_coords(strides=6)
    coords["pelvis_tilt"] = 10 + 2 * np.sin(2 * np.pi * 0.95 * t)
    cols = ["time", "pelvis_tilt"] + [k for k in coords if k != "pelvis_tilt"]
    rows = np.column_stack([t] + [coords[c] for c in cols[1:]])
    mot = tmp_path / "g.mot"
    mot.write_text("\n".join(["g", f"nColumns={len(cols)}", "inDegrees=yes", "endheader",
                              "\t".join(cols)] +
                             ["\t".join(f"{v:.4f}" for v in r) for r in rows]) + "\n")
    html = (tmp_path / "r.html")
    build_html_report(mot, html, gait_speed_m_s=1.2)
    text = html.read_text()
    assert "Gait-cycle kinematics" in text          # used the ensemble plot, not raw time series
    assert "ensemble mean" in text
