"""Tests for the clinical-signature rule engine (offline)."""

from gait_analysis.analysis.signatures import Context, detect, format_findings


def _ids(findings):
    return {f.rule_id for f in findings}


def test_flags_multiple_patterns():
    summary = {
        "in_degrees": True,
        "rom": {
            "knee_angle_r": {"min": 5, "max": 35, "range": 30, "mean": 20},   # stiff-knee + low excursion
            "knee_angle_l": {"min": 35, "max": 70, "range": 35, "mean": 50},  # crouch + low excursion
            "hip_flexion_r": {"min": 5, "max": 30, "range": 25, "mean": 15},  # no hip extension
            "ankle_angle_r": {"min": -20, "max": 2, "range": 22, "mean": -5}, # foot drop
        },
        "symmetry_LR": {"hip_flexion": 0.5, "knee_angle": 1.0},               # hip asymmetry
    }
    ids = _ids(detect(summary))
    assert "stiff_knee_swing" in ids
    assert "crouch_knee" in ids
    assert "reduced_hip_extension" in ids
    assert "foot_drop" in ids
    assert "reduced_knee_excursion" in ids
    assert "rom_asymmetry" in ids


def test_clean_gait_triggers_nothing():
    summary = {
        "in_degrees": True,
        "rom": {
            "knee_angle_r": {"min": 3, "max": 62, "range": 59, "mean": 30},
            "knee_angle_l": {"min": 3, "max": 62, "range": 59, "mean": 30},
            "hip_flexion_r": {"min": -10, "max": 30, "range": 40, "mean": 10},
            "hip_flexion_l": {"min": -10, "max": 30, "range": 40, "mean": 10},
            "ankle_angle_r": {"min": -18, "max": 11, "range": 29, "mean": 0},
            "ankle_angle_l": {"min": -18, "max": 11, "range": 29, "mean": 0},
        },
        "symmetry_LR": {"hip_flexion": 1.0, "knee_angle": 1.0},
    }
    assert detect(summary) == []


def test_format_includes_global_caveats():
    summary = {"in_degrees": True, "rom": {}, "symmetry_LR": {}}
    text = format_findings(detect(summary), Context(gait_speed_m_s=1.1))
    assert "NOT diagnosis" in text
    assert "speed" in text.lower()
    assert "1.10 m/s" in text
