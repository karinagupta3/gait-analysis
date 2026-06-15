"""Tests for task detection, squat/STS metrics, and the treatment knowledge base."""

import numpy as np

from gait_analysis.analysis import interpretation, tasks


def _ids(findings):
    return {f.rule_id for f in findings}


def test_detect_task_gait_vs_squat():
    t = np.linspace(0, 4, 400)
    w = 2 * np.pi * 1.0 * t
    gait = {"knee_angle_r": 30 + 30 * np.sin(w), "knee_angle_l": 30 + 30 * np.sin(w + np.pi)}
    squat = {"knee_angle_r": 30 + 30 * np.sin(w), "knee_angle_l": 30 + 30 * np.sin(w)}
    assert tasks.detect_task(gait) == "gait"      # knees antiphase
    assert tasks.detect_task(squat) == "squat"    # knees in phase


def test_squat_metrics_and_flags():
    t = np.linspace(0, 6, 600)
    w = 2 * np.pi * 0.5 * t                        # 3 reps
    coords = {
        "knee_angle_r": 30 - 20 * np.cos(w),       # 10..50 (limited depth)
        "knee_angle_l": 35 - 25 * np.cos(w),       # 10..60 -> L/R asymmetry
        "hip_flexion_r": 30 - 25 * np.cos(w),
        "hip_flexion_l": 30 - 25 * np.cos(w),
        "hip_adduction_r": 6 - 8 * np.cos(w),      # peak ~14 -> dynamic valgus
        "hip_adduction_l": 2 - 3 * np.cos(w),
        "ankle_angle_r": 15 - 5 * np.cos(w),
        "ankle_angle_l": 15 - 5 * np.cos(w),
        "pelvis_tilt": -8 + 4 * np.sin(w),
    }
    task, metrics, findings = tasks.analyze_task(t, coords, "squat")
    assert metrics["n_reps"] >= 2
    ids = _ids(findings)
    assert "squat_depth_limited" in ids
    assert "dynamic_valgus" in ids
    assert "squat_asymmetry" in ids


def test_route_dispatches_by_task():
    t = np.linspace(0, 4, 400)
    w = 2 * np.pi * 1.0 * t
    squat = {"knee_angle_r": 30 - 20 * np.cos(w), "knee_angle_l": 30 - 20 * np.cos(w)}
    task, findings, metrics = tasks.route(t, squat, {"rom": {}, "symmetry_LR": {}}, None)
    assert task == "squat"


def test_treatment_kb_covers_key_rules():
    for rid in ("stiff_knee_swing", "reduced_hip_extension", "foot_drop",
                "dynamic_valgus", "squat_depth_limited", "ankle_df_restriction",
                "butt_wink", "sts_slow"):
        g = interpretation.guidance_for(rid)
        assert g is not None and g.meaning and g.treatment and g.tracking
