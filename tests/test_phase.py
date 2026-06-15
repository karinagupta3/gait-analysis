"""Tests for gait-cycle phase features and phase-aware signature flags."""

import numpy as np

from gait_analysis.analysis.gait_cycle import PhaseFeatures, compute_phase_features
from gait_analysis.analysis.signatures import Context, detect


def _ids(findings):
    return {f.rule_id for f in findings}


def test_compute_phase_features_counts_cycles():
    fps = 100.0
    t = np.arange(0, 4.0, 1 / fps)            # 4 s
    f = 1.0                                    # 1 Hz -> ~4 cycles
    coords = {
        "hip_flexion_r": 20 * np.sin(2 * np.pi * f * t),
        "hip_flexion_l": 20 * np.sin(2 * np.pi * f * t),
        "knee_angle_r": 30 + 30 * np.sin(2 * np.pi * f * t),
        "knee_angle_l": 30 + 30 * np.sin(2 * np.pi * f * t),
        "ankle_angle_r": 10 * np.sin(2 * np.pi * f * t),
        "ankle_angle_l": 10 * np.sin(2 * np.pi * f * t),
    }
    feat = compute_phase_features(t, coords)
    assert feat.n_cycles >= 3                  # ~4 strides in 4 s
    assert "r" in feat.peak_swing_knee_flexion
    assert "r" in feat.terminal_stance_hip_ext


def test_phase_rules_fire_per_window():
    phase = PhaseFeatures(
        n_cycles=3,
        peak_swing_knee_flexion={"r": 35.0, "l": 62.0},   # r stiff, l normal
        terminal_stance_hip_ext={"r": 5.0},                # r no extension
        swing_dorsiflexion={"r": 2.0},                     # r foot drop
        stance_min_knee={"r": 35.0},                       # r crouch
    )
    summary = {"in_degrees": True, "rom": {}, "symmetry_LR": {}}
    ids = _ids(detect(summary, Context(phase=phase)))
    assert "stiff_knee_swing" in ids
    assert "reduced_hip_extension" in ids
    assert "foot_drop" in ids
    assert "crouch_knee" in ids


def test_short_trial_suppresses_asymmetry():
    phase = PhaseFeatures(n_cycles=1, peak_swing_knee_flexion={"r": 35.0})
    summary = {"in_degrees": True, "rom": {}, "symmetry_LR": {"hip_flexion": 0.5}}
    ids = _ids(detect(summary, Context(phase=phase)))
    assert "stiff_knee_swing" in ids           # phase rule still fires
    assert "rom_asymmetry" not in ids          # 1 cycle -> asymmetry suppressed


def test_asymmetry_emitted_with_enough_cycles():
    phase = PhaseFeatures(n_cycles=3)
    summary = {"in_degrees": True, "rom": {}, "symmetry_LR": {"hip_flexion": 0.5}}
    assert "rom_asymmetry" in _ids(detect(summary, Context(phase=phase)))


def test_no_phase_uses_global_rules():
    summary = {
        "in_degrees": True,
        "rom": {"knee_angle_r": {"min": 5, "max": 35, "range": 30, "mean": 20}},
        "symmetry_LR": {},
    }
    ids = _ids(detect(summary, Context()))     # no phase -> global fallback
    assert "stiff_knee_swing" in ids
