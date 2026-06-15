"""Gait-cycle segmentation + phase-specific features from OpenSim kinematics.

The signature rules in their first form used GLOBAL min/max over the whole clip,
which throws false flags on short/noisy trials (see docs/04). Clinically, the value
that matters is phase-specific: peak knee flexion *in swing*, hip extension *at
terminal stance*, ankle dorsiflexion *in swing*. This module finds the gait cycles
and extracts those windowed features.

Event detection here is ANGLE-ONLY (no GRF, no marker positions), so it runs from a
.mot alone: heel strike ~ peak hip flexion, toe-off ~ peak hip extension (a standard
kinematic approximation -- Zeni's foot-position method or GRF is more accurate when
available). We report the number of detected cycles so downstream rules can lower
confidence when the trial is too short to trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks


@dataclass
class PhaseFeatures:
    """Per-side, phase-windowed kinematic features (degrees). NaN if unavailable."""
    n_cycles: int = 0
    peak_swing_knee_flexion: dict[str, float] = field(default_factory=dict)   # max knee in swing
    terminal_stance_hip_ext: dict[str, float] = field(default_factory=dict)   # min hip_flexion in stance
    swing_dorsiflexion: dict[str, float] = field(default_factory=dict)        # max ankle in swing
    stance_min_knee: dict[str, float] = field(default_factory=dict)           # min knee in stance (crouch)


def _events_from_hip(hip_flexion: np.ndarray, min_sep: int):
    """Heel strikes ~ hip-flexion maxima; toe-offs ~ hip-flexion minima."""
    hs, _ = find_peaks(hip_flexion, distance=min_sep)
    to, _ = find_peaks(-hip_flexion, distance=min_sep)
    return hs, to


def _stance_swing_windows(hs: np.ndarray, to: np.ndarray):
    """Pair events into (stance: HS->next TO) and (swing: TO->next HS) index windows."""
    stance, swing = [], []
    for h in hs:
        later_to = to[to > h]
        if later_to.size:
            stance.append((h, int(later_to[0])))
    for t in to:
        later_hs = hs[hs > t]
        if later_hs.size:
            swing.append((t, int(later_hs[0])))
    return stance, swing


def compute_phase_features(time: np.ndarray, coords: dict[str, np.ndarray],
                           min_event_sep_s: float = 0.4) -> PhaseFeatures:
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0 / 60
    min_sep = max(1, int(min_event_sep_s / dt))
    feat = PhaseFeatures()
    cycle_counts = []

    for side in ("r", "l"):
        hip = coords.get(f"hip_flexion_{side}")
        knee = coords.get(f"knee_angle_{side}")
        ankle = coords.get(f"ankle_angle_{side}")
        if hip is None:
            continue
        hs, to = _events_from_hip(hip, min_sep)
        stance, swing = _stance_swing_windows(hs, to)
        cycle_counts.append(max(len(stance), len(swing)))

        if knee is not None and swing:
            feat.peak_swing_knee_flexion[side] = float(
                np.mean([np.max(knee[a:b + 1]) for a, b in swing]))
        if knee is not None and stance:
            feat.stance_min_knee[side] = float(
                np.mean([np.min(knee[a:b + 1]) for a, b in stance]))
        if stance:
            feat.terminal_stance_hip_ext[side] = float(
                np.mean([np.min(hip[a:b + 1]) for a, b in stance]))
        if ankle is not None and swing:
            feat.swing_dorsiflexion[side] = float(
                np.mean([np.max(ankle[a:b + 1]) for a, b in swing]))

    feat.n_cycles = min(cycle_counts) if cycle_counts else 0
    return feat
