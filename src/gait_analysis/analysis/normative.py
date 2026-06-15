"""Representative adult sagittal-plane gait reference bands (% gait cycle, heel-strike = 0).

HONESTY: these are *approximate, representative* normal curves (mean +/- ~1SD) capturing the
well-established morphology of adult comfortable-speed walking (after Winter, *The Biomechanics
and Motor Control of Human Gait*, 1991; consistent with Fukuchi 2018). They are a generic visual
reference, NOT a population-matched normative database -- replace `BANDS_21` with age/speed-matched
data (e.g. the public Fukuchi 2018 dataset) before any clinical use. Only sagittal hip/knee/ankle
and pelvic tilt are provided; frontal/transverse normals for markerless data aren't trustworthy.
"""

from __future__ import annotations

import numpy as np

# 21 samples at 0,5,...,100 % gait cycle: (mean, sd) in degrees. +knee/hip = flexion, +ankle = dorsiflexion.
_PCT21 = np.linspace(0, 100, 21)
BANDS_21 = {
    "hip_flexion": (
        [32, 30, 27, 23, 18, 13, 8, 3, -2, -6, -8, -7, -2, 6, 15, 22, 27, 30, 32, 33, 32],
        [5] * 21),
    "knee_angle": (
        [5, 15, 18, 15, 10, 6, 4, 3, 4, 6, 9, 14, 22, 35, 50, 58, 55, 40, 22, 10, 5],
        [6] * 21),
    "ankle_angle": (
        [0, -4, -5, -2, 2, 5, 8, 10, 11, 10, 6, -3, -12, -15, -10, -4, -1, 0, 1, 0, 0],
        [4] * 21),
    "pelvis_tilt": (
        [10, 10, 11, 11, 10, 10, 9, 9, 10, 10, 11, 11, 10, 10, 9, 9, 10, 10, 11, 10, 10],
        [3] * 21),
}


def band(base: str, n_points: int = 101):
    """Return (mean, sd) for a coordinate base over n_points of % gait cycle, or None."""
    if base not in BANDS_21:
        return None
    mean21, sd21 = BANDS_21[base]
    grid = np.linspace(0, 100, n_points)
    return np.interp(grid, _PCT21, mean21), np.interp(grid, _PCT21, sd21)
