"""Single-camera 2D metrics for REP-BASED clinical movements: squat and
sit-to-stand. Gait has its own module (gait_metrics_2d); these share the same
honest 2D framing but measure cyclic flexion/extension instead of strides.

What it computes (side view, camera-facing leg):
  * rep detection (descent->ascent cycles) from the smoothed knee-flexion signal
  * per-rep peak knee flexion (depth), peak hip flexion, peak trunk lean
  * tempo: time per rep; squat descent/ascent; sit-to-stand rise time + the
    5x sit-to-stand total time (a standard clinical test)
  * left/right symmetry of peak knee flexion (experimental — far leg noisier)

HONESTY: single camera, sagittal plane. Reliable for flexion depth, timing, and
rep counting from a clean side view; NOT for frontal-plane knee valgus (that needs
a front view + is low-confidence in 2D). Screening estimate, not a diagnosis.
"""
from __future__ import annotations

import numpy as np

from ..pose.mediapipe3d import BLAZEPOSE_33
from .gait_metrics_2d import _savgol, _side_angles
from .sagittal2d import valid_frame_mask

_IDX = {name: i for i, name in enumerate(BLAZEPOSE_33)}

# Reference values (general clinical literature; refine with cited sources later).
# 5x sit-to-stand age norms (mean seconds) — Bohannon, Percept Mot Skills 2006.
STS_5X_NORMS = {"(<60)": 10.0, "60-69": 11.4, "70-79": 12.6, "80-89": 14.8}
STS_5X_FALLRISK_S = 15.0       # >~15 s widely cited as elevated fall risk
SQUAT_PARALLEL_KNEE = (100, 130)   # deg knee flexion at a parallel/deep squat bottom


def _trunk_series(px, vis, valid):
    sh = (px[:, _IDX["left_shoulder"]] + px[:, _IDX["right_shoulder"]]) / 2
    hipm = (px[:, _IDX["left_hip"]] + px[:, _IDX["right_hip"]]) / 2
    v = np.minimum(
        np.minimum(vis[:, _IDX["left_shoulder"]], vis[:, _IDX["right_shoulder"]]),
        np.minimum(vis[:, _IDX["left_hip"]], vis[:, _IDX["right_hip"]]))
    trunk = sh - hipm
    ang = np.degrees(np.arctan2(trunk[:, 0], -trunk[:, 1]))   # 0 = upright, + = lean
    return np.where(valid & (v >= 0.5), np.abs(ang), np.nan)


def _detect_reps(knee_flex, fps):
    """Rep bottoms = knee-flexion maxima; returns (bottom_idx, top_idx_between)."""
    from scipy.signal import find_peaks
    y = knee_flex.copy()
    bad = ~np.isfinite(y)
    if bad.all():
        return [], []
    y[bad] = np.nanmin(y[~bad])
    sep = max(1, int(0.5 * fps))
    # a real squat/sit excursion is large; require prominence to ignore wobble
    bottoms, _ = find_peaks(y, distance=sep, prominence=20)
    tops, _ = find_peaks(-y, distance=sep, prominence=20)
    return bottoms.tolist(), tops.tolist()


def _sym(a, b):
    if a is None or b is None or max(abs(a), abs(b)) < 1e-6:
        return None
    return round(100.0 * (1 - abs(a - b) / max(abs(a), abs(b))), 0)


def compute_movement_metrics(image_landmarks, visibility, width, height, fps, task) -> dict:
    fps = float(fps) or 30.0
    px = image_landmarks.astype(float) * np.array([width, height])
    vis = visibility.astype(float)

    valids = {s: valid_frame_mask(image_landmarks, vis, side=s) for s in ("right", "left")}
    primary = "right" if valids["right"].sum() >= valids["left"].sum() else "left"
    other = "left" if primary == "right" else "right"

    kf, hf, nvalid = _side_angles(px, vis, primary, valids[primary])
    kf, hf = _savgol(kf, fps), _savgol(hf, fps)
    okf, _, _ = _side_angles(px, vis, other, valids[other])
    okf = _savgol(okf, fps)
    trunk = _savgol(_trunk_series(px, vis, valids[primary]), fps)

    bottoms, tops = _detect_reps(kf, fps)
    per_rep = []
    for a, b in zip(bottoms[:-1], bottoms[1:]):     # one rep = bottom-to-bottom
        seg_k, seg_h, seg_t = kf[a:b + 1], hf[a:b + 1], trunk[a:b + 1]
        rep = {"knee_peak": _nanmax(seg_k), "hip_peak": _nanmax(seg_h),
               "trunk_peak": _nanmax(seg_t), "duration_s": round((b - a) / fps, 2)}
        per_rep.append(rep)
    # peak at each detected bottom (more robust count than bottom-to-bottom windows)
    knee_peaks = [v for v in (_at(kf, i) for i in bottoms) if v is not None]
    hip_peaks = [v for v in (_at(hf, i) for i in bottoms) if v is not None]
    trunk_peaks = [v for v in (_at(trunk, i) for i in bottoms) if v is not None]
    okf_peaks = [v for v in (_at(okf, i) for i in bottoms) if v is not None]

    n_reps = len(bottoms)
    out = {
        "task": task, "fps": round(fps, 1),
        "frames_total": int(len(image_landmarks)),
        "frames_used": int(nvalid), "primary_side": primary,
        "n_reps": n_reps,
        "knee_peak_mean": _mean(knee_peaks), "knee_peak_sd": _sd(knee_peaks),
        "hip_peak_mean": _mean(hip_peaks),
        "trunk_peak_mean": _mean(trunk_peaks),
        "symmetry_knee_pct": _sym(_mean(knee_peaks), _mean(okf_peaks)),
        "per_rep": per_rep,
        "_series": {"knee": kf, "hip": hf, "trunk": trunk},
        "_bottoms": bottoms, "_tops": tops,
    }

    # Timing: total active time = first to last bottom; mean rep time.
    if n_reps >= 2:
        total = (bottoms[-1] - bottoms[0]) / fps
        out["total_time_s"] = round(total, 2)
        out["time_per_rep_s"] = round(total / (n_reps - 1), 2)

    if task == "sit_to_stand":
        # 5x STS: time for 5 full rises. Use the time spanning 5 cycles when present.
        if n_reps >= 6:                     # 6 bottoms bound 5 complete cycles
            out["sts_5x_time_s"] = round((bottoms[5] - bottoms[0]) / fps, 1)
        elif n_reps >= 2:
            # extrapolate per-rep time to 5 reps (clearly labeled as estimated)
            out["sts_5x_time_s_est"] = round(out["time_per_rep_s"] * 5, 1)
    elif task == "squat":
        km = out["knee_peak_mean"]
        if km is not None:
            out["depth_class"] = ("deep / past parallel" if km >= SQUAT_PARALLEL_KNEE[0]
                                  else "partial (above parallel)" if km >= 70
                                  else "shallow")
    return out


def _nanmax(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.max(a)) if a.size else None


def _at(series, i):
    v = series[i]
    return float(v) if np.isfinite(v) else None


def _mean(xs):
    return round(float(np.mean(xs)), 0) if xs else None


def _sd(xs):
    return round(float(np.std(xs)), 0) if xs else None
