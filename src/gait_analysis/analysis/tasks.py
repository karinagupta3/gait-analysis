"""Task-aware analysis: squat and sit-to-stand (NOT gait).

Gait rules (heel-strike/swing/stance) are wrong for a squat. This module detects the
task, segments reps, and extracts task-specific metrics + flags with the thresholds
from the squat/STS literature (see docs/07): squat depth (~95 deg hip / ~40 deg ankle
DF for a flat-heel deep squat), "butt wink" (pelvis reversing toward posterior tilt
near max depth), dynamic valgus (rising hip_adduction; FPPA >=10 deg increase flags
it), and L/R symmetry; 5x sit-to-stand timing (>15 s = fall risk).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .signatures import CONF_HIGH, CONF_LOW, CONF_MOD, Finding
from .signatures import detect as detect_gait


def detect_task(coords: dict[str, np.ndarray]) -> str:
    """Gait vs a bilateral task (squat/STS) from L/R knee-flexion phase.

    Walking: knees move out of phase (negative correlation). Squat/STS: both knees
    flex/extend together (positive correlation). STS vs squat is left to the caller.
    """
    kr, kl = coords.get("knee_angle_r"), coords.get("knee_angle_l")
    if kr is None or kl is None or len(kr) < 4:
        return "gait"
    kr, kl = np.asarray(kr, float), np.asarray(kl, float)
    good = np.isfinite(kr) & np.isfinite(kl)
    if good.sum() < 4 or kr[good].std() < 1e-6 or kl[good].std() < 1e-6:
        return "gait"
    c = np.corrcoef(kr[good], kl[good])[0, 1]
    if not np.isfinite(c):
        return "gait"
    return "squat" if c > 0.3 else "gait"


def _mean_knee(coords):
    parts = [coords[k] for k in ("knee_angle_r", "knee_angle_l") if k in coords]
    return np.nanmean(np.stack(parts), axis=0) if parts else None


@dataclass
class Rep:
    start: int
    bottom: int
    end: int
    metrics: dict = field(default_factory=dict)


def detect_reps(time, coords, min_sep_s=0.8, prominence=15.0) -> list[Rep]:
    """Segment squat/STS reps from the mean knee-flexion signal (bottom = peak flexion)."""
    knee = _mean_knee(coords)
    if knee is None:
        return []
    knee = knee.astype(float)
    knee = np.where(np.isfinite(knee), knee, np.nanmedian(knee[np.isfinite(knee)]))
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1 / 60
    sep = max(1, int(min_sep_s / dt))
    bottoms, _ = find_peaks(knee, distance=sep, prominence=prominence)
    troughs, _ = find_peaks(-knee, distance=sep)
    reps = []
    for b in bottoms:
        before = troughs[troughs < b]
        after = troughs[troughs > b]
        start = int(before[-1]) if before.size else 0
        end = int(after[0]) if after.size else len(knee) - 1
        reps.append(Rep(start, int(b), end))
    return reps


def squat_metrics(time, coords) -> dict:
    """Per-rep squat metrics aggregated across reps."""
    reps = detect_reps(time, coords)
    if not reps:
        return {"n_reps": 0}

    def side_peak(name, lo, hi, fn=np.max):
        vals = []
        for r in reps:
            arr = coords.get(name)
            if arr is None:
                return None
            seg = np.asarray(arr[lo(r):hi(r) + 1], float)
            seg = seg[np.isfinite(seg)]
            if seg.size:
                vals.append(fn(seg))
        return float(np.mean(vals)) if vals else None

    full = (lambda r: r.start, lambda r: r.end)
    desc = (lambda r: r.start, lambda r: r.bottom)

    out = {"n_reps": len(reps)}
    for s in ("r", "l"):
        out[f"peak_knee_flexion_{s}"] = side_peak(f"knee_angle_{s}", *full)
        out[f"peak_hip_flexion_{s}"] = side_peak(f"hip_flexion_{s}", *full)
        out[f"peak_ankle_df_{s}"] = side_peak(f"ankle_angle_{s}", *full)
        out[f"peak_hip_adduction_{s}"] = side_peak(f"hip_adduction_{s}", *desc)

    # Butt wink: pelvis tilt reverses toward posterior near the bottom -> the anterior
    # peak during descent minus the value at the bottom (assumes anterior tilt positive).
    pt = coords.get("pelvis_tilt")
    if pt is not None:
        winks = []
        for r in reps:
            seg = np.asarray(pt[r.start:r.bottom + 1], float)
            seg = seg[np.isfinite(seg)]
            if seg.size:
                winks.append(float(np.max(seg) - pt[r.bottom]))
        out["butt_wink_posterior_excursion"] = float(np.mean(winks)) if winks else None
    return out


def squat_flags(metrics: dict) -> list[Finding]:
    out = []
    if not metrics or metrics.get("n_reps", 0) == 0:
        return out

    knees = [metrics.get("peak_knee_flexion_r"), metrics.get("peak_knee_flexion_l")]
    knees = [k for k in knees if k is not None]
    if knees:
        depth = float(np.mean(knees))
        band = ("deep" if depth >= 100 else "~parallel" if depth >= 70
                else "half" if depth >= 50 else "shallow/limited")
        if depth < 70:
            out.append(Finding(
                "squat_depth_limited", "mobility",
                f"Limited squat depth (peak knee flexion ~{depth:.0f} deg, {band})",
                "knee_angle (peak)", round(depth, 1), ">=~90 for parallel, >=100 for deep",
                ["ankle dorsiflexion restriction (most common limiter)",
                 "hip flexion ROM / impingement (FAI)", "strength or motor control / fear"],
                CONF_MOD, ["A flat-heel deep squat needs ~95 deg hip flexion + ~40 deg ankle DF."]))

    for s, name in (("r", "right"), ("l", "left")):
        va = metrics.get(f"peak_hip_adduction_{s}")
        if va is not None and va > 10:
            out.append(Finding(
                "dynamic_valgus", "motor_control",
                f"Dynamic knee valgus / hip adduction in descent ({name})",
                f"hip_adduction_{s} (peak)", round(va, 1), ">~10 deg (normal DKV ~5)",
                ["hip-abductor / external-rotator weakness (glute med/max)",
                 "overactive adductors/TFL", "ankle DF limit / foot pronation"],
                CONF_MOD, ["Frontal-plane angle -- lower markerless confidence; links to ACL/PFP risk."]))
        df = metrics.get(f"peak_ankle_df_{s}")
        if df is not None and df < 10:
            out.append(Finding(
                "ankle_df_restriction", "mobility",
                f"Reduced ankle dorsiflexion in the squat ({name})",
                f"ankle_angle_{s} (peak)", round(df, 1), "<~10 deg",
                ["calf (gastroc/soleus) tightness", "talocrural joint restriction"],
                CONF_MOD, ["Confirm with a knee-to-wall test; distinguish joint vs muscle limit."]))

    wink = metrics.get("butt_wink_posterior_excursion")
    if wink is not None and wink > 8:
        out.append(Finding(
            "butt_wink", "mobility",
            f"'Butt wink' -- pelvis reverses toward posterior tilt near depth (~{wink:.0f} deg)",
            "pelvis_tilt (descent->bottom)", round(wink, 1), ">~8 deg reversal",
            ["ankle dorsiflexion limit", "hip morphology / end-range hip flexion (FAI)",
             "motor control (NOT usually hamstrings)"],
            CONF_LOW, ["Sign convention is model-dependent -- verify anterior=positive. "
                       "Concern is load-dependent; benign at bodyweight."]))

    kr, kl = metrics.get("peak_knee_flexion_r"), metrics.get("peak_knee_flexion_l")
    if kr and kl and max(kr, kl) > 0:
        diff = abs(kr - kl) / (0.5 * (kr + kl))
        if diff > 0.15:
            out.append(Finding(
                "squat_asymmetry", "asymmetry",
                f"Left/right squat-depth asymmetry ({diff * 100:.0f}%)",
                "knee_angle peak L vs R", round(diff, 3), ">15% difference",
                ["weight-shift away from a painful or weak limb", "unilateral mobility limit"],
                CONF_MOD, ["The 15% cut-point is borrowed from hop/strength testing; treat as a hint."]))
    return out


# --- sit-to-stand -----------------------------------------------------------

def sts_metrics(time, coords) -> dict:
    """Rise times from STS reps (sit = peak knee flexion, stand = trough)."""
    reps = detect_reps(time, coords, min_sep_s=1.0, prominence=20.0)
    if not reps:
        return {"n_rises": 0}
    rise_times = [(r.end - r.bottom) * float(np.median(np.diff(time))) for r in reps]
    return {"n_rises": len(reps), "mean_rise_time_s": float(np.mean(rise_times)),
            "total_time_s": float(time[-1] - time[0])}


def sts_flags(metrics: dict) -> list[Finding]:
    out = []
    n = metrics.get("n_rises", 0)
    if n >= 5 and metrics.get("total_time_s"):
        t5 = metrics["total_time_s"]
        if t5 > 15:
            out.append(Finding(
                "sts_slow", "weakness",
                f"Slow 5x sit-to-stand (~{t5:.1f} s)",
                "5xSTS time", round(t5, 1), ">15 s = elevated fall risk",
                ["lower-limb (quadriceps/glute) weakness", "balance/postural-control deficit"],
                CONF_MOD, ["Age norms: 60s ~11.4s, 70s ~12.6s, 80s ~14.8s; >12s warrants assessment."]))
    return out


def analyze_task(time, coords, task: str | None = None) -> tuple[str, dict, list[Finding]]:
    """Dispatch: returns (task, metrics, findings). task auto-detected if None."""
    task = task or detect_task(coords)
    if task == "squat":
        m = squat_metrics(time, coords)
        return task, m, squat_flags(m)
    if task == "sts":
        m = sts_metrics(time, coords)
        return task, m, sts_flags(m)
    return "gait", {}, []   # gait handled by signatures.py phase rules


def route(time, coords, summary, ctx, task: str | None = None):
    """Pick task (auto unless given) and return (task, findings, metrics).

    Gait -> phase-gated signature rules; squat/STS -> task-specific flags.
    """
    task = task or detect_task(coords)
    if task == "gait":
        return "gait", detect_gait(summary, ctx), {}
    t, metrics, findings = analyze_task(time, coords, task)
    return t, findings, metrics
