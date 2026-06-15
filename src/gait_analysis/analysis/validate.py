"""Concurrent-validity comparison of two OpenSim .mot files.

Use it to validate Track B (single-phone) against Track A (Pose2Sim/2-phone) on the
same subject, or any quick-vs-accurate comparison. Per shared coordinate it reports
RMSE, bias (mean offset), and Pearson correlation, time-normalized so trials of
different length/rate are comparable. Sagittal (bend/straighten) coordinates are the
ones held to the clinical target (~5 deg); frontal/transverse are reported but advisory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .kinematics import read_storage

SAGITTAL_TARGET_DEG = 5.0
_N = 101   # time-normalized samples (0..100%)


def _plane(coord: str) -> str:
    c = coord.lower()
    if any(k in c for k in ("rotation", "_rot", "pro_sup")):
        return "transverse"
    if any(k in c for k in ("adduction", "_add", "list", "bending")):
        return "frontal"
    return "sagittal"   # flexion/extension, knee_angle, ankle_angle, tilt, elbow_flex...


def _normalize(time: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Resample a series onto N points over its normalized 0..1 duration."""
    t = np.asarray(time, float)
    y = np.asarray(y, float)
    good = np.isfinite(y)
    if good.sum() < 2:
        return np.full(_N, np.nan)
    tn = (t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else np.linspace(0, 1, len(t))
    return np.interp(np.linspace(0, 1, _N), tn[good], y[good])


def compare(ref_mot, test_mot) -> dict:
    """Per-coordinate RMSE/bias/r between two .mot files, time-normalized."""
    rt, rc, _ = read_storage(ref_mot)
    tt, tc, _ = read_storage(test_mot)
    shared = sorted(set(rc) & set(tc))

    per = {}
    for name in shared:
        a = _normalize(rt, rc[name])
        b = _normalize(tt, tc[name])
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            continue
        a, b = a[m], b[m]
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        bias = float(np.mean(b - a))
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")
        per[name] = {"plane": _plane(name), "rmse": rmse, "bias": bias, "r": r}

    sag = [v["rmse"] for v in per.values() if v["plane"] == "sagittal"]
    summary = {
        "n_coordinates": len(per),
        "sagittal_rmse_mean": float(np.mean(sag)) if sag else float("nan"),
        "sagittal_rmse_max": float(np.max(sag)) if sag else float("nan"),
        "target_deg": SAGITTAL_TARGET_DEG,
        "pass": bool(sag) and float(np.mean(sag)) <= SAGITTAL_TARGET_DEG,
    }
    return {"per_coordinate": per, "summary": summary}


def format_report(result: dict) -> str:
    s = result["summary"]
    lines = ["=== Concurrent validity (test vs reference) ==="]
    lines.append(f"Shared coordinates: {s['n_coordinates']}  "
                 f"sagittal RMSE mean {s['sagittal_rmse_mean']:.1f} / max {s['sagittal_rmse_max']:.1f} deg "
                 f"(target <= {s['target_deg']:.0f})  -> {'PASS' if s['pass'] else 'FAIL'}")
    lines.append("")
    lines.append(f"{'coordinate':<22}{'plane':>11}{'rmse':>8}{'bias':>8}{'r':>7}")
    for name in sorted(result["per_coordinate"], key=lambda n: result["per_coordinate"][n]["rmse"], reverse=True):
        v = result["per_coordinate"][name]
        lines.append(f"{name:<22}{v['plane']:>11}{v['rmse']:>8.1f}{v['bias']:>8.1f}{v['r']:>7.2f}")
    lines.append("")
    lines.append("Sagittal RMSE is the headline; frontal/transverse are advisory (markerless-weak).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Concurrent validity of two OpenSim .mot files")
    ap.add_argument("--ref", required=True, help="Reference .mot (e.g. Track A / Pose2Sim)")
    ap.add_argument("--test", required=True, help="Test .mot (e.g. Track B / single-phone)")
    args = ap.parse_args(argv)
    print(format_report(compare(args.ref, args.test)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
