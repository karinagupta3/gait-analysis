"""Compare an OpenSim .mot against a ground-truth .mot: per-DOF RMSE / MAE / r.

The reusable core of the ground-truth benchmark. Given our pipeline's joint-angle
output and a reference (e.g. OpenCap LabValidation's Vicon->OpenSim IK), it:
  1. parses both .mot/.sto files,
  2. resamples to a common time grid over their overlap,
  3. corrects a constant time lag by cross-correlating a reference DOF,
  4. reports, per shared coordinate: RMSE, MAE, Pearson r, and OFFSET-REMOVED RMSE
     (markerset differences cause constant offsets that aren't a tracking error).

Usage: compare_mot(pred_path, truth_path) -> dict of per-coordinate stats.
Validate the logic by comparing any two of our own .mot files before real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_mot(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lines = Path(path).read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("time"))
    cols = lines[hdr].split("\t") if "\t" in lines[hdr] else lines[hdr].split()
    rows = []
    for l in lines[hdr + 1:]:
        if not l.strip():
            continue
        parts = l.split("\t") if "\t" in l else l.split()
        rows.append([float(x) for x in parts])
    data = np.asarray(rows)
    t = data[:, 0]
    return t, {c: data[:, i] for i, c in enumerate(cols) if i > 0}


def _resample(t, y, grid):
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full(grid.shape, np.nan)
    return np.interp(grid, t[ok], y[ok])


def _best_lag(a, b, max_lag):
    """Integer-sample lag (apply to b) maximizing correlation with a."""
    a = a - np.nanmean(a); b = b - np.nanmean(b)
    best, bl = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        bb = np.roll(b, lag)
        sl = slice(max_lag, -max_lag) if max_lag else slice(None)
        c = np.nansum(a[sl] * bb[sl])
        if c > best:
            best, bl = c, lag
    return bl


def compare_mot(pred_path, truth_path, ref="knee_angle_r", dt=0.02,
                coords=None) -> dict:
    tp, P = read_mot(pred_path)
    tt, T = read_mot(truth_path)
    lo, hi = max(tp.min(), tt.min()), min(tp.max(), tt.max())
    if hi - lo < 0.3:
        raise ValueError(f"insufficient time overlap: [{lo:.2f},{hi:.2f}]")
    grid = np.arange(lo, hi, dt)
    shared = [c for c in P if c in T]
    if coords:
        shared = [c for c in shared if c in coords]

    Pr = {c: _resample(tp, P[c], grid) for c in shared}
    Tr = {c: _resample(tt, T[c], grid) for c in shared}

    # de-lag using the reference DOF if present (handles a start-time offset)
    lag = 0
    if ref in Pr and np.isfinite(Pr[ref]).any() and np.isfinite(Tr[ref]).any():
        lag = _best_lag(Tr[ref], Pr[ref], max_lag=min(50, len(grid) // 4))
        Pr = {c: np.roll(v, lag) for c, v in Pr.items()}

    out = {"_lag_s": lag * dt, "_overlap_s": hi - lo, "_n": len(grid), "coords": {}}
    for c in shared:
        a, b = Tr[c], Pr[c]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5:
            continue
        a, b = a[m], b[m]
        err = b - a
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        rmse_off = float(np.sqrt(np.mean((err - err.mean()) ** 2)))  # constant-offset removed
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else float("nan")
        out["coords"][c] = dict(rmse=rmse, mae=mae, rmse_offset_removed=rmse_off, r=r)
    return out


def print_report(res: dict, title=""):
    print(f"\n=== {title} ===  (lag {res['_lag_s']:+.2f}s, overlap {res['_overlap_s']:.1f}s)")
    print(f"{'coordinate':24s} {'RMSE':>7} {'MAE':>7} {'RMSE-off':>8} {'r':>6}")
    sag = ["pelvis_tilt", "hip_flexion_r", "hip_flexion_l", "knee_angle_r",
           "knee_angle_l", "ankle_angle_r", "ankle_angle_l"]
    items = sorted(res["coords"].items(), key=lambda kv: (kv[0] not in sag, kv[0]))
    rmses = []
    for c, s in items:
        tag = "  <-sagittal" if c in sag else ""
        print(f"{c:24s} {s['rmse']:7.1f} {s['mae']:7.1f} {s['rmse_offset_removed']:8.1f} {s['r']:6.2f}{tag}")
        if c in sag:
            rmses.append(s["rmse"])
    if rmses:
        print(f"{'-- sagittal mean RMSE --':24s} {np.mean(rmses):7.1f}")


if __name__ == "__main__":
    import sys
    res = compare_mot(sys.argv[1], sys.argv[2])
    print_report(res, f"{Path(sys.argv[1]).parent.name} vs {Path(sys.argv[2]).parent.name}")
