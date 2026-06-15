"""Metric spatiotemporal gait parameters from a 3D marker .trc.

Unlike spatiotemporal.py (2D pixel space, temporal only), a .trc carries 3D marker
positions in METRES (from Pose2Sim triangulation or scaled monocular), so true spatial
parameters are valid: step/stride LENGTH, walking speed, step width.

Gait events use Zeni et al. (2008): heel strike = foot marker most anterior relative to
the pelvis along the walking direction; toe-off = most posterior. Vertical axis is taken
as the marker axis with the largest mean height; the horizontal axis with the larger
pelvis travel is "forward", the remaining horizontal axis is "lateral".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

# Marker-name candidates (covers HALPE_26 / Pose2Sim and our blazepose markerset).
_CANDS = {
    "RHip": ["RHip", "hip_r", "RASI", "R_Hip"],
    "LHip": ["LHip", "hip_l", "LASI", "L_Hip"],
    "RFoot": ["RHeel", "RAnkle", "RBigToe", "ankle_r", "calc_r"],
    "LFoot": ["LHeel", "LAnkle", "LBigToe", "ankle_l", "calc_l"],
}


def read_trc(path: str | Path):
    """Return (times (T,), names list, positions (T, M, 3) in file units, with NaN gaps)."""
    lines = Path(path).read_text().splitlines()
    name_row = lines[3].split("\t")
    names = [name_row[i].strip() for i in range(2, len(name_row), 3) if name_row[i].strip()]
    times, rows = [], []
    for ln in lines[4:]:
        parts = ln.split("\t")
        if not parts or not parts[0].strip().isdigit():
            continue
        times.append(float(parts[1]))
        rows.append(parts)
    T, M = len(rows), len(names)
    pos = np.full((T, M, 3), np.nan)
    for f, parts in enumerate(rows):
        for m in range(M):
            for a in range(3):
                v = parts[2 + 3 * m + a] if 2 + 3 * m + a < len(parts) else ""
                if v.strip():
                    pos[f, m, a] = float(v)
    return np.asarray(times, float), names, pos


def _find(names, key):
    for cand in _CANDS[key]:
        if cand in names:
            return names.index(cand)
    return None


def _interp(x):
    x = x.copy()
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() >= 2:
        x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def compute(trc_path, min_event_sep_s: float = 0.4) -> dict:
    times, names, pos = read_trc(trc_path)
    out: dict = {"available": False, "_note": ""}
    idx = {k: _find(names, k) for k in _CANDS}
    if idx["RHip"] is None or idx["LHip"] is None or idx["RFoot"] is None or idx["LFoot"] is None:
        out["_note"] = "needs hip + foot markers (RHip/LHip + R/L heel or ankle)"
        return out

    pelvis = np.nanmean(np.stack([pos[:, idx["RHip"]], pos[:, idx["LHip"]]]), axis=0)  # (T,3)
    pelvis = np.column_stack([_interp(pelvis[:, a]) for a in range(3)])
    # Vertical = the axis where hips sit consistently ABOVE feet (robust to forward drift).
    feet_all = np.concatenate([pos[:, idx["RFoot"]], pos[:, idx["LFoot"]]], axis=0)
    vert = int(np.argmax(np.nanmean(pelvis, axis=0) - np.nanmean(feet_all, axis=0)))
    horiz = [a for a in range(3) if a != vert]
    fwd = horiz[int(np.argmax([np.ptp(pelvis[:, a]) for a in horiz]))]
    lat = horiz[1 - horiz.index(fwd)]

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 60
    min_sep = max(1, int(min_event_sep_s / dt))
    direction = np.sign(pelvis[-1, fwd] - pelvis[0, fwd]) or 1.0

    ev = {}
    for side, fkey in (("r", "RFoot"), ("l", "LFoot")):
        foot = np.column_stack([_interp(pos[:, idx[fkey], a]) for a in range(3)])
        rel = (foot[:, fwd] - pelvis[:, fwd]) * direction
        hs, _ = find_peaks(rel, distance=min_sep)
        to, _ = find_peaks(-rel, distance=min_sep)
        ev[side] = {"hs": hs, "to": to, "foot": foot}

    # Speed (m/s) from pelvis forward travel over the analysed window.
    dur = times[-1] - times[0]
    speed = abs(pelvis[-1, fwd] - pelvis[0, fwd]) / dur if dur > 0 else np.nan

    # Cadence from all heel strikes.
    all_hs = np.sort(np.concatenate([ev["r"]["hs"], ev["l"]["hs"]]))
    cadence = (len(all_hs) - 1) / ((all_hs[-1] - all_hs[0]) * dt) * 60 if len(all_hs) >= 2 else np.nan

    def stride_len(side):
        hs = ev[side]["hs"]
        foot = ev[side]["foot"]
        if len(hs) < 2:
            return np.nan
        return float(np.mean(np.abs(np.diff(foot[hs, fwd]))))

    def stance_pct(side):
        hs, to = ev[side]["hs"], ev[side]["to"]
        if len(hs) < 2 or len(to) < 1:
            return np.nan
        pcts = []
        for a, b in zip(hs[:-1], hs[1:]):
            tos = to[(to > a) & (to < b)]
            if tos.size:
                pcts.append((tos[0] - a) / (b - a) * 100)
        return float(np.mean(pcts)) if pcts else np.nan

    # Step width: lateral distance between feet at heel strikes.
    sw = []
    for side in ("r", "l"):
        for h in ev[side]["hs"]:
            other = "l" if side == "r" else "r"
            sw.append(abs(ev[side]["foot"][h, lat] - ev[other]["foot"][h, lat]))
    step_width = float(np.mean(sw)) if sw else np.nan

    slr, sll = stride_len("r"), stride_len("l")
    out.update({
        "available": True,
        "speed_m_s": float(speed),
        "cadence_steps_min": float(cadence),
        "stride_length_m": {"r": slr, "l": sll},
        "stance_pct": {"r": stance_pct("r"), "l": stance_pct("l")},
        "swing_pct": {"r": 100 - stance_pct("r") if stance_pct("r") == stance_pct("r") else np.nan,
                      "l": 100 - stance_pct("l") if stance_pct("l") == stance_pct("l") else np.nan},
        "step_width_m": step_width,
        "stride_length_symmetry": (min(slr, sll) / max(slr, sll)
                                   if slr == slr and sll == sll and max(slr, sll) > 0 else np.nan),
        "n_heel_strikes": {"r": int(len(ev["r"]["hs"])), "l": int(len(ev["l"]["hs"]))},
    })
    return out
