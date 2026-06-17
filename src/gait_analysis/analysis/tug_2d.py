"""Timed Up & Go (TUG) from a single side/oblique-view phone clip.

TUG protocol: subject starts seated, on "go" stands, walks 3 m, turns, walks
back, sits. The validated output is TOTAL TIME (>=13.5 s = elevated fall risk,
Shumway-Cook 2000). Subphase splits, step count, and gait speed are best-effort
2D estimates and are clearly labeled as such — the subject rotates out of plane
during the turn, so only timing/total are robust.

Computed from the hip-centre trajectory:
  * vertical (y): seated (low/large-y) vs standing (high/small-y) -> stand & sit events
  * horizontal (x): walk out -> turn (apex of displacement) -> walk back
  * total time = first rise -> final sit
"""
from __future__ import annotations

import numpy as np

from ..pose.mediapipe3d import BLAZEPOSE_33
from .gait_metrics_2d import _savgol

_IDX = {name: i for i, name in enumerate(BLAZEPOSE_33)}

TUG_FALLRISK_S = 13.5          # >=13.5 s = elevated fall risk (Shumway-Cook 2000)
TUG_DEPENDENT_S = 30.0         # >=30 s ~ dependent mobility
TUG_COURSE_M = 3.0            # standard walk distance each way


def _hip_center(px, vis):
    """Hip-centre x,y per frame (pixels); NaN where a hip is low-confidence."""
    lh, rh = px[:, _IDX["left_hip"]], px[:, _IDX["right_hip"]]
    v = np.minimum(vis[:, _IDX["left_hip"]], vis[:, _IDX["right_hip"]])
    c = (lh + rh) / 2.0
    ok = (v >= 0.4) & np.isfinite(c).all(axis=1)
    c[~ok] = np.nan
    return c[:, 0], c[:, 1]


def compute_tug_metrics(image_landmarks, visibility, width, height, fps) -> dict:
    fps = float(fps) or 30.0
    px = image_landmarks.astype(float) * np.array([width, height])
    vis = visibility.astype(float)
    hx, hy = _hip_center(px, vis)
    hy = _savgol(hy, fps)
    hx = _savgol(hx, fps)
    n = len(hy)
    finite = np.isfinite(hy)
    out = {"task": "tug", "fps": round(fps, 1), "frames_total": int(n),
           "frames_used": int(finite.sum())}
    if finite.sum() < int(2 * fps):
        out["total_time_s"] = None
        return out

    # Seated baseline from the first ~0.8 s; standing from the upper part of travel.
    head = hy[:max(1, int(0.8 * fps))]
    seated_y = np.nanmedian(head)                     # larger y = lower in frame = seated
    standing_y = np.nanpercentile(hy[finite], 10)     # smaller y = higher = standing
    rng = seated_y - standing_y
    if rng < 5:                                       # no clear sit->stand excursion
        out["total_time_s"] = None
        return out
    rise_level = seated_y - 0.25 * rng                # crossed once clearly standing-bound
    sit_level = seated_y - 0.25 * rng

    idx = np.where(finite)[0]
    # onset = first frame the hip rises above the rise threshold
    above = idx[hy[idx] < rise_level]
    if above.size == 0:
        out["total_time_s"] = None
        return out
    t_start = int(above[0])
    # final sit = last frame the hip is back near seated (after having stood)
    back = idx[(idx > t_start) & (hy[idx] >= sit_level)]
    t_end = int(back[-1]) if back.size else int(idx[-1])
    out["total_time_s"] = round((t_end - t_start) / fps, 1)

    # Turn = apex of horizontal displacement from the start position.
    x0 = np.nanmedian(hx[t_start:t_start + max(1, int(0.5 * fps))])
    disp = np.abs(hx - x0)
    seg = np.arange(t_start, t_end + 1)
    segdisp = disp[seg]
    if np.isfinite(segdisp).any():
        t_turn = int(seg[np.nanargmax(segdisp)])
        out["turn_time_s"] = round((t_turn - t_start) / fps, 1)
    else:
        t_turn = (t_start + t_end) // 2

    # Stand-up complete = first frame the hip reaches the standing plateau.
    stood = idx[(idx >= t_start) & (hy[idx] <= standing_y + 0.15 * rng)]
    t_stand = int(stood[0]) if stood.size else t_start
    out["stand_time_s"] = round((t_stand - t_start) / fps, 1)

    # Best-effort walk speed (assumes the standard 3 m course, out + back = 6 m).
    walk_frames = max(1, t_end - t_stand)
    walk_s = walk_frames / fps
    out["gait_speed_mps_est"] = round((2 * TUG_COURSE_M) / walk_s, 2) if walk_s > 0 else None

    # Step count (approximate): swing-foot vertical oscillations during the walk.
    try:
        from scipy.signal import find_peaks
        anky = _savgol((px[:, _IDX["left_ankle"], 1] + px[:, _IDX["right_ankle"], 1]) / 2, fps)
        wseg = anky[t_stand:t_end + 1]
        wseg = wseg[np.isfinite(wseg)]
        if wseg.size > 5:
            pk, _ = find_peaks(-wseg, distance=max(1, int(0.3 * fps)), prominence=5)
            out["steps_est"] = int(len(pk))
    except Exception:
        pass

    out["_series"] = {"hip_y": hy, "hip_x": hx}
    out["_events"] = {"start": t_start, "stand": t_stand, "turn": t_turn, "end": t_end}
    return out


DISCLAIMER = (
    "Screening estimate — not a diagnosis. Single camera; TOTAL TIME is the validated "
    "metric (>=13.5 s = elevated fall risk, Shumway-Cook 2000). Sub-phase splits, step "
    "count, and gait speed are approximate 2D estimates (the subject turns out of plane). "
    "Gait speed assumes the standard 3 m course."
)


def build_tug_report(metrics: dict, out_html, subject: str = ""):
    from .movement_report import _row, _wrap, _write
    tt = metrics.get("total_time_s")
    used, total = metrics.get("frames_used", 0), metrics.get("frames_total", 0)
    meta_line = f"{subject or 'Subject'} · total time: {tt if tt is not None else '—'} s · usable frames: {used}/{total}"

    if tt is None:
        body = ('<div style="background:#fee2e2;border:1px solid #ef4444;padding:14px 16px;'
                'border-radius:8px;font-size:14px"><b>&#9888; Could not time the test.</b> '
                'Record the FULL test from the side: subject seated in a chair (in frame), stands, '
                'walks ~3 m, turns, walks back, sits. Keep the body in frame the whole time.</div>')
        return _write(out_html, _wrap(subject, "Timed Up & Go", meta_line, body, DISCLAIMER))

    flag = ("&#9888; dependent-mobility range (≥30 s)" if tt >= TUG_DEPENDENT_S
            else "&#9888; elevated fall risk (≥13.5 s)" if tt >= TUG_FALLRISK_S
            else "within typical range (<13.5 s)")
    rows = _row("Total time", f"{tt} s", "≥13.5 s = elevated fall risk (Shumway-Cook 2000)", flag)
    rows += _row("Stand-up time", f"{metrics.get('stand_time_s')} s" if metrics.get("stand_time_s") is not None else None, "approx")
    rows += _row("Time to turn", f"{metrics.get('turn_time_s')} s" if metrics.get("turn_time_s") is not None else None, "approx")
    rows += _row("Gait speed (est.)", f"{metrics.get('gait_speed_mps_est')} m/s" if metrics.get("gait_speed_mps_est") is not None else None,
                 "assumes standard 3 m course", "approx")
    rows += _row("Steps (est.)", metrics.get("steps_est"), "", "approx")

    plot = _tug_plot(metrics)
    plot_html = f"<img src='data:image/png;base64,{plot}'>" if plot else ""
    body = (f"<h2 class='sec'>Timed Up &amp; Go</h2>"
            f"<table><thead><tr><th>Measure</th><th>Result</th><th>Typical</th><th>Note</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            f"<h2 class='sec'>Movement trace</h2>{plot_html}")
    return _write(out_html, _wrap(subject, "Timed Up & Go", meta_line, body, DISCLAIMER))


def _tug_plot(metrics):
    s = metrics.get("_series")
    ev = metrics.get("_events")
    if not s or not ev:
        return None
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fps = metrics["fps"]
    t = np.arange(len(s["hip_y"])) / fps
    fig, ax = plt.subplots(figsize=(8, 3.0))
    # invert hip_y so "up" on the plot = standing up
    ax.plot(t, -s["hip_y"], color="#2563eb", lw=1.4, label="hip height (up = standing)")
    ax.plot(t, s["hip_x"], color="#16a34a", lw=1.0, alpha=0.7, label="hip horizontal (walk/turn)")
    for name, col in (("start", "#15803d"), ("stand", "#0891b2"), ("turn", "#b45309"), ("end", "#b91c1c")):
        ax.axvline(ev[name] / fps, color=col, lw=1.0, ls=":", label=name)
    ax.set_xlabel("seconds"); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=7, ncol=2); ax.grid(alpha=0.2)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
