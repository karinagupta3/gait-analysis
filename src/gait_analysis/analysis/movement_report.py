"""HTML report for rep-based movements (squat, sit-to-stand) from movement_2d.

Same look + honesty as the gait screening report: a summary keyed to the test,
per-rep kinematics, a time-series plot with rep markers, norms, and a disclaimer.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

from .movement_2d import (
    STS_5X_FALLRISK_S, STS_5X_NORMS, STS_5X_SCREEN_S, SQUAT_PARALLEL_KNEE)

TASK_LABEL = {"squat": "Squat", "sit_to_stand": "Sit-to-stand"}
DISCLAIMER = (
    "Screening estimate — not a diagnosis. Single camera, side view, sagittal plane; "
    "reliable for flexion depth, timing, and rep counting from a clean side view. "
    "Frontal-plane knee valgus is NOT measured here (needs a front view). For clinical "
    "decisions use validated testing and clinician judgment."
)


def _plot(metrics) -> str | None:
    s = metrics["_series"]
    if not np.isfinite(s["knee"]).any():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fps = metrics["fps"]
    t = np.arange(len(s["knee"])) / fps
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for key, color, label in (("knee", "#2563eb", "Knee flexion"),
                              ("hip", "#16a34a", "Hip flexion"),
                              ("trunk", "#b45309", "Trunk lean")):
        y = np.asarray(s[key], float)
        if np.isfinite(y).any():
            ax.plot(t, y, color=color, lw=1.4, label=label)
    for i in metrics.get("_bottoms", []):
        ax.axvline(i / fps, color="#94a3b8", lw=0.8, ls=":")
    ax.set_xlabel("seconds (dotted = each rep)"); ax.set_ylabel("degrees")
    ax.set_title(f"{TASK_LABEL.get(metrics['task'], metrics['task'])} — joint angles over time", fontsize=10)
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.25)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _row(label, value, norm="", note=""):
    v = "—" if value is None else value
    return f"<tr><td>{label}</td><td><b>{v}</b></td><td>{norm}</td><td>{note}</td></tr>"


def build_movement_report(metrics: dict, out_html, subject: str = "") -> Path:
    task = metrics["task"]
    label = TASK_LABEL.get(task, task)
    n = metrics["n_reps"]
    used, total = metrics.get("frames_used", 0), metrics.get("frames_total", 0)
    meta_line = f"{subject or 'Subject'} · reps: {n} · usable frames: {used}/{total}"

    if n == 0:
        body = ('<div style="background:#fee2e2;border:1px solid #ef4444;padding:14px 16px;'
                'border-radius:8px;font-size:14px"><b>&#9888; No reps detected.</b> Record the '
                '<b>whole body from the side</b>, fully in frame, performing the movement '
                f'({"5 rises for 5x sit-to-stand" if task=="sit_to_stand" else "3–5 squat reps"}).</div>')
        return _write(out_html, _wrap(subject, label, meta_line, body))

    rows = ""
    if task == "sit_to_stand":
        five = metrics.get("sts_5x_time_s") or metrics.get("sts_5x_time_s_est")
        est = " (est. from fewer reps)" if metrics.get("sts_5x_time_s_est") else ""
        note = ""
        if five is not None:
            note = ("&#9888; recurrent-faller range (>15 s)" if five > STS_5X_FALLRISK_S
                    else "&#9888; screen-positive for fall risk (≥12 s)" if five >= STS_5X_SCREEN_S
                    else "within typical range")
        norm_txt = "60–69y ≈11.4 s, 70–79y ≈12.6 s; ≥12 s screen-positive (Tiedemann 2008)"
        rows += _row("5× sit-to-stand time", f"{five} s{est}" if five else None, norm_txt, note)
        rows += _row("Stands detected", metrics.get("stands", n))
        # 30-second STS: only when a full ~30 s test was recorded.
        c30 = metrics.get("sts_30s_count")
        if c30 is not None:
            rows += _row("30-second STS (count)", c30,
                         "age/sex bands — Rikli & Jones 1999 / CDC STEADI",
                         "below the age band = elevated fall risk")
        # Leg power (Alcazar) — needs height + weight entered.
        pwkg = metrics.get("power_wkg")
        if pwkg is not None:
            prisk = ("&#9888; low (men <2.5 / women <2.0 W/kg)" if pwkg < 2.53 else "within range")
            rows += _row("Leg power", f"{pwkg} W/kg ({metrics.get('power_w'):.0f} W)",
                         "low: men <2.53, women <2.01 W/kg (Garcia-Aguirre 2025)", prisk)
        rows += _row("Time per rise", f"{metrics.get('time_per_rep_s')} s" if metrics.get("time_per_rep_s") else None, "≈2–3 s")
        rows += _row("Peak trunk lean (rise)", f"{metrics.get('trunk_peak_mean'):.0f}°" if metrics.get("trunk_peak_mean") is not None else None,
                     "healthy ≈59° at seat-off (Frontiers 2025)")
    else:  # squat
        km = metrics.get("knee_peak_mean")
        rows += _row("Reps detected", n)
        rows += _row("Peak knee flexion (depth)",
                     f"{km:.0f} ± {metrics.get('knee_peak_sd',0):.0f}°" if km is not None else None,
                     f"parallel ≈ {SQUAT_PARALLEL_KNEE[0]}–{SQUAT_PARALLEL_KNEE[1]}°",
                     metrics.get("depth_class", ""))
        rows += _row("Peak hip flexion", f"{metrics.get('hip_peak_mean'):.0f}°" if metrics.get("hip_peak_mean") is not None else None)
        rows += _row("Peak trunk lean", f"{metrics.get('trunk_peak_mean'):.0f}°" if metrics.get("trunk_peak_mean") is not None else None,
                     "more lean = ankle/hip mobility or compensation")
        rows += _row("Time per rep", f"{metrics.get('time_per_rep_s')} s" if metrics.get("time_per_rep_s") else None)

    sym = metrics.get("symmetry_knee_pct")
    sym_html = ""
    if sym is not None:
        col = "#15803d" if sym >= 90 else "#b45309"
        sym_html = (f"<h2 class='sec'>Left / right symmetry <span class='exp'>experimental</span></h2>"
                    f"<p><b style='color:{col}'>{sym:.0f}% symmetric</b> (peak knee flexion). "
                    f"The far leg is noisier in one camera — treat as a flag, not a measurement.</p>")

    plot = _plot(metrics)
    plot_html = f"<img src='data:image/png;base64,{plot}'>" if plot else ""

    body = (f"<h2 class='sec'>{label} summary</h2>"
            f"<table><thead><tr><th>Measure</th><th>Result</th><th>Typical</th><th>Note</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            f"<h2 class='sec'>Joint angles over time</h2>{plot_html}"
            f"{sym_html}")
    return _write(out_html, _wrap(subject, label, meta_line, body))


def _wrap(subject, label, meta_line, body, disclaimer=None) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{label} screening — {subject or 'Subject'}</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;max-width:860px;margin:24px auto;color:#111;padding:0 16px}}
 h1{{font-size:21px;margin-bottom:2px}} .meta{{color:#6b7280;font-size:13px}}
 h2.sec{{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#475569;margin:24px 0 8px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}}
 .exp{{font-size:10px;background:#fef3c7;color:#92400e;padding:1px 6px;border-radius:6px;letter-spacing:0;text-transform:none}}
 table{{border-collapse:collapse;width:100%;margin:6px 0}} th,td{{border:1px solid #e5e7eb;padding:7px 9px;text-align:left;font-size:14px}}
 th{{background:#f9fafb;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#475569}}
 .disc{{background:#fef3c7;border:1px solid #f59e0b;padding:10px 14px;border-radius:8px;font-size:13px;margin:14px 0}}
 img{{max-width:100%;border:1px solid #eee;border-radius:6px}}
</style></head><body>
<h1>{label} Screening Report</h1>
<div class="meta">{meta_line}</div>
<div class="disc"><b>⚠ {disclaimer or DISCLAIMER}</b></div>
{body}
<p class="meta">Method: single-camera 2D pose (MediaPipe BlazePose). Screening tool only.</p>
</body></html>"""


def _write(out_html, html) -> Path:
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)
    return out_html
