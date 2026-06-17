"""HTML report for the single-phone 2D sagittal SCREENING mode.

Consumes the rich metrics from gait_metrics_2d (both legs: per-stride knee/hip
peaks, cadence, stride/step time, stance/swing, symmetry, trunk lean, view
quality) and renders a clinician-readable report with gait-cycle plots, against
published sagittal norms, with a prominent screening disclaimer.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

# Published sagittal-plane norms for comfortable level walking (screening reference).
# Perry & Burnfield, Gait Analysis 2nd ed.; cadence per Bohannon 1997.
NORMS = {
    "cadence_spm": (100, 120, "steps/min"),
    "stride_time_s": (1.0, 1.2, "s"),
    "stance_pct": (58, 62, "% of stride"),
    "swing_pct": (38, 42, "% of stride"),
    "knee_peak": (55, 65, "deg (swing)"),
    "hip_flex": (25, 35, "deg"),
    "hip_ext": (-15, -5, "deg"),
}
DISCLAIMER = (
    "Screening estimate — not a diagnosis. Single camera, side view, sagittal plane "
    "only; the camera-facing leg is most reliable, the far leg and any out-of-plane "
    "motion are low-confidence, and gait events are estimated from joint angles (no "
    "force plate). For clinical decisions use validated motion capture and clinician "
    "judgment."
)


def _normalized_cycles(series, hs, n=101):
    series = np.asarray(series, float)
    cycles = []
    for a, b in zip(hs[:-1], hs[1:]):
        if b - a < 4:
            continue
        seg = series[a:b + 1]
        idx = np.arange(len(seg))
        good = np.isfinite(seg)
        if good.sum() < 3:
            continue
        seg = np.interp(idx, idx[good], seg[good])
        cycles.append(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(seg)), seg))
    return np.array(cycles) if cycles else None


def _plot_cycle(metrics, primary) -> str | None:
    sd = metrics["sides"][primary]
    hs = np.asarray(sd["_events"]["hs"], int)
    knee = _normalized_cycles(sd["_series"]["knee"], hs)
    hip = _normalized_cycles(sd["_series"]["hip"], hs)
    if knee is None and hip is None:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(0, 100, 101)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for cyc, color, label in ((knee, "#2563eb", "Knee flexion"), (hip, "#16a34a", "Hip flexion")):
        if cyc is None:
            continue
        m, s = np.nanmean(cyc, axis=0), np.nanstd(cyc, axis=0)
        ax.plot(x, m, color=color, lw=2, label=f"{label} (n={len(cyc)} strides)")
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)
    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.set_xlabel("% gait cycle (heel strike → next heel strike)")
    ax.set_ylabel("degrees")
    ax.set_title(f"Sagittal joint angles over the gait cycle — {primary} leg (mean ± SD)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _flag(value, lo, hi):
    if value is None:
        return "—", ""
    if value < lo:
        return f"{value:g}", "low"
    if value > hi:
        return f"{value:g}", "high"
    return f"{value:g}", "ok"


def _metric_row(label, value, norm_key, fmt="{:.0f}"):
    lo, hi, unit = NORMS[norm_key]
    shown = "—" if value is None else fmt.format(value)
    cls = ""
    if value is not None:
        if value < lo:
            cls = "low"
        elif value > hi:
            cls = "high"
        else:
            cls = "ok"
    badge = {"ok": "within range", "low": "below typical", "high": "above typical", "": "no data"}[cls]
    color = {"ok": "#15803d", "low": "#b45309", "high": "#b45309", "": "#6b7280"}[cls]
    return (f"<tr><td>{label}</td><td><b>{shown}</b></td>"
            f"<td>{lo}–{hi} {unit}</td>"
            f"<td style='color:{color}'>{badge}</td></tr>")


def build_screening_report(metrics: dict, out_html, subject: str = "") -> Path:
    primary = metrics["primary_side"]
    other = "left" if primary == "right" else "right"
    used, total = metrics.get("frames_used", 0), metrics.get("frames_total", 0)
    vq = metrics.get("view_quality", {})

    # --- unusable / marginal guard ---
    pside = metrics["sides"][primary]
    if used == 0 or pside["peaks"] is None:
        body = (
            '<div style="background:#fee2e2;border:1px solid #ef4444;padding:14px 16px;'
            'border-radius:8px;margin:12px 0;font-size:14px"><b>&#9888; Not enough clean gait '
            'detected.</b> The pose model never saw enough full, in-frame strides from the side. '
            'Record the <b>whole body</b> (head to feet) from the <b>side</b>, ~3–4 m away, with '
            'the subject fully in frame walking 4–6 strides across the view.</div>')
        html = _wrap(subject, primary, used, total, vq, body)
        return _write(out_html, html)

    pk = pside["peaks"]
    tmp = pside["temporal"]
    opk = metrics["sides"][other]["peaks"]
    otmp = metrics["sides"][other]["temporal"]

    # Temporal table (gait timing).
    temporal_rows = (
        _metric_row("Cadence", metrics.get("cadence_spm"), "cadence_spm")
        + _metric_row(f"Stride time ({primary})", tmp.get("stride_time_s"), "stride_time_s", "{:.2f}")
        + _metric_row(f"Stance ({primary})", tmp.get("stance_pct"), "stance_pct")
        + _metric_row(f"Swing ({primary})", tmp.get("swing_pct"), "swing_pct"))

    # Kinematics table — both legs.
    def kin_rows(side, peaks):
        if not peaks:
            return f"<tr><td colspan='4'><i>{side} leg: too few clean strides</i></td></tr>"
        kp = f"{peaks['knee_peak_mean']:.0f} ± {peaks['knee_peak_sd']:.0f}"
        kn_lo, kn_hi, _ = NORMS["knee_peak"]
        kn_badge = "within range" if kn_lo <= peaks['knee_peak_mean'] <= kn_hi else (
            "below typical" if peaks['knee_peak_mean'] < kn_lo else "above typical")
        kn_col = "#15803d" if kn_lo <= peaks['knee_peak_mean'] <= kn_hi else "#b45309"
        rows = (f"<tr><td>Peak knee flexion ({side}, swing)</td><td><b>{kp}°</b></td>"
                f"<td>{kn_lo}–{kn_hi}°</td><td style='color:{kn_col}'>{kn_badge}</td></tr>")
        rows += _metric_row(f"Peak hip flexion ({side})", peaks.get("hip_flex_mean"), "hip_flex", "{:.0f}")
        rows += _metric_row(f"Peak hip extension ({side})", peaks.get("hip_ext_mean"), "hip_ext", "{:.0f}")
        return rows
    kinematic_rows = kin_rows(primary, pk) + kin_rows(other, opk)

    # Symmetry.
    sym = metrics.get("symmetry")
    sym_html = ""
    if sym and (sym.get("knee_peak_pct") is not None):
        def symbadge(v):
            if v is None:
                return "—", "#6b7280"
            return f"{v:.0f}% symmetric", ("#15803d" if v >= 90 else "#b45309")
        ks, kc = symbadge(sym.get("knee_peak_pct"))
        ss, sc = symbadge(sym.get("stride_time_pct"))
        sym_html = (
            "<h2 class='sec'>Left / right symmetry <span class='exp'>experimental</span></h2>"
            "<table><tbody>"
            f"<tr><td>Peak knee flexion</td><td style='color:{kc}'><b>{ks}</b></td></tr>"
            f"<tr><td>Stride time</td><td style='color:{sc}'><b>{ss}</b></td></tr>"
            "</tbody></table>"
            "<p class='note'>100% = identical legs. The far leg is noisier in a single-camera "
            "view, so treat asymmetry as a flag to look closer, not a measurement.</p>")

    trunk = metrics.get("trunk_lean_deg")
    trunk_html = (f"<p class='note'>Trunk lean: <b>{trunk:+.1f}°</b> from vertical "
                  f"(− = leaning back, + = forward).</p>" if trunk is not None else "")

    plot = _plot_cycle(metrics, primary)
    plot_html = (f"<img src='data:image/png;base64,{plot}' alt='gait-cycle joint angles'>"
                 if plot else "<p class='note'>Not enough strides for a gait-cycle plot.</p>")

    strides_note = (f"{pk['n_strides']} strides analyzed on the {primary} leg"
                    + (f", {opk['n_strides']} on the {other} leg" if opk else "") + ".")

    body = f"""
<h2 class="sec">Gait timing</h2>
<table><thead><tr><th>Measure</th><th>Result</th><th>Typical</th><th>Screening note</th></tr></thead>
<tbody>{temporal_rows}</tbody></table>

<h2 class="sec">Joint kinematics (sagittal)</h2>
<table><thead><tr><th>Measure</th><th>Result</th><th>Typical</th><th>Screening note</th></tr></thead>
<tbody>{kinematic_rows}</tbody></table>
<p class="note">{strides_note} Peaks are mean ± SD across strides.</p>

<h2 class="sec">Gait-cycle curves</h2>
{plot_html}

{sym_html}
{trunk_html}
"""
    return _write(out_html, _wrap(subject, primary, used, total, vq, body))


def _wrap(subject, primary, used, total, vq, body) -> str:
    label = vq.get("label", "")
    vq_color = "#15803d" if label.startswith("good") else "#b45309"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Gait Screening — {subject or 'Subject'}</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;max-width:860px;margin:24px auto;color:#111;padding:0 16px}}
 h1{{font-size:21px;margin-bottom:2px}} .meta{{color:#6b7280;font-size:13px}}
 h2.sec{{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#475569;margin:26px 0 8px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}}
 .exp{{font-size:10px;background:#fef3c7;color:#92400e;padding:1px 6px;border-radius:6px;letter-spacing:0;text-transform:none}}
 table{{border-collapse:collapse;width:100%;margin:6px 0 4px}} th,td{{border:1px solid #e5e7eb;padding:7px 9px;text-align:left;font-size:14px}}
 th{{background:#f9fafb;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#475569}}
 .disc{{background:#fef3c7;border:1px solid #f59e0b;padding:10px 14px;border-radius:8px;font-size:13px;margin:14px 0}}
 .vq{{display:inline-block;font-size:12px;font-weight:600;padding:2px 10px;border-radius:999px;color:#fff;background:{vq_color}}}
 img{{max-width:100%;border:1px solid #eee;border-radius:6px;margin-top:6px}} .note{{color:#6b7280;font-size:13px;margin:6px 0}}
</style></head><body>
<h1>Gait Screening Report</h1>
<div class="meta">{subject or 'Subject'} · analyzed leg: <b>{primary}</b> · usable frames: {used}/{total}
&nbsp; <span class="vq">{label}</span></div>
<div class="disc"><b>⚠ {DISCLAIMER}</b></div>
{body}
<p class="meta">Method: single-camera 2D pose (MediaPipe BlazePose) → sagittal joint angles + kinematic
gait events. Norms: Perry &amp; Burnfield, <i>Gait Analysis</i> (2nd ed.); Bohannon 1997 (cadence).
Screening tool only.</p>
</body></html>"""


def _write(out_html, html) -> Path:
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)
    return out_html
