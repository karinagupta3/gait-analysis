"""HTML report for the single-phone 2D sagittal SCREENING mode.

Renders the sagittal2d angles against published normal ranges, with a curve plot
and a prominent screening disclaimer. Self-contained HTML (base64 plot), no JS.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

# Published sagittal-plane norms for level walking (screening reference).
# Perry & Burnfield, Gait Analysis 2nd ed.; spatiotemporal per Bohannon 1997.
NORMS = {
    "knee_flexion":       {"label": "Knee flexion",        "lo": 0,   "hi": 63,  "rom": 60,
                           "note": "0° at stance → ~60-65° peak in swing"},
    "hip_flexion":        {"label": "Hip flexion",         "lo": -10, "hi": 30,  "rom": 40,
                           "note": "~10° extension (stance) → ~30° flexion (swing)"},
    "ankle_dorsiflexion": {"label": "Ankle dorsiflexion",  "lo": -20, "hi": 10,  "rom": 30,
                           "note": "low confidence in 2D — interpret cautiously"},
}
DISCLAIMER = (
    "Screening estimate — not a diagnosis. Single-camera, side-view, sagittal-plane "
    "angles only; accuracy is limited and ankle/out-of-plane values are unreliable. "
    "For clinical decisions use validated motion capture and clinician judgment."
)


def _plot_b64(series: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2))
    for key, color in (("knee_flexion", "#2563eb"), ("hip_flexion", "#16a34a")):
        y = series.get(key)
        if y is None:
            continue
        y = np.asarray(y, dtype=float)
        ax.plot(np.arange(len(y)), y, color=color, lw=1.2, label=NORMS[key]["label"])
    ax.set_xlabel("frame"); ax.set_ylabel("degrees"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Sagittal joint flexion over the trial", fontsize=10)
    ax.grid(alpha=0.25)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _row(key: str, measured: dict | None) -> str:
    n = NORMS[key]
    if not measured:
        return f"<tr><td>{n['label']}</td><td>—</td><td>{n['lo']}° to {n['hi']}°</td><td>no data</td></tr>"
    rng = f"{measured['min']:.0f}° to {measured['max']:.0f}° (ROM {measured['rom']:.0f}°)"
    # Flag if measured ROM is well outside the normal ROM (screening heuristic).
    flag = "within range"
    if measured["rom"] < 0.6 * n["rom"]:
        flag = "⚠ reduced ROM"
    elif measured["rom"] > 1.6 * n["rom"]:
        flag = "⚠ excess/variable"
    if measured.get("low_confidence"):
        flag += " · low confidence"
    return (f"<tr><td>{n['label']}</td><td>{rng}</td>"
            f"<td>{n['lo']}° to {n['hi']}° (ROM ~{n['rom']}°)<br><small>{n['note']}</small></td>"
            f"<td>{flag}</td></tr>")


def build_screening_report(result: dict, out_html: str | Path, subject: str = "") -> Path:
    rows = "".join(_row(k, result.get(k)) for k in ("knee_flexion", "hip_flexion", "ankle_dorsiflexion"))
    plot = _plot_b64(result.get("_series", {}))
    used, total = result.get("frames_used", 0), result.get("frames_total", 0)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Gait Screening — {subject or 'Subject'}</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;max-width:820px;margin:24px auto;color:#111;padding:0 16px}}
 .disc{{background:#fef3c7;border:1px solid #f59e0b;padding:10px 14px;border-radius:8px;font-size:13px;margin:12px 0}}
 table{{border-collapse:collapse;width:100%;margin:14px 0}} th,td{{border:1px solid #e5e7eb;padding:8px;text-align:left;font-size:14px;vertical-align:top}}
 th{{background:#f9fafb}} h1{{font-size:20px;margin-bottom:2px}} .meta{{color:#6b7280;font-size:13px}}
 img{{max-width:100%;border:1px solid #eee;border-radius:6px}}
</style></head><body>
<h1>Gait Screening Report</h1>
<div class="meta">{subject or 'Subject'} · side analyzed: <b>{result.get('side','?')}</b> · frames used: {used}/{total}</div>
<div class="disc"><b>⚠ {DISCLAIMER}</b></div>
<table><thead><tr><th>Joint (sagittal)</th><th>Measured</th><th>Typical (level walking)</th><th>Screening note</th></tr></thead>
<tbody>{rows}</tbody></table>
<img src="data:image/png;base64,{plot}" alt="joint angle curves">
<p class="meta">Method: single-camera 2D pose (MediaPipe) → sagittal joint flexion in the image plane.
Norms: Perry &amp; Burnfield, <i>Gait Analysis</i> (2nd ed.). This is a screening tool only.</p>
</body></html>"""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)
    return out_html
