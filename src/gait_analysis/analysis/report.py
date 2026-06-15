"""Thorough, self-contained HTML gait report.

Combines everything into one shareable .html file: a confidence banner, the clinical
signature flags (with interpretations + the confirming clinical test), the joint-angle
curves (embedded), a ROM table vs normative references, and the honesty/limitations
footer. Self-contained (plots are base64-embedded) so it opens in any browser and is
the basis for the interactive UI (docs/06).
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import html
import io
from pathlib import Path

from . import gait_cycle, interpretation, kinematics, signatures, tasks

# Normative sagittal references (Perry & Burnfield / AAPM&R; see docs/04 Section A).
# Keyed by coordinate base (strip _r/_l/_beta). Display-only reference strings.
NORM_REF = {
    "hip_flexion": "IC ~+30 flex, terminal stance ~ -10 (ext)",
    "knee_angle": "LR flex ~15, swing peak ~60-65, ~0 terminal stance",
    "ankle_angle": "~+10 DF (stance), ~-15..-20 PF (toe-off)",
    "hip_adduction": "~+5..10 add, ~-5 abd (swing)",
    "hip_rotation": "~+/-5 (markerless: low confidence)",
    "pelvis_tilt": "~7-13 ant tilt, ~4 cyclic excursion",
    "pelvis_list": "~+/-4-5",
    "pelvis_rotation": "~+/-5-8 (markerless: low confidence)",
    "subtalar_angle": "small (markerless: low confidence)",
    "arm_flex": "~20-25 excursion (noisy markerless)",
}

_CONF_COLOR = {"high": "#c0392b", "moderate": "#d68910", "low": "#7f8c8d"}

# Clinically key coordinates to chart by default (one clean panel each, like OpenCap).
KEY_COORDS = ["pelvis_tilt", "pelvis_list", "pelvis_rotation",
              "hip_flexion", "hip_adduction", "knee_angle", "ankle_angle", "arm_flex"]


def _base(name: str) -> str:
    for suf in ("_r", "_l", "_beta"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name


# Which coordinates matter clinically for each action (what to chart).
TASK_COORDS = {
    "squat": ["pelvis_tilt", "hip_flexion", "knee_angle", "ankle_angle", "hip_adduction"],
    "gait":  ["pelvis_tilt", "pelvis_list", "hip_flexion", "hip_adduction", "knee_angle", "ankle_angle"],
    "sts":   ["pelvis_tilt", "hip_flexion", "knee_angle", "ankle_angle"],
}

TASK_TITLE = {"squat": "Squat", "gait": "Walking / gait", "sts": "Sit-to-stand"}


def _fmt(v, unit="deg"):
    return f"{v:.1f} {unit}" if isinstance(v, (int, float)) and v == v else "n/a"


def _table(rows: list[tuple[str, str, str]]) -> str:
    body = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(str(b))}</td>"
                   f"<td class='ref'>{html.escape(c)}</td></tr>" for a, b, c in rows)
    return ("<table><thead><tr><th>metric</th><th>value</th><th>clinical reference</th>"
            f"</tr></thead><tbody>{body}</tbody></table>")


def _squat_panel(m: dict) -> str:
    if not m or m.get("n_reps", 0) == 0:
        return "<p class='meta'>No squat reps detected.</p>"
    kr, kl = m.get("peak_knee_flexion_r"), m.get("peak_knee_flexion_l")
    rows = [
        ("Reps analyzed", m.get("n_reps", "n/a"), ""),
        ("Squat depth — peak knee flexion (R / L)", f"{_fmt(kr)} / {_fmt(kl)}",
         ">=90 parallel, >=100 deep"),
        ("Peak hip flexion (R / L)",
         f"{_fmt(m.get('peak_hip_flexion_r'))} / {_fmt(m.get('peak_hip_flexion_l'))}",
         "~95 needed for a deep squat"),
        ("Dynamic valgus — peak hip adduction (R / L)",
         f"{_fmt(m.get('peak_hip_adduction_r'))} / {_fmt(m.get('peak_hip_adduction_l'))}",
         ">~10 = medial collapse (ACL/PFP link)"),
        ("Ankle dorsiflexion (R / L)",
         f"{_fmt(m.get('peak_ankle_df_r'))} / {_fmt(m.get('peak_ankle_df_l'))}",
         "limited DF caps depth / drives butt wink"),
        ("Butt wink — posterior pelvic-tilt reversal",
         _fmt(m.get("butt_wink_posterior_excursion")), ">~8 = butt wink (load-dependent risk)"),
    ]
    if kr and kl:
        rows.append(("L / R depth asymmetry",
                     f"{abs(kr - kl) / (0.5 * (kr + kl)) * 100:.0f} %", "<15% (borrowed cut-point)"))
    return _table(rows)


def _sts_panel(m: dict) -> str:
    if not m or m.get("n_rises", 0) == 0:
        return "<p class='meta'>No sit-to-stand rises detected.</p>"
    rows = [
        ("Rises analyzed", m.get("n_rises", "n/a"), ""),
        ("Mean rise time", _fmt(m.get("mean_rise_time_s"), "s"), ""),
    ]
    if m.get("n_rises", 0) >= 5:
        rows.append(("5x sit-to-stand time", _fmt(m.get("total_time_s"), "s"),
                     "60s ~11.4, 70s ~12.6, 80s ~14.8; >12 assess, >15 fall risk"))
    return _table(rows)


def _gait_panel(phase, summary) -> str:
    n = phase.n_cycles
    dur = summary["duration_s"]
    cad = (n * 2) / dur * 60 if (dur > 0 and n) else None   # 2 steps per cycle
    rows = [
        ("Gait cycles analyzed", n, ""),
        ("Cadence (estimated)", _fmt(cad, "steps/min") if cad else "n/a", "~100-120"),
        ("Stride/step length, double-support, step width", "needs the marker .trc",
         "spatiotemporal from foot markers (Track A) -- coming"),
    ]
    return _table(rows)


def _task_panel(task: str, metrics: dict, phase, summary) -> str:
    if task == "squat":
        return _squat_panel(metrics)
    if task == "sts":
        return _sts_panel(metrics)
    return _gait_panel(phase, summary)


def _plot_b64(time, coords, names=None) -> str:
    """One clean panel per clinically-relevant coordinate (real .mot data vs time)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    want = names or KEY_COORDS
    bases = [b for b in want
             if b in coords or f"{b}_r" in coords or f"{b}_l" in coords]
    if not bases:                                   # fall back to whatever exists
        bases = sorted({_base(c) for c in coords})[:8]
    n = len(bases) or 1
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.0 * n), squeeze=False, sharex=True)
    for i, b in enumerate(bases):
        ax = axes[i][0]
        bilateral = False
        if f"{b}_r" in coords:
            ax.plot(time, coords[f"{b}_r"], color="#1f77b4", lw=1.6, label="right")
            bilateral = True
        if f"{b}_l" in coords:
            ax.plot(time, coords[f"{b}_l"], color="#d62728", lw=1.6, label="left")
            bilateral = True
        if b in coords:
            ax.plot(time, coords[b], color="#2a9d8f", lw=1.6, label=b)
        ax.set_title(b, fontsize=10, loc="left", fontweight="bold")
        ax.set_ylabel("degrees", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        if bilateral:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
    axes[-1][0].set_xlabel("time (s)", fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _flag_cards(findings) -> str:
    if not findings:
        return "<p class='ok'>No threshold-grade signature flags triggered.</p>"
    cards = []
    for f in findings:
        color = _CONF_COLOR.get(f.confidence, "#7f8c8d")
        interps = "".join(f"<li>{html.escape(i)}</li>" for i in f.interpretations)
        caveats = "".join(f"<div class='caveat'>{html.escape(c)}</div>" for c in f.caveats)
        g = interpretation.guidance_for(f.rule_id)
        gblock = ""
        if g:
            gblock = (f"<div class='guide'>"
                      f"<div><b>What it means:</b> {html.escape(g.meaning)}</div>"
                      f"<div><b>What helps:</b> {html.escape(g.treatment)}</div>"
                      f"<div><b>Tracking change:</b> {html.escape(g.tracking)}</div></div>")
        cards.append(f"""
        <div class="card" style="border-left:6px solid {color}">
          <div class="card-h"><span class="conf" style="background:{color}">{f.confidence.upper()}</span>
            {html.escape(f.title)}</div>
          <div class="val"><code>{html.escape(f.coordinate)} = {f.observed}</code>
            &nbsp;<span class="thr">flag: {html.escape(f.threshold)}</span></div>
          <div class="lbl">Consistent with:</div><ul>{interps}</ul>{caveats}{gblock}
        </div>""")
    return "\n".join(cards)


def _rom_table(summary) -> str:
    rows = []
    for name in sorted(summary["rom"]):
        r = summary["rom"][name]
        ref = NORM_REF.get(_base(name), "")
        rows.append(f"<tr><td><code>{html.escape(name)}</code></td>"
                    f"<td>{r['min']:.1f}</td><td>{r['max']:.1f}</td><td>{r['range']:.1f}</td>"
                    f"<td class='ref'>{html.escape(ref)}</td></tr>")
    return ("<table><thead><tr><th>coordinate</th><th>min</th><th>max</th><th>range</th>"
            "<th>normal reference (sagittal)</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def build_html_report(mot_path, out_html, gait_speed_m_s=None,
                      subject: str | None = None, title: str = "Gait Analysis Report") -> Path:
    time, coords, meta = kinematics.read_storage(mot_path)
    summary = kinematics.summarize(time, coords, meta)
    phase = gait_cycle.compute_phase_features(time, coords)
    ctx = signatures.Context(gait_speed_m_s=gait_speed_m_s, phase=phase)
    task, findings, metrics = tasks.route(time, coords, summary, ctx)
    task_name = TASK_TITLE.get(task, task)

    n = phase.n_cycles
    conf = ("GOOD" if n >= 3 else "LOW -- short trial, interpret cautiously" if n >= 1
            else "NO CYCLES -- global fallback, unreliable")
    speed = f"{gait_speed_m_s:.2f} m/s" if gait_speed_m_s is not None else "n/a"
    caveats = "".join(f"<li>{html.escape(c)}</li>" for c in signatures.GLOBAL_CAVEATS)

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}}
h1{{margin:0 0 4px}} h2{{border-bottom:2px solid #eee;padding-bottom:4px;margin-top:32px}}
.meta{{color:#666;font-size:13px}} .banner{{padding:10px 14px;border-radius:8px;background:#f4f6f8;margin:14px 0}}
.card{{background:#fafbfc;border:1px solid #eee;border-radius:8px;padding:10px 14px;margin:10px 0}}
.card-h{{font-weight:600}} .conf{{color:#fff;font-size:11px;padding:2px 6px;border-radius:4px;margin-right:8px}}
.val{{margin:6px 0}} .thr{{color:#888}} .lbl{{font-size:12px;color:#666;margin-top:6px}}
ul{{margin:4px 0 4px 18px}} .caveat{{font-size:12px;color:#a55;margin-top:4px}}
.ok{{color:#27ae60}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #eee;padding:4px 8px;text-align:left}} th{{background:#f4f6f8}}
.ref{{color:#777;font-size:12px}} img{{max-width:100%;border:1px solid #eee;border-radius:8px}}
.disclaim{{font-size:12px;color:#777;background:#fffbe6;border:1px solid #f0e2a0;border-radius:8px;padding:10px 14px}}
.guide{{margin-top:8px;padding:8px 10px;background:#eefaf3;border-radius:6px;font-size:13px}}
.guide div{{margin:3px 0}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">{('Subject: ' + html.escape(subject) + ' &middot; ') if subject else ''}
Generated {(_dt.date.today().isoformat())} &middot; angles in {'deg' if summary['in_degrees'] else 'rad'}</div>
<div class="banner"><b>Task:</b> {task} &nbsp;|&nbsp; <b>Data confidence:</b> {conf}
{('&nbsp;|&nbsp; gait cycles: ' + str(n)) if task == 'gait' else ''}
&nbsp;|&nbsp; frames: {summary['n_frames']} ({summary['duration_s']:.1f}s)
&nbsp;|&nbsp; gait speed: {speed} &nbsp;|&nbsp; coordinates: {summary['n_coordinates']}</div>

<h2>Key metrics &mdash; {html.escape(task_name)}</h2>
<p class="meta">The measures that matter clinically for this action.</p>
{_task_panel(task, metrics, phase, summary)}

<h2>Clinical signature flags</h2>
<p class="meta">Research decision-support, <b>not a diagnosis</b>. Each flag lists multiple
plausible causes and the confirming clinical test.</p>
{_flag_cards(findings)}

<h2>Joint-angle curves ({html.escape(task_name)}-relevant, R vs L)</h2>
<img src="data:image/png;base64,{_plot_b64(time, coords, TASK_COORDS.get(task))}" alt="joint angle curves"/>

<h2>Range of motion vs normative reference</h2>
{_rom_table(summary)}

<h2>Limitations</h2>
<div class="disclaim"><ul>{caveats}</ul></div>
</body></html>"""

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc)
    return out_html


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a self-contained HTML gait report from a .mot")
    ap.add_argument("--mot", required=True)
    ap.add_argument("--out", required=True, help="Output .html")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--subject", default=None)
    args = ap.parse_args(argv)
    out = build_html_report(args.mot, args.out, args.speed, args.subject)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
