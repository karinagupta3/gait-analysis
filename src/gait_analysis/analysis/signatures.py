"""Clinical-signature rule engine over OpenSim kinematics.

Turns the per-coordinate ROM + L/R symmetry summary (analysis/kinematics.summarize)
into FLAGS that say "this movement pattern is consistent with X" -- e.g., reduced
peak swing knee flexion (stiff-knee), reduced terminal-stance hip extension (tight
hip flexors), reduced swing dorsiflexion (foot drop / equinus), inter-limb asymmetry.

DESIGN PRINCIPLES (from docs/04-clinical-signatures.md, which cites the evidence):
  * Hard numeric thresholds are applied ONLY to sagittal-plane coordinates, which
    have the best normative data and best markerless reliability.
  * Every flag carries multiple plausible interpretations -- a kinematic pattern is
    NOT disease-specific (weakness, tightness, structural limits, and pain produce
    overlapping patterns). We localize the pattern; we do not diagnose.
  * Walking SPEED confounds nearly all "truncation" signatures (slow gait reduces
    flexion peaks and flattens curves), so speed is reported and flagged.
  * Out-of-plane (frontal/transverse) coordinates are ADVISORY only.

These flags are decision-support for research, not medical diagnoses.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from .gait_cycle import PhaseFeatures

# Confidence reflects strength of the underlying evidence + markerless reliability.
CONF_HIGH, CONF_MOD, CONF_LOW = "high", "moderate", "low"

SIDES = {"r": "right", "l": "left"}

GLOBAL_CAVEATS = [
    "This is decision-support, not a diagnosis. The same movement pattern can come from "
    "muscle weakness, tightness, a joint/structural limit, or pain -- so a flag shows WHERE "
    "to look, not the cause.",
    "Walking slowly naturally reduces motion, so a 'reduced' flag may just reflect a slow "
    "walk. Always read these alongside the walking speed.",
    "Side-to-side (frontal) and rotation angles are less reliable from markerless video -- "
    "treat those as a hint, not a measurement. Bend/straighten (sagittal) angles are the "
    "trustworthy ones.",
    "Confirm any flag with a hands-on test (e.g. muscle length or strength testing).",
]


@dataclass
class Finding:
    rule_id: str
    category: str          # tightness | weakness | neuro | asymmetry | pain
    title: str
    coordinate: str
    observed: float
    threshold: str
    interpretations: list[str]
    confidence: str
    caveats: list[str] = field(default_factory=list)


@dataclass
class Context:
    gait_speed_m_s: float | None = None
    cadence_steps_per_min: float | None = None
    phase: PhaseFeatures | None = None     # phase-windowed features (preferred over global min/max)


def _rom(summary: dict, name: str):
    return summary.get("rom", {}).get(name)


# --- individual rules -------------------------------------------------------
# Each rule takes (summary, ctx) and returns a list[Finding].

def rule_stiff_knee_swing(summary, ctx):
    """Reduced PEAK knee flexion in swing. Normal ~60-65 deg; stiff-knee <~45 deg.

    Evidence: between-limb/absolute peak swing knee flexion <44.3 deg identifies
    stiff-knee gait (Goldberg/Campanini lineage). Proxy: global max of knee_angle.
    """
    out = []
    for s in SIDES:
        r = _rom(summary, f"knee_angle_{s}")
        if r and r["max"] < 45:
            out.append(Finding(
                "stiff_knee_swing", "neuro",
                f"Reduced peak knee flexion in swing ({SIDES[s]})",
                f"knee_angle_{s}", round(r["max"], 1), "<45 deg (normal ~60-65)",
                ["stiff-knee gait (rectus femoris over-activity / spasticity)",
                 "quadriceps over-activity or reduced pre-swing knee flexion velocity",
                 "post-stroke spastic hemiplegia"],
                CONF_HIGH,
                ["Speed lowers normal swing flexion too -- prefer between-limb difference."]))
    return out


def rule_crouch_stance_knee(summary, ctx):
    """Excess knee flexion sustained through stance. Crouch if min stance knee >30 deg."""
    out = []
    for s in SIDES:
        r = _rom(summary, f"knee_angle_{s}")
        if r and r["min"] > 30:
            out.append(Finding(
                "crouch_knee", "tightness",
                f"Knee never extends in stance ({SIDES[s]}) -- crouch pattern",
                f"knee_angle_{s}", round(r["min"], 1), ">30 deg min (normal <~20)",
                ["hamstring tightness/short MT length", "hip-flexor contracture",
                 "plantarflexor weakness", "quadriceps weakness"],
                CONF_MOD,
                ["Crouch is multifactorial; confirm hamstring length (popliteal angle) "
                 "and distinguish short vs slow (Arnold)."]))
    return out


def rule_reduced_hip_extension(summary, ctx):
    """Reduced terminal-stance hip extension. Normal peak extension ~ -10 deg
    (hip_flexion min negative). If min stays >0, the hip never extends -> tight hip flexors."""
    out = []
    for s in SIDES:
        r = _rom(summary, f"hip_flexion_{s}")
        if r and r["min"] > 0:
            out.append(Finding(
                "reduced_hip_extension", "tightness",
                f"Hip does not reach extension in terminal stance ({SIDES[s]})",
                f"hip_flexion_{s}", round(r["min"], 1), ">0 deg min (normal ~ -10)",
                ["tight hip flexors (iliopsoas / rectus femoris)",
                 "anterior pelvic tilt compensation", "hip flexion contracture"],
                CONF_MOD,
                ["Corroborate with anterior pelvic tilt and a Thomas test."]))
    return out


def rule_foot_drop(summary, ctx):
    """Reduced swing dorsiflexion. Normal swing brings ankle to ~neutral (0 deg) or
    slight dorsiflexion. If peak dorsiflexion stays plantarflexed -> foot drop / equinus."""
    out = []
    for s in SIDES:
        r = _rom(summary, f"ankle_angle_{s}")
        if r and r["max"] < 5:
            out.append(Finding(
                "foot_drop", "weakness",
                f"Reduced ankle dorsiflexion ({SIDES[s]}) -- foot-drop / equinus pattern",
                f"ankle_angle_{s}", round(r["max"], 1), "<5 deg peak DF (normal ~10)",
                ["dorsiflexor (tibialis anterior) weakness / foot drop",
                 "gastroc-soleus tightness (equinus)",
                 "peroneal nerve palsy or UMN lesion"],
                CONF_MOD,
                ["Distal/foot markerless reliability is lower -- treat small deficits "
                 "cautiously; check for steppage compensation (hip/knee swing flexion)."]))
    return out


def rule_reduced_knee_excursion(summary, ctx):
    """Reduced sagittal knee excursion ('knee stiffening', e.g. knee OA). Normal
    peak-to-peak ~60 deg; flag if range <50 deg."""
    out = []
    for s in SIDES:
        r = _rom(summary, f"knee_angle_{s}")
        if r and r["range"] < 50:
            out.append(Finding(
                "reduced_knee_excursion", "pain",
                f"Reduced knee sagittal excursion ({SIDES[s]}) -- 'stiffening'",
                f"knee_angle_{s}", round(r["range"], 1), "<50 deg range (normal ~60)",
                ["knee OA / joint guarding (antalgic)", "pain-related movement avoidance",
                 "reduced walking speed"],
                CONF_MOD,
                ["Strongly speed-confounded -- always pair with gait speed."]))
    return out


def rule_asymmetry(summary, ctx):
    """Inter-limb ROM asymmetry on bilateral coordinates. Flag outside 0.90-1.10."""
    out = []
    for base, ratio in summary.get("symmetry_LR", {}).items():
        if ratio < 0.90 or ratio > 1.10:
            cat = "neuro" if base in ("arm_flex",) else "asymmetry"
            interp = ["unilateral pathology (injury, pain, or weakness on one side)",
                      "compensation pattern"]
            if base == "arm_flex":
                interp = ["reduced/asymmetric arm swing (early Parkinson's marker; "
                          "asymmetry > amplitude)", "hemiparesis (stroke)"]
            out.append(Finding(
                "rom_asymmetry", cat,
                f"Left/right asymmetry in {base} ROM",
                base, round(ratio, 3), "outside 0.90-1.10 (1.0 = symmetric)",
                interp, CONF_MOD,
                ["If both sides are affected equally, this left/right comparison can look "
                 "normal -- so also compare each side to typical values, not just to each other."]))
    return out


RULES = [
    rule_stiff_knee_swing,
    rule_crouch_stance_knee,
    rule_reduced_hip_extension,
    rule_foot_drop,
    rule_reduced_knee_excursion,
    rule_asymmetry,
]


def phase_findings(phase: PhaseFeatures) -> list[Finding]:
    """Phase-windowed versions of the sagittal rules (preferred when cycles are found)."""
    out: list[Finding] = []
    for s in SIDES:
        ks = phase.peak_swing_knee_flexion.get(s)
        if ks is not None and ks < 45:
            out.append(Finding(
                "stiff_knee_swing", "neuro",
                f"Reduced peak knee flexion in swing ({SIDES[s]})",
                f"knee_angle_{s} (swing peak)", round(ks, 1), "<45 deg (normal ~60-65)",
                ["stiff-knee gait (rectus femoris over-activity / spasticity)",
                 "quadriceps over-activity or reduced pre-swing knee flexion velocity",
                 "post-stroke spastic hemiplegia"],
                CONF_HIGH, ["Prefer the between-limb difference; speed lowers swing flexion too."]))
        hx = phase.terminal_stance_hip_ext.get(s)
        if hx is not None and hx > 0:
            out.append(Finding(
                "reduced_hip_extension", "tightness",
                f"Hip does not reach extension at terminal stance ({SIDES[s]})",
                f"hip_flexion_{s} (terminal stance)", round(hx, 1), ">0 deg (normal ~ -10)",
                ["tight hip flexors (iliopsoas / rectus femoris)",
                 "anterior pelvic tilt compensation", "hip flexion contracture"],
                CONF_MOD, ["Corroborate with anterior pelvic tilt and a Thomas test."]))
        df = phase.swing_dorsiflexion.get(s)
        if df is not None and df < 5:
            out.append(Finding(
                "foot_drop", "weakness",
                f"Reduced swing dorsiflexion ({SIDES[s]}) -- foot-drop / equinus pattern",
                f"ankle_angle_{s} (swing)", round(df, 1), "<5 deg peak DF (normal ~10)",
                ["dorsiflexor (tibialis anterior) weakness / foot drop",
                 "gastroc-soleus tightness (equinus)", "peroneal nerve palsy or UMN lesion"],
                CONF_MOD, ["Distal/foot markerless reliability is lower -- treat small deficits cautiously."]))
        ck = phase.stance_min_knee.get(s)
        if ck is not None and ck > 30:
            out.append(Finding(
                "crouch_knee", "tightness",
                f"Knee never extends in stance ({SIDES[s]}) -- crouch pattern",
                f"knee_angle_{s} (stance min)", round(ck, 1), ">30 deg (normal <~20)",
                ["hamstring tightness/short MT length", "hip-flexor contracture",
                 "plantarflexor weakness", "quadriceps weakness"],
                CONF_MOD, ["Multifactorial; confirm hamstring length (popliteal angle)."]))
    return out

_CONF_ORDER = {CONF_HIGH: 0, CONF_MOD: 1, CONF_LOW: 2}


def detect(summary: dict, ctx: Context | None = None) -> list[Finding]:
    ctx = ctx or Context()
    findings: list[Finding] = []
    phase = ctx.phase

    if phase is not None and phase.n_cycles >= 1:
        # Phase-windowed rules (clinically correct windows) replace the global ones.
        findings.extend(phase_findings(phase))
        # Asymmetry is noise on <2 cycles -- only emit it when the trial is long enough.
        if phase.n_cycles >= 2:
            findings.extend(rule_asymmetry(summary, ctx))
    else:
        # No gait cycles detected -> fall back to global min/max rules.
        for rule in RULES:
            findings.extend(rule(summary, ctx))

    findings.sort(key=lambda f: _CONF_ORDER.get(f.confidence, 9))
    return findings


def format_findings(findings: list[Finding], ctx: Context | None = None) -> str:
    ctx = ctx or Context()
    lines = ["=== Clinical signature flags (research decision-support, NOT diagnosis) ==="]
    if ctx.gait_speed_m_s is not None:
        lines.append(f"Gait speed: {ctx.gait_speed_m_s:.2f} m/s "
                     f"(interpret truncation flags relative to this)")
    if ctx.phase is not None:
        n = ctx.phase.n_cycles
        note = "phase-windowed (swing/stance) flags" if n >= 1 else "no gait cycles found -> global fallback"
        warn = "  -- TOO FEW CYCLES: low confidence, asymmetry suppressed" if 0 < n < 2 else ""
        lines.append(f"Gait cycles detected: {n}  ({note}){warn}")
    if not findings:
        lines.append("No threshold-grade signature flags triggered.")
    for f in findings:
        lines.append("")
        lines.append(f"[{f.confidence.upper()}] {f.title}")
        lines.append(f"    {f.coordinate} = {f.observed}  (flag: {f.threshold})")
        lines.append(f"    consistent with: {'; '.join(f.interpretations)}")
        for c in f.caveats:
            lines.append(f"    caveat: {c}")
    lines.append("")
    lines.append("General caveats:")
    for c in GLOBAL_CAVEATS:
        lines.append(f"  - {c}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .kinematics import read_storage, summarize
    ap = argparse.ArgumentParser(description="Clinical signature flags from an OpenSim .mot")
    ap.add_argument("--mot", required=True)
    ap.add_argument("--speed", type=float, default=None, help="Gait speed (m/s) for context")
    args = ap.parse_args(argv)

    time, coords, meta = read_storage(args.mot)
    summary = summarize(time, coords, meta)
    ctx = Context(gait_speed_m_s=args.speed)
    print(format_findings(detect(summary, ctx), ctx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
