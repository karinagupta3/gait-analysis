"""Plain-language interpretation + treatment guidance per finding (cited).

Keyed by signature/task rule_id. The report renders this under each flag so a clinician
sees, in plain words: what it likely means, what the evidence says to do about it, and
how to know a later change is real (not noise). Full citations: docs/07.

HONESTY: these are evidence-informed options, not prescriptions; effect sizes and the
"strength of evidence" notes are carried so nothing is over-sold.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Guidance:
    meaning: str       # plain-language what the pattern suggests
    treatment: str     # evidence-based options (with effect sizes / strength)
    tracking: str      # what change counts as real on re-test (MCID/MDC)


# Shared "what change is real" notes (docs/07 §rehab monitoring).
_TRACK_SPEED = ("Re-test the same way; a gait-speed change of ~0.10-0.17 m/s is the "
                "well-supported 'real and meaningful' band (Bohannon 2014).")
_TRACK_KINEM = ("Joint-angle change must exceed the between-session error (often >5 deg "
                "for markerless); average >=3 trials and standardize setup before calling it improvement.")

TREATMENT_KB: dict[str, Guidance] = {
    # ---- gait ----
    "stiff_knee_swing": Guidance(
        "The knee doesn't bend enough in swing, so the leg clears the ground poorly "
        "(tripping risk). Often rectus femoris over-activity/spasticity or a weak push-off.",
        "Address the cause: for spastic rectus femoris, tone management (botulinum, "
        "rectus transfer in CP) or pre-swing push-off training; confirm with a fast Duncan-Ely.",
        _TRACK_KINEM),
    "crouch_knee": Guidance(
        "The knee stays bent through stance (crouch). Multifactorial -- hamstring length, "
        "hip-flexor contracture, or plantarflexor weakness.",
        "Don't assume short hamstrings -- model muscle length or use a popliteal-angle test "
        "first (most 'crouch' hamstrings are NOT short; Arnold 2006). Treat the actual limiter.",
        _TRACK_KINEM),
    "reduced_hip_extension": Guidance(
        "The hip doesn't reach a neutral/extended position at the end of stance, often with "
        "increased anterior pelvic tilt -- consistent with tight hip flexors.",
        "Hip-flexor stretching restored dynamic peak hip extension in older adults "
        "(Kerrigan RCT: +1.6 deg static, longer stride). Confirm with a pelvis-controlled Thomas test.",
        _TRACK_KINEM),
    "foot_drop": Guidance(
        "The foot doesn't lift in swing (toe drag / slap at contact) -- dorsiflexor weakness "
        "or equinus.",
        "AFO/FES for clearance; dorsiflexor strengthening; treat spastic/contracted calf "
        "(stretch, tone management) if equinus. Gait retraining can help carry-over.",
        _TRACK_KINEM),
    "reduced_knee_excursion": Guidance(
        "The knee moves through a smaller arc than normal ('stiffening') -- classic in knee OA "
        "and pain-guarded gait. Strongly affected by walking slowly.",
        "Pain management + quadriceps/hip strengthening; gait-speed normalization. For medial "
        "knee OA, gait modification (lateral trunk lean / toe-in) cut the knee load ~20% with "
        "WOMAC pain -29% in small trials (Shull 2013) -- promising but short-term.",
        _TRACK_SPEED),
    "rom_asymmetry": Guidance(
        "One side moves differently than the other -- a hint of one-sided injury, pain, or "
        "weakness (or, for arm swing, an early Parkinson's marker).",
        "Find and treat the affected side (strength, mobility, or pain). Symmetry biofeedback "
        "improves symmetry acutely, best-evidenced in subacute stroke.",
        "Compare the symmetry ratio over time, but only trust shifts larger than the measure's "
        "test-retest error; small ratios near 1.0 are easily within noise."),
    # ---- squat ----
    "squat_depth_limited": Guidance(
        "The squat is shallow. The most common mechanical limiter is ankle dorsiflexion, then "
        "hip flexion ROM / impingement; strength and motor control also play in.",
        "First rule mobility: knee-to-wall (ankle), hip flexion check. Heel elevation is an "
        "immediate accommodation; build ankle/hip mobility; squat to the depth you can control.",
        "Re-test peak knee/hip flexion; expect change only beyond the markerless error band."),
    "dynamic_valgus": Guidance(
        "The knee collapses inward (hip adduction) under load -- linked to ACL injury risk and "
        "patellofemoral pain; relates to hip-abductor weakness and motor control.",
        "Hip-abductor/external-rotator strengthening (endurance-biased) PLUS neuromuscular/"
        "motor-control retraining (cue 'knees out'). Honest caveat: strength reliably cuts PAIN, "
        "but its effect on the valgus ANGLE itself is inconsistent -- pair strength with retraining.",
        "Re-test peak hip_adduction / frontal-plane projection angle; frontal-plane markerless "
        "data is lower-confidence, so look for clear changes only."),
    "ankle_df_restriction": Guidance(
        "Limited ankle bend in the squat -- forces heel rise, forward lean, or contributes to "
        "butt wink and knee valgus.",
        "If the block is the JOINT, use talocrural mobilization (calf stretching won't fix it); "
        "if MUSCLE, gastroc + soleus stretching (>=30s) +/- soft-tissue work. An 8-week combined "
        "program gave clinically significant DF gains (RCT).",
        "Track knee-to-wall distance (cm; ~1 cm ~ 3.6 deg); side-to-side difference is most useful."),
    "butt_wink": Guidance(
        "Near the bottom the pelvis tucks under and the low back rounds. Usually an ankle or hip "
        "mobility limit running out -- NOT typically tight hamstrings (a common myth).",
        "Fix the limiter (ankle/hip mobility), squat to a controllable depth, heel elevation as a "
        "stop-gap. Note: some lumbar flexion is normal; injury concern scales with LOAD, not "
        "bodyweight squats (evidence is weak/contested).",
        "Track the depth at which the reversal appears as mobility improves."),
    "squat_asymmetry": Guidance(
        "The two legs aren't sharing the squat evenly -- often a weight-shift away from a painful "
        "or weaker limb.",
        "Address the limited side (strength, mobility, or pain); unilateral work (split squats) "
        "and real-time feedback on even loading.",
        "Re-test L/R depth difference; the 15% cut-point is borrowed from hop testing -- a hint, "
        "not a hard line."),
    # ---- sit-to-stand ----
    "sts_slow": Guidance(
        "Rising from a chair is slow -- a strong, simple marker of lower-limb weakness and fall "
        "risk in older adults.",
        "Progressive lower-limb strengthening (sit-to-stand reps, leg press); add balance/"
        "postural training if the slowness is in the stabilization phase after lift-off.",
        "5xSTS age norms: 60s ~11.4s, 70s ~12.6s, 80s ~14.8s; >12s warrants assessment, >15s = "
        "elevated fall risk. Re-test the same chair/protocol."),
}


def guidance_for(rule_id: str) -> Guidance | None:
    return TREATMENT_KB.get(rule_id)
