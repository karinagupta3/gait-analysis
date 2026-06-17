# Clinical movement research — single-phone 2D screening

_Generated 2026-06-17 by a 6-agent research workflow (squat, sit-to-stand, single-leg, jump/balance/TUG, 2D-validity + synthesis). Cited norms; honest 2D feasibility ranking._

I'll synthesize this research into an implementation-ready build plan. This is a synthesis task — the research is already provided, so I'll work directly from it without needing tools.

mcp__visualize__read_me isn't needed here; the deliverable is a developer-facing plan returned as text. Let me produce it.

# BUILD PLAN — Adding Clinical Movements to a Single-Phone 2D BlazePose Screening Tool

## Guiding principle (the ranking rule)
2D single-camera markerless is **excellent for timing/counts and large in-plane SAGITTAL angles**, **acceptable-as-a-flag for FRONTAL projection angles only as within-session CHANGE**, and **invalid for absolute frontal valgus, rotation, and any force/moment**. So I rank by `(clinical value) x (honest 2D feasibility)`, which deliberately pushes timing/sagittal tests above the frontal-plane valgus tests the user is most excited about. I'm being explicit where my ranking diverges from stated interest.

---

## (1) RANKED SHORTLIST — what to build first

**Tier 1 — Build now (high value, high honest feasibility, reuses your sagittal pipeline)**

1. **Five Times Sit-to-Stand (5xSTS)** — side view. Timer + rep state machine. Strongest evidence base, hardest clinical cutoffs, near-perfect 2D agreement (timing ICC 0.92–0.99). Reuses gait pipeline.
2. **30-Second Sit-to-Stand (30CST)** — side view. Same state machine, fixed 30s window, output = count. CDC STEADI fall-risk tool. No floor effect for fitter patients.
3. **STS leg-power estimate (Alcazar equation)** — side view. A *free add-on* riding on #1/#2: feed reps/time + entered height & mass into a validated equation. Highest-value single derived metric (power predicts frailty/mortality better than strength).
4. **Bilateral / overhead deep squat — SAGITTAL screen** — side view. Depth, knee/hip flexion, trunk lean, tibia inclination, tempo, heel rise. Sagittal = the plane 2D does best. Directly extends gait module.
5. **Timed Up and Go (TUG)** — side/oblique view. Total time + subphases. Canonical fall-risk cutoff (≥13.5s), excellent 2D timing validity. Reuses gait pipeline for the walk segment.

**Tier 2 — Build next (high value, but feasibility caveats you must surface in UI)**

6. **Single-Leg Stance balance time** — front view. Output is a *duration* with clean keypoint-detectable stop events (high feasibility); sway is low-confidence, so ship hold-time + fault detection only.
7. **Single-Leg Squat (SLS) FPPA** — front view. The user's interest. Feasible **only as within-session FPPA excursion + L/R asymmetry**, never absolute valgus. This is where honesty discipline matters most.
8. **Lateral Step-Down (Piva 0–6)** — front view. Same frontal cautions as SLS; value is replacing unreliable human visual grading with reproducible continuous angles, then mapping to the familiar 0–6 category.
9. **STS trunk-flexion / quality layer** — side view. Free sagittal add-on to #1/#2 (peak trunk lean, rise smoothness). Skip the symmetry sub-metric (needs front view, no validated cutoff).

**Tier 3 — Later / specialist (good 2D fit but narrower use or needs extra hardware/setup)**

10. **Countermovement Jump (CMJ) via flight time** — side view. Very phone-friendly *if* you can capture 240fps slow-mo; athletic/RTS niche.
11. **Single-Leg Hop battery (LSI)** — side view. Needs an in-frame scale reference; ACL RTS niche.
12. **Drop Vertical Jump FPPA / LESS** — front view. ACL screen, but absolute valgus invalid; prefer the LESS checklist. High fps required.

**Deliberately deprioritized / honesty-flagged**
- **Trendelenburg / contralateral pelvic drop** — feasibility MEDIUM at best: the effect (single-digit degrees) is near the BlazePose hip-landmark jitter floor. Build only after frontal pipeline is mature; report change/asymmetry, never absolute degrees.
- **FMS deep-squat 0–3 ordinal score** — even trained humans have weak per-item reliability (weighted kappa ~0.0–0.44). Do NOT ship an "automated FMS score." Surface the continuous components and a tentative score only. Pain (score 0) is patient-reported, never vision-detected.
- **1-Minute STS** — fine mechanically, but its clinical purpose is SpO2 desaturation, which a phone can't capture. Low priority; if built, show rep count + fatigue only, no cardiorespiratory verdict.

---

## (2) PER-MOVEMENT IMPLEMENTATION SPECS

Shared conventions: BlazePose **Heavy**; ≥60fps (≥120–240fps for jumps/landings); camera perpendicular & level on a tripod; low-pass filter landmarks at **4–6 Hz** before differentiating; landmark indices — shoulders 11/12, hips 23/24, knees 25/26, ankles 27/28, heels 29/30, foot-index 31/32, wrists 15/16. Angle helper: `angle(A,B,C)` = interior angle at B from vectors B→A and B→C.

### 1. Five Times Sit-to-Stand (5xSTS) — SIDE view
- **Metrics:** total time (s); fall-risk flag; per-rep time trend; peak trunk flexion at seat-off; (optional) Alcazar power.
- **Compute:**
  - Rep state machine on mid-hip y and knee angle = `angle(hip,knee,ankle)`. **Hysteresis:** stand when knee >150°, sit when knee <100° (prevents jitter double-counts). One rep = trough→peak→trough.
  - Calibrate seated/standing baselines from first clean rep.
  - Start timer at first upward hip-velocity zero-crossing after "Go"; stop when hip returns to seated baseline after the 5th stand.
  - Seat-off event = hip-y velocity crosses from negative to positive (or knee angle starts increasing).
  - Trunk flexion = angle of (shoulder→hip) vector vs image vertical; report peak per rep.
  - **Validity guard:** detect arm push-off (wrist leaves chest / elbow extension during rise) → flag trial invalid, mirroring the clinical "no hands" rule.
- **Norms/cutoffs:** Age means (community) 60–69y 11.4s, 70–79y 12.6s, 80–89y 14.8s (**Bohannon 2006, Percept Mot Skills 103:215**). Fall-risk: **≥12s** screen-positive (**Tiedemann 2008, Age Ageing 37:430**); **>15s** recurrent fallers (**Buatois 2010, Phys Ther**); **>16s** Parkinson (**Duncan 2011, Phys Ther 91:344**). MDC/MCID ~2.3s.

### 2. 30-Second Sit-to-Stand (30CST) — SIDE view
- **Metrics:** number of full stands in 30s; below-average fall-risk flag; relative STS power (W/kg); per-rep peak power; rep-rate decay (first vs last 10s).
- **Compute:** Identical state machine to #1, window fixed to 30s, output = count. Apply the **"halfway-up" rule:** credit a stand if hip-y crosses the 50% point between seated and standing baselines at t=30s. Arm push-off → score 0 per clinical rule. **Rep counting needs no pixel calibration** (ratios/angles only).
- **Norms/cutoffs:** **Rikli & Jones 1999, J Aging Phys Act 7(2):162** 25th–75th bands (e.g., 60–64 M14–19/W12–17; 80–84 M10–15/W9–14). **CDC STEADI** "below average = high fall risk" = below the band's lower bound (e.g., M65–69 <12, W65–69 <11). Low-power cutoffs: **men <2.53, women <2.01 W/kg** (**Garcia-Aguirre 2025, J Cachexia Sarcopenia Muscle**).

### 3. STS leg-power estimate (Alcazar) — SIDE view (rides on #1/#2)
- **Metrics:** relative power (W/kg) [primary], absolute power (W), power-based risk flags.
- **Compute (Path A, recommended — most validated):** no extra pose needed.
  `Absolute power (W) = [body_mass · g · (0.9·body_height·0.5 − chair_height)] / [(test_time/reps) · 0.5]`; `relative = absolute / body_mass`. For 30s, mean time/rep = 30/reps. `body_mass`, `body_height` user-entered; `chair_height` known (standard **0.43 m**).
  - Path B (pose-derived instantaneous power, **experimental**): COM trajectory from sagittal keypoints, low-pass 4Hz, `P = m·(a_COM + g)·v_COM` per stand. ICC 0.94 vs force plate **but systematic +~38W bias** and assumes bilateral symmetry — label experimental.
- **Norms/source:** Equation **Alcazar 2018, Exp Gerontol 112:38**; validated **Baltasar-Fernandez 2021, Exp Gerontol 158:111652**; cutoffs as in #2.

### 4. Bilateral / overhead deep squat — SAGITTAL — SIDE view
- **Metrics:** squat depth (peak knee flexion); peak hip flexion; trunk forward-lean; tibia inclination (ankle DF proxy); descent tempo; heel rise (binary).
- **Compute:** knee flexion = `angle(hip,knee,ankle)`; hip flexion = `angle(shoulder,hip,knee)`; trunk lean = (shoulder→hip) vs vertical; tibia inclination = (knee→ankle) vs vertical. Bottom = local min hip-y / max knee flexion; rep start/end = near-full knee extension. Rep count via knee-angle threshold crossings (~30°) or hip-velocity zero-crossings. Tempo = frames(start→depth)/fps. Heel rise = upward displacement of heel vs foot-index baseline. In true sagittal the far limb is occluded — **report camera-side limb only**.
- **Norms/cutoffs:** Depth bands partial 0–90°, parallel 90–110°, deep 110–135° knee flexion; deep-squat targets ~124° knee, ~124–125° hip, ~23–26° ankle DF (**Straub & Powers, IJSPT 2024**; verify against Endo/Miura/Sakamoto, J Phys Ther Sci 2020). Ankle DF restriction: shin-to-vertical **<35–38°** on weight-bearing lunge. Trunk lean has **no universal cutoff** — interpret via trunk−tibia difference (>10° hip-biased, <−10° knee-biased). **Honesty:** BlazePose sagittal knee angle MAE ~7° with ~5° underestimation, bias is angle-dependent (overestimates shallow, underestimates deep), test-retest ICC=0.094 (**Yue, Gu & Yu, JMIR preprint 102399, 2026**) → prefer **depth band, within-session change, and L/R symmetry over absolute degrees**.

### 5. Timed Up and Go (TUG) — SIDE / oblique view
- **Metrics:** total time; subphase durations (sit-to-stand, gait, turn, turn-to-sit); gait speed; steps-to-turn / turn duration.
- **Compute:** total = hip rises off chair baseline → hip returns to baseline & stabilizes. Subphases: STS = hip rise + knee extension; gait onset = pelvis horizontal velocity > threshold; turn = direction reversal / shoulder-hip orientation flip (visible even sagittally as body-width change); turn-to-sit mirrors start. Gait speed from pelvis horizontal displacement over the calibrated 3 m. Steps from foot-landmark oscillation peaks. **Gate on landmark confidence during the turn** (subject rotates out of plane) and fall back to timing if angles unreliable.
- **Norms/cutoffs:** **≥13.5s** = elevated fall risk, ~87% sens/spec (**Shumway-Cook 2000, Phys Ther 80:896**); ≥30s ~ dependent mobility; community women 65–85 ~12s reference (**Bischoff 2003**). Markerless single-camera subtask validity r=0.67–0.91 (**Kamnardsiri 2023, PLoS ONE**). **Setup note:** one fixed phone may lose the subject over 3 m + turn — use wider/farther placement; total time + STS + turn are most robust.

### 6. Single-Leg Stance balance time — FRONT view
- **Metrics:** hold time eyes-open; hold time eyes-closed (instruction toggle — phone can't verify eyes); best-of-3 & mean; L/R asymmetry; (optional, low-confidence) sway.
- **Compute:** start when raised-foot ankle leaves ground (ankle y/velocity vs stance ankle); **stop on any fault:** raised foot touches down/touches stance leg, wrist leaves hip beyond threshold, stance ankle x shifts/hops, or excessive trunk lean. Cap at 30s to match norm tables. Sway = SD of hip-midpoint x during hold (label low-confidence — near jitter floor).
- **Norms/cutoffs (eyes-open):** 60–69y ~27s, 70–79y ~17–19s, 80–99y ~8.5–9s (**Springer 2007, J Geriatr Phys Ther 30(1):8**; **Bohannon 2006**). Fall risk: **<5s eyes-open** (**Vellas 1997**, RR ~2.13). Clinical banding: >20s low, 5–20s moderate, <5s high risk.

### 7. Single-Leg Squat (SLS) FPPA — FRONT view
- **Metrics:** **FPPA EXCURSION** (depth minus standing baseline) [the trustworthy one]; medial knee displacement (normalized, as flag); pelvic drop; trunk lateral lean; **L/R FPPA asymmetry**.
- **Compute:** FPPA = angle between (hip→knee) and (ankle→knee) on the stance leg in the image plane. Depth frame = stance-leg knee-flexion peak / hip-y min; baseline = single-leg standing frame. Pelvic drop = angle of (left-hip→right-hip) line vs horizontal. Trunk lean = (shoulder-mid→hip-mid) vs vertical. Knee-over-toe flag = knee_x vs ankle_x / foot-index_x. **Exclude balance-loss/foot-repositioning frames.**
- **Norms/cutoffs:** FPPA normal **7–13°**, **>13°** excessive DKV (**Munro et al.**; applied **Jamaludin 2022, IJSPT** normal ~10.3° vs excessive ~16.0°). **HARD HONESTY:** single-camera markerless **absolute** valgus is invalid — ~18.8–19.7° overestimation, no concurrent validity vs 3D; **only normalized change correlates** (r 0.55–0.71, <5° error) (**Asaeda 2024, Heliyon**; **Lopes 2018, JOSPT** advises against 2D for accurate frontal angles). BlazePose single-leg-stance landmark validity poor (ICC=0.136, Yue 2026). **MDC ~7.5–9°** (Munro 2012) → only flag a change/asymmetry if it exceeds ~8–9°. Report as **screen-positive/negative flag**, label "frontal-plane projection angle (2D estimate), not measured 3D knee valgus."

### 8. Lateral Step-Down (Piva 0–6) — FRONT view
- **Metrics:** total deviation score 0–6 → Good (0–1)/Fair (2–3)/Poor (≥4); arm strategy (+1); trunk movement (+1); pelvis plane (+1); knee position (+1 medial to 2nd toe, +2 medial to foot border); steady stance (+1).
- **Compute:** step ~20cm, hands on hips, 5 reps, contralateral heel taps floor. Knee position = knee_x vs foot-index/ankle_x at lowest point. Pelvis plane = hip-hip line vs horizontal (threshold ~5–10°). Trunk = (shoulder-mid→hip-mid) vs vertical. Arm strategy = wrist displacement from hip beyond threshold. Steady stance = premature contralateral-foot contact / large COM sway. Event = contralateral-ankle vertical min (touch frame). **Output both continuous angles (reproducible) AND the 0–6 category** for clinical familiarity.
- **Norms/cutoffs:** Categories from **Piva 2006, BMC Musculoskelet Disord 7:33** (Good 0–1, Fair 2–3, Poor ≥4). Human inter-rater agreement only fair-moderate (kappa 0.40–0.67; **Mansfield 2021**) — which is the argument FOR objective quantification.

### 9. STS trunk-flexion / quality layer — SIDE view (free add-on to #1/#2)
- **Metrics:** peak trunk flexion (deg); movement-strategy class (momentum-transfer / exaggerated-trunk-flexion / dominant-vertical-rise); rise smoothness (time-to-peak-velocity / jerk). **Skip** the frontal symmetry sub-metric.
- **Compute:** trunk flexion = (shoulder→hip) vs vertical, peak per rep, in the SAME sagittal recording as the timed test. Smoothness from hip/COM ascent velocity profile.
- **Norms:** healthy peak trunk flexion at seat-off ~**58.8° (SD 17.9)** (**Frontiers Bioeng Biotechnol 2025**); larger lean = knee-extensor weakness compensation (medRxiv 2023 simulation). **No validated 2D symmetry cutoff exists** — don't ship a force-symmetry number.

### Tier 3 quick specs
- **CMJ (side):** height = `9.81·t²/8`, t = flight time from foot-off to foot-on frames. **Do not pixel-track height.** Requires **240fps** (My Jump uses 240fps; at 30fps one frame ≈33ms ≈ several cm error). Validity ICC >0.97 (**Balsalobre-Fernandez 2015, J Sports Sci 33:1574**). Restrict arm swing.
- **Hop battery (side):** distance = horizontal toe/heel displacement takeoff→**held** landing, scaled by an **in-frame reference** (scale calibration essential). LSI = involved/uninvolved ×100; ≥90% conventional pass (**Noyes 1991, AJSM 19:513**) — but pair with height-normalized norms (**Brumitt 2018/2020**) since LSI masks bilateral deficits.
- **DVJ (front):** FPPA at IC and peak flexion + excursion; prefer the **LESS checklist** (≥5 errors: sens 86%/spec 64%, **Padua 2009/2015**) over a single angle. ≥120fps.

---

## (3) WHAT SINGLE-CAMERA 2D MUST NOT CLAIM

State this in the UI and in any exported report:
- **No diagnosis.** It is a screen that flags for clinical referral, not a diagnostic.
- **No true 3D joint rotation / out-of-plane angles.** Transverse/rotational measures have RMSE ~21°.
- **No absolute frontal-plane knee valgus.** MediaPipe overestimates ~19° with no concurrent validity vs 3D; only **within-session normalized change** is valid. FPPA is a 2D *projection*, not anatomical valgus, and conflates true valgus with hip adduction/internal rotation.
- **No forces, moments, or power-as-ground-truth.** GRF/joint moments are invisible to a camera. The ACL prospective predictor in Hewett 2005 was the knee-abduction *moment*, not the visual angle — FPPA is a PFP/movement-quality screen, NOT a validated ACL predictor.
- **No weight-bearing/limb-load symmetry.** Sagittal can't resolve which limb bears more load (far limb occluded; GRF invisible). STS "symmetry" is an unvalidated flag only.
- **No absolute spatial distances (cm step length, cm jump height) without an in-frame scale reference.** Report normalized/relative, or require calibration.
- **No SpO2 / cardiorespiratory verdict** (relevant to 1-min STS).
- **No pain detection** (FMS score 0 is patient-reported).
- **No force-plate-equivalent sway / balance.**
- **Longitudinal comparisons require identical fixed camera setup;** absolute degrees drift across sessions (test-retest ICC as low as 0.094 sagittal squat; frontal inter-rater ICC 0.03–0.59).
- **Honest defaults:** prefer **depth bands, within-session change, L/R asymmetry, timing, and rep counts** over absolute degrees. Apply **MDC thresholds (~8–9° frontal)** before declaring any change real. Surface a low-landmark-confidence warning when it fires.

---

## (4) UX RECOMMENDATIONS — movement picking & camera guidance

**Movement picker organized by camera view (so the user never sets up wrong):**
- Group the menu into **two columns: "Side view (phone beside you)" and "Front view (phone facing you)."** This makes the camera requirement structural, not a forgettable instruction.
  - **Side:** Gait (existing), 5xSTS, 30CST, Deep squat, TUG, CMJ, Hop.
  - **Front:** Single-leg stance, SLS FPPA, Lateral step-down, DVJ.
- Optionally group by **clinical goal** as a secondary filter: *Fall risk* (5xSTS, 30CST, TUG, SLS-stance), *Strength/power* (30CST+Alcazar, CMJ), *Movement quality / knee* (deep squat, SLS FPPA, step-down), *Return-to-sport* (CMJ, hop, DVJ).

**Per-movement setup gate (before recording starts):**
1. Big icon + one line: **"Place phone to your SIDE"** or **"Place phone FACING you,"** with a stick-figure diagram.
2. Concrete placement: perpendicular to motion, **lens at hip height (~1 m)**, **~3 m away**, full body in frame, **on a tripod / propped (do not hand-hold)**.
3. Frame rate hint where it matters: "Use slow-motion (120–240fps) for jumps/landings."
4. **Live framing check:** before "Go," verify all required landmarks are visible with adequate confidence and the subject is square to the camera; block start otherwise. For frontal tests, add a "stand square, camera perpendicular" alignment check (small trunk rotation corrupts FPPA).
5. For STS: enter **height, weight** (for power) and confirm **chair height** (default 0.43 m). For hop: confirm the in-frame scale reference is placed.

**During/after capture:**
- Show the detected events (rep markers, takeoff/landing, seat-off) overlaid so the user can sanity-check counts.
- Report results as **band/flag + value + comparison-to-norm + source**, e.g., "5xSTS = 13.2s — above the 12s fall-risk threshold (Tiedemann 2008)." Always show the **"Screen, not a diagnosis"** banner.
- For frontal tests, present **excursion + asymmetry + screen-positive flag**, never a bare "valgus = X°," and gray out / footnote absolute degrees.
- Two-capture flows where clinically meaningful: **heels-flat vs heels-elevated** squat retest (isolates ankle); **eyes-open vs eyes-closed** single-leg stance (toggle instruction).
- Lock camera setup metadata to the patient record and **warn if a longitudinal re-test's setup differs**.

---

## Recommended build order (one sprint each)
1. **STS family (5xSTS + 30CST + Alcazar + trunk-flexion layer)** — one side-view recording yields four tests; highest value-per-effort, reuses your pipeline.
2. **Deep squat (sagittal)** + **TUG** — both extend the gait/sagittal pipeline.
3. **Frontal pipeline foundation** (front-view capture, alignment check, FPPA-excursion engine, MDC gating) → then **Single-leg stance**, **SLS FPPA**, **Lateral step-down**.
4. **Tier 3 athletic tests** (CMJ → hop → DVJ) once high-fps capture + scale-reference calibration exist.

The single biggest leverage move: ship the **STS recording once → 5xSTS, 30CST, leg-power, and trunk-flexion quality** all at once. The single biggest honesty risk: shipping any **absolute frontal valgus number** — constrain all frontal output to within-session change, asymmetry, and pass/fail flags with the ~8–9° MDC gate.