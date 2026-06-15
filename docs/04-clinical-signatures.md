# Clinical Signatures from Gait/Movement Kinematics

How to turn the OpenSim/OpenCap coordinate set (the full joint-angle data — `pelvis_tilt/list/rotation`
+ `tx/ty/tz`, `hip_flexion/adduction/rotation`, `knee_angle`, `ankle_angle`, `subtalar_angle`,
`mtp_angle`, `lumbar_extension/bending/rotation`, `arm_flex/add/rot`, `elbow_flex`, `pro_sup`, both
sides) into **flags** for tightness, weakness, neuro/motor deficits, concussion/vestibular instability,
pain/guarding, and injury-risk patterns.

This is the evidence base behind `src/gait_analysis/analysis/signatures.py`. It is a research
decision-support reference, **not** a diagnostic manual.

---

## The non-negotiable honesty frame (read first)

Five rules constrain every signature below. They are baked into `signatures.py`:

1. **Deviation ≠ cause.** A joint-angle deviation localizes *where/that* something is wrong, never *why*.
   Reduced peak hip extension is equally consistent with tight hip flexors, weak hip extensors, an
   anterior-pelvic-tilt offset, or simply slow walking — these are *mechanically degenerate*
   (Goldberg & Neptune, *Gait Posture* 2006; Arnold et al. 2006 showed **~80% of "crouch" hamstrings are
   normal/long, not short**). Report the *pattern* and *candidate* cause, then recommend a confirming test.
2. **Speed confounds almost everything.** Slower gait intrinsically reduces flexion peaks, push-off, and
   excursions. Always capture and report **gait speed**; "truncation" near a control's slow-speed value is
   not pathology. (Hamstring/hip length are speed-dependent — Agarwal-Harding/Schwartz/Delp 2010.)
3. **Markerless reliability is plane-dependent.** Sagittal (flexion/extension) is trustworthy
   (~3–5°); **frontal is moderate; transverse/rotation is near the noise floor** (OpenCap mean ~4.5°, but
   hip rotation locally >10°; soft-tissue artifact corrupts transverse even in marker-based mocap). So:
   **hard thresholds only on sagittal coordinates; frontal/transverse = advisory.**
4. **Asymmetry vs absolute.** L/R asymmetry is powerful for *unilateral* problems and robust to absolute
   offset error — but it **fails for bilateral disease** (both sides truncated → ratio ≈ 1.0). Pair
   asymmetry flags with absolute thresholds.
5. **Not a diagnosis; not a validated individual predictor.** For injury risk especially, group-level
   associations rarely translate to individual prediction (Bahr 2016). Flags are hypotheses to confirm.

---

## A. Normative sagittal gait values (the comparison baseline)

Healthy-adult reference (Perry & Burnfield; AAPM&R Biomechanics of Normal Gait). These are what
`kinematics.py` compares against and what defines "truncated/flatlined."

| Coordinate | Normal peak / range | Timing | Markerless reliability |
|---|---|---|---|
| `pelvis_tilt` (ant.) | ~7–13° mean tilt; ~2–4° cyclic excursion | 2×/cycle | moderate |
| `pelvis_list` | ~±4–5° | drop toward swing limb | moderate |
| `pelvis_rotation` | ~±5–8° | — | **poor (advisory)** |
| `hip_flexion` | +30° flex (IC) → **~10° extension** (terminal stance) | ext peak ~50% | good |
| `hip_adduction` | ~+5–10° add → ~5° abduction (swing) | — | moderate |
| `hip_rotation` | ~±5° | — | **poor (advisory)** |
| `knee_angle` | **~15° loading-response flex**; ~0° terminal stance; **~60–65° swing peak** | swing peak ~73% | good (sagittal) |
| `ankle_angle` | **~10° dorsiflexion** (stance); ~15–20° plantarflexion (toe-off) | DF ~45% | good (sagittal) |
| `lumbar_rotation` / `bending` | ~±5–7° / ~±5° | counter-rotates pelvis | **poor (advisory)** |
| `arm_flex` (shoulder) | ~20–25° excursion | counter-phase to ipsilateral leg | moderate |

**"Truncated/flatlined" rule:** flag a *sagittal* curve when peak excursion < ~50–60% of normal AND the
deviation exceeds the OpenCap per-task error band (~4–5°). Never hard-threshold transverse/frontal.

---

## B. Muscle TIGHTNESS signatures

| Tight muscle | Kinematic signature | Coordinate(s) | Threshold (sagittal) | Clinical test | Conf. |
|---|---|---|---|---|---|
| **Hip flexors** (iliopsoas/RF) | ↓ peak hip extension in terminal stance; ↑ anterior pelvic tilt | `hip_flexion` (min), `pelvis_tilt` | hip never reaches neutral (min >0°; normal ~ −10°); APT **>8–10°** (CP) | Thomas/Staheli (control pelvic tilt) | mod |
| **Hamstrings** | ↑ knee flexion at IC / ↓ terminal-swing extension; crouch; ↓ ant. pelvic tilt | `knee_angle` (IC), `pelvis_tilt` | knee flex at IC **>20°** (normal ~2–5°); crouch = stance min **>30°** | Popliteal angle (functional) | mod–high |
| **Gastroc/soleus** (equinus) | ↓ stance dorsiflexion; early heel rise; toe-walk; knee recurvatum | `ankle_angle`, `knee_angle` | DF peak **<5° (with compensation) / <10°**; normal ~10° | Silfverskiöld (knee ext vs flex) | mod–high |
| **Rectus femoris** (stiff-knee) | ↓ **and delayed** peak knee flexion in swing; ↓ knee ROM; low toe-off knee-flexion velocity | `knee_angle` (peak + timing) | swing peak **≤45° (CP) / <55°**; normal ~60–65° | Duncan-Ely (fast velocity) | high |
| **Hip adductors** | ↓ hip abduction; scissoring; narrow step width | `hip_adduction`, step width | qualitative (adduction bias); step width < ~3–8 cm | passive abduction (knee ext vs flex) | mod |
| **ITB / TFL** | ↑ stance hip adduction; contralateral pelvic drop; (lateral knee pain in runners) | `hip_adduction`, `pelvis_list` | ITBS runners ~10.4° vs 7.9° hip adduction (Ferber 2010) | Ober (**not validated** — reflects glut med/capsule, Willett 2016) | low–mod |

Key nuances: stiff-knee is driven more by **low knee-flexion velocity at toe-off** (Goldberg 2003: 15/18
limbs) than by swing-phase RF EMG (which Goldberg's 407-limb study found does *not* reliably correlate);
RF length must be computed from **combined hip+knee** kinematics, not `knee_angle` alone (Jonkers 2006
"uncoupling"). For hamstrings, posture ≠ short muscle — confirm with modeled MT length, not the angle.

Refs: Hip flexor contracture validity (JNER 2011, ES 0.74); Kerrigan 2003 stretch RCT (hip ext 6.1°→7.7°);
Whitehead 2007 (terminal-swing knee 25.6° vs 2°); equinus thresholds (MDPI Children 2022; PubMed 23465759);
Jonkers 2006 (PMID 16399519); Goldberg 2003/2004; Ferber 2010 (PMID 20118523); Willett 2016 Ober anatomy.

---

## C. Muscle WEAKNESS signatures (weakness shows as *compensation*, not a strength number)

| Weak muscle (nerve) | Signature | Coordinate(s) | Flag | Conf. |
|---|---|---|---|---|
| **Glute medius** (sup. gluteal) | contralateral pelvic drop ± compensated trunk lean over stance limb (Duchenne) | `pelvis_list`; `lumbar_bending`, `pelvis_tz` | drop **>4–5°** (normal ±4–5°) | mod (frontal) |
| **Glute maximus** (inf. gluteal) | posterior trunk lurch at initial contact | trunk / `lumbar_extension`, `hip_flexion` | qualitative, timed to IC | mod |
| **Quadriceps** (femoral, L2–4) | knee hyperextension (recurvatum); lost loading-response flexion; forward trunk lean | `knee_angle` | LR flexion ≪15°; hyperext <0° (sig >5–10°) | mod–high |
| **Plantarflexors** (tibial, S1–2) | lost push-off; ↓ toe-off plantarflexion; "calcaneal" gait; ↓ speed | `ankle_angle`, ankle power, speed | TO PF ≪15–20°; power drop (severe >60% strength loss) | mod–high |
| **Dorsiflexors / TA** (deep fibular, L4–5) | **foot drop**: ↓ swing dorsiflexion; foot slap; steppage (↑ swing hip/knee flexion) | `ankle_angle` (swing), `hip_flexion`/`knee_angle` | swing ankle stays plantarflexed (max <5°) | mod |
| **Hemiparesis / UMN** | reduced/asymmetric arm swing | `arm_flex_r` vs `_l` | amplitude asymmetry; lost 1:1 ratio | mod |

Gait is robust to hip/knee **extensor** weakness but sensitive to **plantarflexors, hip abductors, hip
flexors** (van der Krogt 2012 — tolerates ~40% generalized weakness before normal gait fails). Gait reveals
weakness via compensation and *demand*; it does **NOT** measure maximal strength (MVC) or muscle size —
those need dynamometry/MMT/imaging. OpenSim Static-Optimization muscle-force estimates are *model
hypotheses*, best distally (ankle), weak at the knee (under-predicts co-contraction).

Refs: Trendelenburg/steppage (StatPearls NBK541094/NBK547672); van der Krogt 2012 (S0966636212000392);
Ong 2019 plantarflexor sim (pcbi.1006993); Heintz 2007 SO-vs-EMG.

---

## D. NEURO / MOTOR signatures

| Pattern | Coordinate | Signature | Numbers | Conf. |
|---|---|---|---|---|
| **Foot drop / steppage** | `ankle_angle` (swing) + `hip`/`knee` (swing) | ankle stays plantarflexed in swing; compensatory ↑ hip/knee flexion | swing DF fails to reach ~0° | high (qualitative) |
| **Reduced arm swing — PD** | `arm_flex_r` vs `_l` | **asymmetry > amplitude**; early/prodromal marker | asymmetry ~14% vs ~5% controls; meta SMD 0.84; ROM ↓~7° | high (asymmetry) |
| **Reduced arm swing — stroke** | `arm_flex` paretic | reduced amplitude, disrupted L/R coordination | speed-dependent; no clean cutoff | mod |
| **Stiff-knee gait** | `knee_angle` peak swing | reduced peak swing flexion | **<44.3°** identifies SKG (normal ~60°); between-limb Δ best | high |
| **Sit-to-stand / bradykinesia** | `hip/knee/ankle` triple-extension + trunk, vs time | prolonged duration; slowed **flexion→extension switch**; ("chaotic" is overstated → "slowed/segmented") | rise time ↑; multi-attempt = advanced disease | mod–high |
| **Spasticity** | any joint + slow/fast passive | **velocity-dependent catch** (Lance 1980) | distinguish via Tardieu R2−R1 or fast vs slow gait | n/a (needs passive test) |

Spasticity vs **fixed contracture** vs dynamic tightness **cannot** be separated from kinematics alone —
it requires a velocity manipulation: Tardieu (R2−R1; Ashworth can't), faster walking unmasks spasticity
(van der Krogt 2009/2014), or a diagnostic nerve block / exam-under-anesthesia (residual ROM = fixed
contracture; McMulkin 2008: under-anesthesia DF gain ~8.5–9.5° in <11yr = the dynamic component).

Refs: Lewek 2010 (PMID 19945285); arm-swing meta (PMC10695845); SKG threshold (S0268003324001839);
Lance 1980 (PMC4229996); Tardieu/Ashworth (Patrick & Ada 2006); McMulkin 2008 (PMID 18841059).

---

## E. CONCUSSION / VESTIBULAR — with the corrections to the original hypotheses

| Claim | Coordinate | Verdict | Detail |
|---|---|---|---|
| Dual-task ↓ speed/cadence/stride + **↑ ML sway & variability** | `pelvis_tz` (ML), step-width var. | **SUPPORTED** (best-validated) | Howell: ML COM displ. ↑ under dual-task, aOR 2.43, persists ~2 mo; effect sizes small (η² ~0.10–0.18); one multicenter study found **no** group ML difference → probabilistic, not deterministic |
| Vestibular = **"flatlined spine"** (reduced `lumbar_rotation`) | `lumbar_rotation`/`bending` | **PARTIAL / MIS-LOCATED** | En-bloc trunk *stiffening* (reduced axial rotation) is real in balance/perceptual disorders, BUT vestibular loss usually gives **increased** trunk/ML sway, not a flatline. The genuine "guarding" is **reduced HEAD AP excursion** (Wilkerson 2021: head AP −2.3 cm, ML pelvis +7.5 cm). Expect **dissociation**, not both flat. |
| Balance/sway (tandem, single-leg, foam/eyes-closed) | COM/`pelvis_tz` RMS + velocity | **SUPPORTED** | sway ↑, most sensitive on foam/eyes-closed; instrumented sway outlasts BESS recovery to ~90 days |

**Practical concussion signature:** dual-task **↑ `pelvis_tz` ML range + ML velocity + step-width
variability** with **↓ gait speed and ↓ sagittal COM excursion** — and, if you have a head segment,
**↓ head AP excursion**. Do **not** key it on a flat `lumbar_rotation`.

Refs: Howell 2013/2018 (PMID 23643687, 29457997); multicenter null (PMID 29966907); Wilkerson 2021
(PMC8384176); Buckley/Howell systematic review (PMID 32132473).

---

## F. PAIN / JOINT GUARDING — with the direction correction

| Claim | Coordinate | Verdict | Detail |
|---|---|---|---|
| Antalgic: ↓ stance time / loading on painful limb | per-limb stance time, vGRF | **SUPPORTED** | single-support asymmetry ↑; vGRF "M" flattens; speed-confounded |
| **Truncated joint ROM on affected side** | `knee_angle_r` vs `_l` | **SUPPORTED** | knee OA: peak swing flex ~42° vs 49° (~7°); excursion 13.6° vs 16.2°; **up to halved (17.4°→9.7°) in provocative tasks** → use squat/stairs to expose it. Hip OA truncates **extension**, not the flexion peak. |
| Trunk/pelvis shift **away** from painful limb | `pelvis_tz`, `lumbar_bending` | **BACKWARDS for hip** | Hip pain → **Duchenne lean TOWARD** the affected stance limb (shortens the joint moment arm). Separate "limb **unloading**" (load shifts away, e.g. in a squat) from "**trunk lateral lean**" (toward) — conflating them is the error. |
| Kinesiophobia → guarding | `lumbar_extension` ROM, trunk velocity | **SUPPORTED (weak)** | reduced trunk ROM + co-contraction; **task-specific** fear predicts better than generic TSK |

Refs: knee OA flexion excursion (PMC8229136; JOSPT 2015 instability cohort); hip OA Duchenne (PMC3274426);
antalgic (StatPearls NBK559243); kinesiophobia (PMC11396003; PAIN Reports 2022).

---

## G. INJURY SUSCEPTIBILITY — real risk factors, weak individual prediction

| Risk factor | Coordinate/metric | Threshold | Injury | Evidence |
|---|---|---|---|---|
| Knee valgus / abduction moment (drop jump) | knee frontal angle/moment, `hip_adduction` | ~8° more valgus; KAM ~25 Nm cut | ACL | Hewett 2005 (78%/73% sens/spec, n=9) **— failed to replicate** in Krosshaug 2016 (n=710) |
| Contralateral pelvic drop (running) | `pelvis_list`, `hip_adduction` | ~3° greater; **OR ~1.8 per 1°** | PFP/ITBS/RRI | Bramah 2018 (cross-sectional; strongest single discriminator) |
| Peak hip adduction (running) | `hip_adduction` | ~12° vs 8° | PFP (prospective) | Noehren 2013 |
| Vertical loading rate | AVLR (needs force/IMU) | AVLR >66 BW/s (OR 2.72); 79 vs 66 (TSF) | tibial stress fracture | Davis 2016 (prospective); Milner 2006 — surrogate validity now questioned |
| Limb Symmetry Index | quad strength + hop LSI | ≥90% to clear RTS | ACL re-rupture | Kyritsis 2016 (HR ~4.1); Grindem 2016 (84% risk ↓) — **strongest predictive evidence**, but LSI is a flawed metric |
| FMS composite | screen score | ≤14 | general injury | **weak** — AUC ~0.59, sens ~25%, spec ~86% (Moran 2017) |

**Overarching caveat (Bahr 2016, BMJSM):** even significant group-level risk factors rarely predict
*individual* injury — distributions overlap, PPV is low. Frame outputs as "modifiable movement pattern to
address," **not** "your ACL/stress-fracture risk is X." The ACL-RTS batteries (Kyritsis/Grindem) are the only
ones approaching predictive + interventional validity, and even they rely on the imperfect LSI.

---

## H. What you CANNOT claim from kinematics — and the confirming modality

| To establish... | You also need | Why |
|---|---|---|
| Spastic vs fixed contracture vs dynamic | **Tardieu** (velocity), faster-gait test, dynamic EMG, or nerve block / EUA | velocity-dependence / removing neural drive |
| A muscle is truly *short/slow* | **OpenSim MT-length/velocity modeling** (hip+knee combined) | posture ≠ length (Arnold 2006); RF "uncoupling" (Jonkers 2006) |
| Muscle is *active* when it shouldn't be | **dynamic EMG** (surface/fine-wire) | kinematics show position, not activation |
| Maximal **strength** / capacity | **dynamometry / MMT** | gait = submaximal demand, not MVC |
| Muscle **size / structure / stiffness** | **MRI / ultrasound / shear-wave elastography** | CP gastroc ~20% smaller volume (Handsfield 2016); SWE quantifies passive stiffness in kPa |
| Localize one-joint vs two-joint | knee-position tests (Thomas knee-flexed; Silfverskiöld) | biarticular muscles slacken with the second joint |

**The product line to hold:** report joint angles, ROM, spatiotemporal, asymmetry, GDI/GPS, and (with GRF)
joint moments. Report muscle **loading patterns** only as labeled model estimates. **Never** report muscle
strength, capacity, or "structure" from gait — recommend the confirming test instead.

---

## How the code uses this (`analysis/signatures.py`)

The rule engine consumes `kinematics.summarize()` (per-coordinate ROM + L/R symmetry) plus gait speed, and
emits `Finding`s with: the flagged coordinate + observed value, multiple plausible interpretations (never
one diagnosis), a confidence level, and per-rule + global caveats. Current sagittal-threshold rules:
stiff-knee swing (<45°), crouch stance knee (>30°), reduced hip extension (min >0°), foot drop / equinus
(swing DF <5°), reduced knee excursion (<50°), and L/R ROM asymmetry (outside 0.90–1.10). Phase-gating
(true swing/stance windows rather than global min/max) and the frontal/concussion channels are the next
additions; transverse-plane rules are intentionally withheld until accurate-mode multi-camera capture.

---

## Source index (primary, by section)

- **Tightness:** Arnold/Delp 2006 hamstring length (PMID 15964759); Jonkers 2006 RF (16399519); Whitehead 2007
  (16214274); equinus (MDPI Children 2022; 23465759); hip-flexor contracture validity (JNER 1743-0003-8-4);
  Kerrigan 2003 (12589613); Ferber 2010 ITBS (20118523); Willett 2016 Ober (AJSM 0363546515621762).
- **Weakness:** van der Krogt 2012 (S0966636212000392); Ong 2019 (pcbi.1006993); StatPearls Trendelenburg
  (NBK541094)/steppage (NBK547672); Heintz 2007 SO-vs-EMG (17071088).
- **Neuro:** Lewek 2010 (19945285); arm-swing meta (PMC10695845); SKG 44.3° (S0268003324001839); Goldberg
  2003 (12831736); Lance 1980 (PMC4229996); Patrick & Ada 2006 Tardieu (0269215506cr922oa).
- **Concussion:** Howell 2013/2018 (23643687, 29457997); multicenter null (29966907); Wilkerson 2021
  (PMC8384176); systematic review (32132473).
- **Pain:** knee OA (PMC8229136; JOSPT 2015); hip OA Duchenne (PMC3274426); antalgic (NBK559243);
  kinesiophobia (PMC11396003).
- **Injury:** Hewett 2005 (0363546504269591); Krosshaug 2016 (26867936); Bramah 2018 (30193080); Noehren
  2013 (23274607); Davis 2016 (26644428); Milner 2006 (16531902); Kyritsis 2016 (27215935); Grindem 2016
  (27162233); Moran 2017 FMS (28360142); **Bahr 2016 (27095747)**.
- **Modalities/limits:** OpenCap (Uhlrich 2023, pcbi.1011462); soft-tissue artifact/markerless plane
  reliability (Wade 2022 PMC8884063); Handsfield 2016 MRI volume (26565390); SWE (Brandenburg 2016, 27374483);
  McMulkin 2008 EUA (18841059).

*Sourcing caveat: many publisher/PMC full texts were 403-blocked during research; numbers were extracted
from abstracts/indexed snippets and corroborated across independent sources. Before any publication or
clinical/regulatory use, confirm load-bearing figures against the original PDFs (e.g., via Penn library
access).*
