# 11 — Clinical Evidence Base for Markerless Movement Analysis

**Scope.** This document grounds the clinical claims behind our two pipelines — (A) two-camera Pose2Sim + OpenSim inverse kinematics, and (B) monocular single-phone MediaPipe 3D → OpenSim — in the peer-reviewed literature. It covers (1) clinical use-cases by condition, (2) a metric-by-metric clinical reference with normal values and meaningful thresholds, (3) validity evidence with actual reported errors, (4) public normative datasets, and (5) evidence gaps and honest limitations.

**Citation discipline.** Every numeric claim below is tied to a named source (author + year + venue) listed in the References section. Where a number could not be independently verified from the retrieved material, it is flagged **(not verified)**. We have deliberately *not* invented DOIs, error magnitudes, or thresholds.

**One framing rule for the whole document.** Our pipelines are strong in the **sagittal plane** and weak in the **frontal/transverse planes**. Nearly every clinically actionable metric below is sorted by us into "trust sagittal," "treat frontal/transverse as screening only," or "do not report." This is the single most important honesty constraint in the system and it is repeated throughout.

---

## 1. Clinical Use-Cases by Condition

For each condition we give: the **movement signature**, **which of our metrics detects it**, the **clinical decision** it informs, and how it **tracks rehab outcomes**. A blunt **plane caveat** is attached wherever the signature lives in the frontal/transverse plane (where our markerless accuracy is weakest — see Section 3).

### 1.1 ACL injury, reconstruction & return-to-sport (RTS)

**Movement signature.** Non-contact ACL injuries predominantly occur during deceleration, landing, or cutting, and the classic at-risk pattern is **dynamic knee valgus** — a combination of knee abduction with hip adduction/internal rotation and limited knee flexion — under high abduction load (Hewett et al., 2005, *Am J Sports Med*). Roughly 70–84% of ACL injuries are non-contact, occurring while decelerating or changing direction (search-synthesized from the FPPA/landing literature; see Refs).

**Which of our metrics detects it.**
- **Sagittal (trust):** peak knee-flexion at landing and during the descent of a single-leg squat; **limb symmetry index (LSI)** of squat depth / single-leg mechanics; knee-flexion ROM asymmetry. These are sagittal and within our reliable band.
- **Frontal/transverse (screening only):** dynamic knee valgus / frontal-plane projection angle (FPPA), hip adduction, hip internal rotation. These live exactly where markerless is least trustworthy. The 2D FPPA itself is an *inconsistent* predictor of true 3D frontal-plane knee kinematics, with poor 2D–3D agreement (multiple 2D-FPPA validity studies, *Int J Sports Phys Ther*). So we treat any valgus number as a **flag to escalate to the 2-camera rig or a force-plate lab**, never as a quantitative endpoint.

**Clinical decision.** RTS readiness. The most commonly reported functional criterion is **LSI ≥ 90%** versus the contralateral limb on a hop-test battery (single hop, triple hop, crossover hop, 6-m timed hop) plus isokinetic quad strength (Return-to-Sport scoping reviews, *PubMed/PMC*, 2024–2025). Clinicians who incorporate hop testing may reduce re-injury risk substantially, but the 90% LSI threshold alone is known to **overestimate** knee function: many athletes who pass 90% LSI still show bilateral deficits versus healthy normative data, and RTS criteria fail to fully identify second-injury risk (Wellsandt et al., 2017, *JOSPT*, "Limb Symmetry Indexes Can Overestimate Knee Function"; Return-to-Sport reviews 2024–2025).

**How it tracks rehab.** Across rehab we track: (i) recovering peak knee-flexion symmetry during landing/squat (sagittal — trust), (ii) restoring squat depth LSI, and (iii) *trend* in valgus screening (frontal — caveat). The Hewett prospective work established that elevated knee-abduction moment during a drop-vertical-jump predicts ACL injury — but knee-abduction *moment* requires inverse dynamics (ground reaction force), which neither of our video-only pipelines measures. We therefore report kinematic proxies only and explicitly state we do **not** estimate knee-abduction moment without force data.

### 1.2 Patellofemoral pain (PFP)

**Movement signature.** PFP is associated with **excessive hip adduction and hip internal rotation** during running, single-leg squat, and landing — frequently with weak hip abductors/external rotators — increasing lateral patellar stress (Powers, 2010, *JOSPT*, "Influence of Abnormal Hip Mechanics on Knee Injury"; PFP running-biomechanics meta-analyses, *Gait Posture / ScienceDirect*). Women with PFP show significantly greater peak hip adduction and internal rotation than controls.

**Which of our metrics detects it.**
- **Frontal/transverse (screening only):** peak hip adduction, hip internal rotation, dynamic valgus during single-leg squat or step-down. Again, these are precisely the planes where our accuracy degrades — usable for *trend/screening*, not for a defensible degree-level diagnosis.
- **Sagittal (trust):** trunk lean, knee-flexion excursion, cadence (running cadence is a sagittal/temporal variable and is robust).

**Clinical decision.** Whether to prescribe **proximal (hip) strengthening** vs **gait retraining**. Evidence: both running retraining and proximal strengthening improve pain and function, but **only running retraining significantly reduces peak hip adduction** (PFP systematic review/meta-analysis, *Gait Posture*). Increasing running cadence (a sagittal/temporal metric we *can* measure well) is an established retraining lever.

**How it tracks rehab.** Cadence change and (with caveats) reduction in peak hip-adduction trend over a retraining block.

### 1.3 Knee osteoarthritis (OA) and TKA rehab

**Movement signature.** Medial knee OA is biomechanically driven by the **external knee adduction moment (KAM)** — elevated first-peak KAM is associated with pain and radiographic progression, and baseline KAM helps predict progression to total knee arthroplasty (TKA) (Miyazaki et al. and the OA biomechanics literature; a multivariate baseline-gait model classified progression-to-TKA at ~74% correct, *PubMed* 25708360). Gait-modification strategies (e.g., medial-thrust gait) reduce KAM by ~21% (first peak) to ~36% (second peak) (medial-thrust KAM study, *Sci Rep*, 2025).

**Which of our metrics detects it.**
- **The KAM itself is a moment — it needs ground reaction force (inverse dynamics) and is NOT recoverable from video kinematics alone.** Neither pipeline reports KAM. This is a hard limitation we state plainly.
- **What we CAN track (sagittal — trust):** gait speed, cadence, stance/swing %, **knee-flexion excursion / quadriceps-avoidance pattern** (reduced stance-phase knee flexion is a hallmark of painful knee OA), and **gait-speed recovery after TKA**. Gait speed and stride length typically return toward or above preoperative levels by ~12 months post-op for related joint replacements.

**Clinical decision.** Conservative-management response, surgical timing/triage signal, and post-TKA rehab progression.

**How it tracks rehab.** Gait speed (sixth vital sign — Section 1.7), stance-phase knee-flexion excursion, and symmetry indices across the rehab arc.

### 1.4 Hip OA and THA rehab

**Movement signature.** Hip OA reduces **peak hip extension** in terminal stance and shortens stride; **hip extension/flexion ROM** is the single variable that best separates healthy controls from OA and from post-THA patients (aberrant pelvis/hip kinematics study, *Gait Posture*, Foucher et al. lineage). A **Trendelenburg gait** (contralateral pelvic drop / trunk lurch from hip-abductor weakness) can persist after THA; reduced eccentric hip-abductor and concentric hip-extensor contraction are determinants (Trendelenburg-after-THA study, *J Orthop*, 2024). Gait speed and stride length typically recover by ~12 months post-THA, though the pattern still differs from healthy controls at 1–2 years.

**Which of our metrics detects it.**
- **Sagittal (trust):** peak hip extension, stride length, gait speed, stance/swing symmetry.
- **Frontal (screening only):** pelvic drop / Trendelenburg sign — frontal-plane, so screening-grade.

**Clinical decision.** Surgical candidacy signal (loss of hip extension), and targeting **hip-abductor/extensor strengthening** post-THA when a Trendelenburg pattern persists.

**How it tracks rehab.** Restoration of hip-extension ROM and stride length toward normative bands; resolution of the (screening-grade) pelvic-drop trend.

### 1.5 Post-stroke hemiparetic gait

**Movement signature.** Decreased cadence and speed; **step-length and stance-time asymmetry** (prolonged paretic swing, prolonged non-paretic stance); **stiff-knee gait** (reduced paretic knee flexion in swing) with compensations such as **circumduction, hip-hiking, and vaulting**; foot-drop and reduced push-off (post-stroke gait scoping reviews, *Front Neurol* 2021; stiff-knee classification, *J Neuroeng Rehabil* 2025). Increased spatiotemporal asymmetry tracks with slower walking.

**Which of our metrics detects it.**
- **Sagittal/temporal (trust):** gait speed, cadence, **step-length asymmetry**, stance/swing %, paretic-knee-flexion-in-swing, double-support time.
- **Frontal/transverse (screening only):** circumduction and hip-hiking (out-of-sagittal compensations) — visually flaggable, but degree-level numbers are screening-grade.

**Clinical decision.** Ambulation-category classification — **household (<0.4 m/s), limited community (0.4–0.8 m/s), community (>0.8 m/s)** ambulator bands (Perry's classic functional gait-speed categories) — and targeting (e.g., AFO for foot-drop, stiff-knee interventions).

**How it tracks rehab.** **Gait-speed MCID ≈ 0.16 m/s in subacute stroke** (Tilson et al., 2010, *Phys Ther*, anchored to modified Rankin Scale in the LEAPS trial; broader pathology consensus 0.10–0.20 m/s, Bohannon & Glenney, 2014, *J Eval Clin Pract*). Symmetry indices improving toward 1.0 is a primary rehab signal.

### 1.6 Cerebral palsy (CP) gait and the GDI

**Movement signature.** Sagittal-plane CP gait patterns (Rodda & Graham classification): **true equinus, jump gait, apparent equinus, crouch gait** (Rodda & Graham, 2001/2004 lineage; CP gait reviews, *EFORT Open Rev* 2016, *PMC* 5489760). Severity scales with **GMFCS** level.

**Which of our metrics detects it.**
- **Sagittal (trust):** ankle equinus, knee-flexion-at-initial-contact and in stance (crouch), hip-flexion pattern — these define the Rodda–Graham buckets and are sagittal.
- **The Gait Deviation Index (GDI)** (Schwartz & Rozumalski, 2008, *Gait Posture*): a single dimensionless score where **100 = typically developing mean, and every 10 points below 100 = 1 SD** of deviation, derived from >6,000 CP strides. GDI is *primarily sagittal-weighted* in practice and is computable from our kinematic waveforms — a key reason it is a good summary metric for our system.

**Clinical decision.** Surgical planning (e.g., single-event multilevel surgery, SEMLS) and orthotic prescription.

**How it tracks rehab/surgery.** GDI change. Reported **MCID for GDI is in the ~5–10 point range** depending on anchor/method — values cited include ≈7.9 points (≈10% improvement) and a ~10-point "clinically significant" threshold (Massaad et al. lineage / GPS-GDI MCID work). We treat **≥10 points** as a conservative, defensible "meaningful improvement" bar. *(Exact MCID is method-dependent — we cite the range rather than a single canonical number.)* Note: a markerless-vs-marker GDI comparison reported between-system GDI differences of ~6.9 points, which is itself near the MCID — a caution for markerless GDI reporting (*Gait Posture* 2024 markerless-GDI study).

### 1.7 Falls risk & aging (sit-to-stand, gait speed)

**Movement signature.** Slowed gait, reduced step length, prolonged double-support, and slowed/strategy-altered sit-to-stand (greater trunk flexion, "dominant vertical rise" compensations in weaker adults).

**Which of our metrics detects it.**
- **Gait speed** is the **"sixth vital sign"** (Studenski et al., 2011, *JAMA*, "Gait Speed and Survival in Older Adults," pooled analysis of 9 cohorts). **<0.8 m/s** is a widely used cutoff for elevated risk of disability, hospitalization, institutionalization, and mortality.
- **Five-times sit-to-stand (5×STS):** **>15 s** is associated with recurrent falls (Buatois et al., 2008 — those >15 s had ~74% greater recurrent-fall risk). Normative means: ~11.4 s (60–69 y), ~12.6 s (70–79 y), ~14.8 s (80–89 y) (Bohannon meta-analytic reference values, *Percept Mot Skills* / RehabMeasures synthesis).
- **Timed Up and Go (TUG):** **≥13.5 s** flags high fall risk; meta-analysis shows it is better at *ruling in* than *ruling out* — pooled specificity ~0.74, sensitivity ~0.31 (Barry et al., 2014, *BMC Geriatr*). Strong in Parkinson's (test–retest ICC ~0.80).

**Clinical decision.** Falls-prevention referral; functional-decline surveillance.

**How it tracks rehab.** Gait-speed and 5×STS-time trajectories, both squarely measurable by our temporal pipelines.

### 1.8 Parkinson's disease (PD)

**Movement signature.** Reduced step length and speed, reduced foot clearance (**shuffling**), **festination**, **freezing of gait**, asymmetric **reduced arm swing**, impaired turning, and **bradykinetic sit-to-stand** (PD gait reviews, *PMC* 7349580, *J Neurol* 2019). Increased step-to-step **temporal variability** with progression.

**Which of our metrics detects it.**
- **Sagittal/temporal (trust):** step length, gait speed, cadence, stride-time variability, turning time (from trajectory), sit-to-stand rise time, arm-swing amplitude (a sagittal swing we can capture).

**Clinical decision.** Medication-state (ON/OFF) monitoring, fall-risk stratification, response to cueing/DBS.

**How it tracks rehab/therapy.** Step length, gait-speed, and stride-time-variability trends; TUG and 5×STS as bundled functional anchors.

### 1.9 Lower-limb amputee / prosthetic gait

**Movement signature.** Marked **temporal-spatial asymmetry** — shortened prosthetic-side stance/support time, the sound limb over-loaded; the **greatest kinematic asymmetry appears in prosthetic-side knee angle during stance**, and is highly sensitive to **socket alignment** (transtibial alignment systematic review, *PLOS One* 2016, *PMC* 5140067; prosthetic-alignment case studies, *Clin Biomech*).

**Which of our metrics detects it.**
- **Sagittal/temporal (trust):** stance/swing %, support-time symmetry, step-length symmetry, prosthetic-vs-sound knee-angle profile.

**Clinical decision.** **Prosthetic alignment** tuning and socket-fit follow-up; the biggest alignment-driven changes occur in prosthetic stance.

**How it tracks rehab.** Symmetry indices toward 1.0; normalization of prosthetic-side stance knee kinematics after alignment changes.

---

## 2. Metric-by-Metric Clinical Reference

Reference values, meaningful thresholds, and MCID/MDC where published. **"Plane" tags** indicate where each metric sits relative to our accuracy envelope (Section 3).

### 2.1 Spatiotemporal gait metrics

| Metric | Normal / reference | Meaningful threshold | MCID / MDC | Plane / trust |
|---|---|---|---|---|
| **Gait speed (comfortable)** | Healthy adult ≈1.2–1.4 m/s | **<0.8 m/s** elevated risk (Studenski 2011); **<0.4 / 0.4–0.8 / >0.8 m/s** = household / limited-community / community ambulator (Perry) | **MCID 0.10–0.20 m/s** (Bohannon & Glenney 2014); **0.16 m/s** subacute stroke (Tilson 2010) | Temporal — **trust** |
| **Cadence** | ≈100–120 steps/min adults | Reduced in PD, stroke, OA | — (not verified) | Temporal — **trust** |
| **Step length** | ≈0.72 m (≈28 in) avg adult | Asymmetry flags hemiparesis/amputee | — (not verified) | Sagittal — **trust** |
| **Stride length** | ≈1.44 m (≈56 in) avg adult | Shortened in PD/OA/hip OA | — (not verified) | Sagittal — **trust** |
| **Stance %** | **≈60%** of gait cycle | Prolonged non-paretic stance post-stroke | — | Temporal — **trust** |
| **Swing %** | **≈40%** of gait cycle | Prolonged paretic swing post-stroke | — | Temporal — **trust** |
| **Double-support time** | **≈20%** of cycle (two ~10% periods) | Increases with falls risk, PD, instability | — | Temporal — **trust** |
| **Step width** | **≈5–10 cm (2–4 in)** | Widened with instability/ataxia | — | Frontal — **screening** |
| **Symmetry indices** | 1.0 / 100% = perfect | Asymmetry tracks stroke, amputee, ACL | population-specific | depends on input plane |

*Sources for the row values: gait-cycle reference syntheses (Perry, 1992; Boston O&P / Physiopedia gait-cycle references), Studenski 2011 (JAMA), Tilson 2010 (Phys Ther), Bohannon & Glenney 2014 (J Eval Clin Pract). Where an MCID/MDC was not retrieved we mark "(not verified)" rather than inventing one.*

### 2.2 Sagittal kinematic waveforms (hip / knee / ankle)

These are the **core trustworthy outputs** of both pipelines.

- **Hip (sagittal):** flexes ~30° at initial contact, extends to ~10° hip extension in terminal stance; **loss of terminal-stance hip extension** is the most discriminating hip-OA feature (hip-OA kinematics literature). *(Exact normative degree band: see Winter / Fukuchi datasets, Section 4.)*
- **Knee (sagittal):** ~5° at initial contact, ~15–20° loading-response flexion, near-extension in mid/terminal stance, **~60° peak swing flexion**. **Stiff-knee gait** = reduced swing-phase peak. **Quadriceps-avoidance** = reduced loading-response flexion in knee OA. *(Use Fukuchi 2018 / Bovi 2011 for full normative waveform bands.)*
- **Ankle (sagittal):** small plantarflexion at contact, controlled dorsiflexion through stance, push-off plantarflexion. **Equinus** (CP, stroke foot-drop) = excess plantarflexion; reduced dorsiflexion limits squat depth.

> **Reporting rule:** report sagittal hip/knee/ankle waveforms against a *cited normative band* (Section 4), not a generic shaded region.

### 2.3 Squat metrics

| Metric | Normal / reference | Clinical meaning | Plane / trust |
|---|---|---|---|
| **Squat depth (peak knee flexion)** | Deep squat ≈120–140° knee flexion **(not verified — varies by definition)** | Depth limited by ankle DF, hip, knee ROM | Sagittal — **trust** |
| **Peak hip flexion** | rises with depth | hip-mobility / pattern | Sagittal — **trust** |
| **Dynamic knee valgus / hip adduction** | minimal in controlled squat | ACL/PFP risk screening | Frontal/transverse — **screening only** |
| **Ankle dorsiflexion** | often the **main limiter** of deep-squat depth | mobility target | Sagittal — **trust** |
| **Posterior pelvic tilt ("butt wink")** | appears at end-range depth | coupled with lumbar flexion → lumbar shear/compression | Sagittal pelvis — **trust (kinematic)** |

*Sources: squat biomechanics review (Lorenzetti / IJSPT "A Biomechanical Review of the Squat"); butt-wink / pelvic-tilt analyses; deep-squat ROM correlation study (PMC 7276781). Note that ankle-DF ROM and knee-flexion ROM correlate positively with achievable squat depth.*

### 2.4 Sit-to-stand (STS) metrics

- **Phases (Schenkman model):** I flexion-momentum, II momentum-transfer (buttocks-off → max ankle DF), III extension ("**triple extension**" of hip/knee/ankle), IV stabilization.
- **5×STS time:** norms ~11.4 s (60–69 y), ~12.6 s (70–79 y), ~14.8 s (80–89 y); **>15 s** → recurrent-fall risk (Bohannon meta-analysis; Buatois 2008).
- **Rise time / strategy:** weaker adults shift to exaggerated trunk flexion or "dominant vertical rise" strategies; greater trunk angular velocity after seat-off with lower strength (STS strategy studies, *PMC* 5948784).
- **Triple extension** completeness (hip + knee + ankle reaching extension) is a sagittal pattern we can score.

> All STS kinematics of interest are **sagittal — trust**; the main risk is event-timing accuracy (seat-off detection) on monocular video.

### 2.5 Frontal/transverse "decision-support, not diagnosis" metrics

Dynamic knee valgus, FPPA, hip adduction, hip internal rotation, pelvic drop/Trendelenburg, circumduction, step width. **All screening-grade in our system.** Use them to *flag* and *escalate to the 2-camera rig or a gold-standard lab*, never to issue a degree-precise diagnosis. The 2D-FPPA–vs–3D agreement is poor even in dedicated 2D studies, independent of our pipeline error.

---

## 3. Validity Evidence (Concurrent Validity vs Marker-Based Gold Standard)

This is the section that defines what we may and may not claim. **Headline rule: sagittal lower-limb kinematics are clinically usable (~3–5° error); frontal/transverse and many squat/landing measures are not.**

### 3.1 OpenCap (two-smartphone → OpenSim) — Uhlrich, Falisse, et al., 2023, *PLoS Comput Biol*

OpenCap is the closest published analog to our Pipeline A (multi-phone → OpenSim).

- **Subjects/method:** validation averaged over **n = 10** participants, multiple activities, vs marker-based motion capture + force plates.
- **Kinematics:** **mean absolute error (MAE) = 4.5°** for joint angles, averaged over activities and DOFs (6 pelvis pose DOFs [kinematics only], 3 lumbar, 3 per hip, 1 per knee, 2 per ankle).
- **Kinetics:** **ground reaction force MAE = 6.2% bodyweight**; **joint moment MAE = 1.2% bodyweight·height** (these require their force-estimation/simulation step — *not* something our video-only kinematic pipeline reproduces).
- **Interpretation:** the ~4.5° overall figure is dominated by well-tracked sagittal DOFs; frontal/transverse DOFs (e.g., hip rotation/adduction) carry larger error. We treat **4.5° as a best-case, well-lit, two-camera number**, not a guarantee.

*Source: Uhlrich SD, Falisse A, Kidziński Ł, et al. "OpenCap: Human movement dynamics from smartphone videos." PLoS Comput Biol 19(10):e1011462, 2023. (Numbers from the paper's validation summary as reported by the authors' Utah MoBL copy and PLOS abstract.)*

### 3.2 OpenCap on functional / return-to-sport tasks (independent validations)

Independent groups have replicated OpenCap *less optimistically* on dynamic tasks — directly relevant to our ACL/squat use-cases:

- **Walking (Horsak et al., 2023–2024):** RMSE range **3.7–10.2°** across lower-extremity kinematics, with greater across-trial variation than marker-based.
- **RTS tasks (Turner, Chaaban, Padua, 2024, *J Biomech*):** agreement **best in sagittal plane** — knee/hip **r > 0.94**, ankle **r = 0.84–0.93** — in athletes 12–18 months post-ACLR doing jump-landing, single-leg hop, lateral-vertical hop.
- **Sagittal RTS RMSE (validity study):** **RMSE ≈ 11.6–14.7°** across tasks in the sagittal plane, with the hip in **single-leg squat ~20% worse**; frontal/transverse hip RMSE reported as relatively low (**< 7.3°**) in that particular study — illustrating that error magnitudes are **task- and plane-dependent and not uniform across studies**.

**Honest read:** even the *two-camera* OpenCap-class system shows **double-digit-degree RMSE on dynamic single-leg and landing tasks**. Our Pipeline A should claim clinical-grade accuracy only for **steady sagittal gait**, and "screening-grade" for dynamic athletic tasks.

### 3.3 Squat validity specifically — *do not trust markerless squat knee angles quantitatively*

- **Pilot validity, back squat (CISS pilot study):** OpenCap **underestimated knee angles** with an average discrepancy of **16.9° (SD 18.3°)**, **RMSE = 24.9°**, **ICC(3,1) = 0.503** (moderate) vs a 10-camera Vicon system. Agreement was acceptable only over ~10–35% of the squat cycle.

This is a **red flag** for any squat-depth claim from video. We will report squat **depth trends and symmetry**, but explicitly disclaim degree-precise peak-knee-flexion in deep squat from monocular (and even two-camera) video.

### 3.4 Pose2Sim (multi-camera → OpenSim) — Pagnon, Domalain, Reveret

This is the published method underpinning our Pipeline A.

- **Accuracy (Part 2; Pagnon et al., 2022, *Sensors* 22(7):2712):** over walking/running/cycling, **mean joint-angle error = 3.0° (walking), 4.1° (running), 4.0° (cycling)**; **ROM error 2.7°/2.3°/4.3°**. Waveform similarity **CMC mostly >0.85 (good) to >0.95 (excellent)**, **>0.9 in the sagittal plane**, *except* a **systematic ~15° hip offset in running (CMC 0.65)** and the **ankle in cycling (CMC 0.75) due to partial occlusion**.
- **Robustness (Part 1; Pagnon et al., 2021, *Sensors* 21(19):6530):** under degraded image quality, **4 vs 8 cameras**, and 1-cm calibration error, stride-to-stride SD stayed **1.7–3.2°** and MAE vs reference condition **0.35–1.6°** — i.e., graceful degradation, but built/tested with **calibrated multi-camera** capture.

**Read:** Pose2Sim's ~3–4° sagittal accuracy is the basis for our "trust sagittal gait" claim — but it assumes **good multi-camera calibration**, and even here non-sagittal DOFs and occluded segments degrade (the 15° running-hip offset is a concrete warning).

### 3.5 Monocular / MediaPipe (single phone) — Pipeline B

Single-camera 3D is fundamentally **ill-posed** (2D→3D depth ambiguity, self-occlusion, left/right swaps, viewpoint sensitivity) — surveys of monocular 3D HPE are explicit that different 3D poses project to near-identical 2D images.

- **MediaPipe/BlazePose knee angle (single camera, JMIR preprint 2024–2025):** dynamic concurrent validity **MAE = 7.07°** with **systematic underestimation −5.12°**; **static accuracy is angle-dependent (RMSE up to 15.49°)**. Accuracy depends strongly on camera viewpoint.
- **Optimized monocular pipelines:** post-processing/filtering improves BlazePose joint-angle prediction by ~10.7% and reduces mean angular error ~16.6% (BlazePose gait-optimization studies, *MDPI Sensors*).
- **Multi-pose/depth-fusion markerless (single 2D camera, gait):** can reach **LCC > 0.96** and **inter-session RMSE < 3°** for hip/knee in *controlled* sagittal gait (multi-pose markerless gait validation, *PMC* 11597901) — i.e., monocular *can* be good for **plain sagittal gait at fixed viewpoint**, but not for frontal/transverse or dynamic tasks.

**Honest read for Pipeline B:** acceptable for **sagittal gait kinematics at a controlled side-on viewpoint** (single-digit-degree error achievable with filtering); **not trustworthy** for frontal/transverse angles, deep squat depth, valgus, or any out-of-plane measure. Camera placement (true side-on) is a hard requirement, not a nicety.

### 3.6 Where markerless is NOT trustworthy — explicit list

1. **Frontal- and transverse-plane joint angles** (hip rotation/adduction, knee valgus, FPPA) — screening only, in *both* pipelines.
2. **Deep-squat peak knee flexion** — OpenCap underestimated by ~17° (RMSE ~25°) vs Vicon.
3. **Dynamic single-leg / landing tasks** — double-digit RMSE even with two cameras.
4. **Any joint moment / KAM / knee-abduction moment** — needs force data; **not produced** by video-only kinematics.
5. **Monocular anything out-of-plane**, plus monocular accuracy collapses off true side-on viewpoint.
6. **Absolute scaling / segment lengths** in monocular without calibration — depth ambiguity.

---

## 4. Normative Datasets (to Replace a Generic Reference Band)

Use these public, citable datasets as the reference band for sagittal waveforms and spatiotemporals — not an unsourced shaded region.

### 4.1 Fukuchi, Fukuchi & Duarte, 2018, *PeerJ* (PRIMARY recommended)
- **Citation:** Fukuchi CA, Fukuchi RK, Duarte M. "A public dataset of overground and treadmill walking kinematics and kinetics in healthy individuals." *PeerJ* 6:e4640, 2018. DOI **10.7717/peerj.4640**. Data on Figshare DOI **10.6084/m9.figshare.5722711**.
- **Contents:** **42 healthy subjects (24 young + 18 older adults)**, overground **and** treadmill, **range of gait speeds**, barefoot; 3D motion capture + force plates + instrumented treadmill. Pelvis + lower-extremity kinematics & kinetics, raw and processed (c3d + txt), plus a Visual3D pipeline.
- **Why we use it:** speed-stratified, includes older adults, open, and directly maps to our sagittal outputs.

### 4.2 Bovi, Rabuffetti, Mazzoleni & Ferrarin, 2011, *Gait Posture*
- **Citation:** Bovi G, Rabuffetti M, Mazzoleni P, Ferrarin M. "A multiple-task gait analysis approach: kinematic, kinetic and EMG reference data for healthy young and adult subjects." *Gait Posture* 33(1):6–13, 2011.
- **Contents:** **50 healthy subjects, ages 6–72**; level walking at multiple speeds, **toe-walking, heel-walking, stair ascent/descent**; 3D marker coordinates, lower-limb joint angles (sagittal/coronal/transverse), GRF/torque, COP, joint moments/power, COM, surface EMG.
- **Why we use it:** multi-task and multi-speed; supports stair and toe/heel reference bands beyond plain gait.

### 4.3 Winter, *Biomechanics and Motor Control of Human Movement* (foundational)
- **Citation:** Winter DA. *Biomechanics and Motor Control of Human Movement* (4th ed., 2009; 5th ed. "Winter's…" eds. Thomas & Zeni). Also Winter DA, *The Biomechanics and Motor Control of Human Gait: Normal, Elderly and Pathological* (2nd ed., 1991).
- **Contents:** canonical normative joint-angle/velocity/moment/power profiles for normal, elderly, and pathological gait; the de-facto textbook reference band.
- **Why we use it:** the classic citable source for "normal" sagittal waveform shape and magnitude.

> **Implementation note:** replace any hard-coded shaded band in the UI with mean ± SD curves time-normalized from Fukuchi 2018 (primary) / Bovi 2011 / Winter, with an explicit on-screen citation and the speed bin used.

---

## 5. Evidence Gaps & Honest Limitations

1. **Markerless clinical validation is thin per-condition.** Most validity studies are in **healthy young/athletic cohorts** doing controlled tasks. Validity in **stroke, CP, PD, amputee, and frail older** populations — where occlusion, assistive devices, atypical postures, and slow speeds are common — is **under-studied**; do not assume the ~4.5° (OpenCap) / ~3–4° (Pose2Sim) numbers transfer to these populations.

2. **Squat & STS markerless validity is weak/sparse.**
   - Squat: the best available data (CISS pilot) shows **~17° underestimation, ~25° RMSE** for back-squat knee angle even with two cameras — so **deep-squat depth is screening-grade at best**.
   - Sit-to-stand: we found **little dedicated concurrent-validity data for markerless STS kinematics**; the main risk is **seat-off / event-timing** accuracy on video. Treat STS kinematic angles as **unvalidated for our pipelines** and lean on STS *timing* (5×STS time), which is robust. **(STS markerless kinematic validity: not verified.)**

3. **Single-camera depth limits are fundamental, not tunable.** Monocular 3D is mathematically ill-posed; filtering helps (~10–17% improvement) but cannot remove depth ambiguity, self-occlusion, or viewpoint sensitivity. **Frontal/transverse angles and absolute depth/scale are not recoverable reliably from one camera.** A true **side-on viewpoint** is mandatory for Pipeline B sagittal accuracy.

4. **No kinetics without force data.** KAM, knee-abduction moment, GRF, and joint moments — central to OA progression and ACL-risk prediction — **cannot be measured by video kinematics alone**. OpenCap obtains them via an added simulation step; our video-only kinematic pipelines do not, and we must not imply otherwise.

5. **Frontal-plane valgus screening is doubly limited.** Even *dedicated* 2D-FPPA methods agree poorly with 3D frontal-plane knee angle; layering our markerless frontal-plane error on top makes valgus a **flag-and-escalate** signal, never a diagnosis.

6. **MCID/MDC are population- and method-specific.** Gait-speed MCID (0.10–0.20 m/s) and GDI MCID (~5–10 pts) vary by anchor and cohort; several metric-level MDCs we sought were **not verified** in the retrieved sources and are marked as such above. Do not present a single universal MCID without its population.

7. **Between-system bias can approach the MCID.** A markerless-vs-marker GDI study found ~6.9-point between-system differences — close to the GDI MCID — meaning **system-to-system bias can masquerade as clinical change**. Track each patient on the *same* pipeline/setup and avoid cross-system comparisons.

**Bottom line for the clinic.** Lead with what is defensible: **sagittal gait kinematics, spatiotemporal metrics, gait speed, cadence, symmetry indices, 5×STS/TUG timing, and GDI as a sagittal summary** — reported against cited normative bands (Fukuchi/Bovi/Winter) with population-appropriate thresholds. Demote **frontal/transverse angles, valgus, deep-squat depth, and all joint moments** to screening-or-escalate. Prefer the **two-camera Pose2Sim/OpenCap-class** rig for anything beyond plain sagittal gait, and keep the **monocular phone** pipeline to **side-on sagittal gait and timing** tasks.

---

## References

1. Uhlrich SD, Falisse A, Kidziński Ł, Muccini J, Ko M, Chaudhari AS, Hicks JL, Delp SL. **OpenCap: Human movement dynamics from smartphone videos.** *PLoS Computational Biology* 19(10):e1011462, 2023. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462 (PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10586693/ ; author copy: https://mobl.mech.utah.edu/wp-content/uploads/2023/12/opencap.pdf )
2. Pagnon D, Domalain M, Reveret L. **Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 2: Accuracy.** *Sensors* 22(7):2712, 2022. https://www.mdpi.com/1424-8220/22/7/2712 (PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9002957/ )
3. Pagnon D, Domalain M, Reveret L. **Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 1: Robustness.** *Sensors* 21(19):6530, 2021. https://www.mdpi.com/1424-8220/21/19/6530 (PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/ )
4. Pose2Sim software: Pagnon D, et al. *perfanalytics/pose2sim* (Python package). https://github.com/perfanalytics/pose2sim
5. Turner J, Chaaban C, Padua D. **Validation of OpenCap: A low-cost markerless motion capture system for lower-extremity kinematics during return-to-sport tasks.** *Journal of Biomechanics*, 2024. https://www.sciencedirect.com/science/article/abs/pii/S0021929024002781 (SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4731641 )
6. Horsak B, et al. **Validation of OpenCap on lower extremity kinematics during functional tasks** (and walking RMSE 3.7–10.2°). *Journal of Biomechanics*, 2025. https://www.sciencedirect.com/science/article/abs/pii/S0021929025001137
7. **Pilot study: Insights into the validity of OpenCap to assess knee kinematics during the back squat.** *Current Issues in Sport Science (CISS)*. https://ciss-journal.org/article/view/10903
8. **Validity and reliability of trunk and lower-limb kinematics during squatting, hopping, jumping and side-stepping using OpenCap.** *Journal of Sports Sciences*, 2024. https://www.tandfonline.com/doi/full/10.1080/02640414.2024.2415233
9. **Assessing lower-limb kinematics via OpenCap during dynamic tasks relevant to ACL injury: a validity study.** *Journal of Science and Medicine in Sport*, 2023. https://www.jsams.org/article/S1440-2440(23)00308-0/fulltext
10. Fukuchi CA, Fukuchi RK, Duarte M. **A public dataset of overground and treadmill walking kinematics and kinetics in healthy individuals.** *PeerJ* 6:e4640, 2018. DOI 10.7717/peerj.4640. https://peerj.com/articles/4640/ ; data: https://figshare.com/articles/dataset/5722711 (DOI 10.6084/m9.figshare.5722711)
11. Bovi G, Rabuffetti M, Mazzoleni P, Ferrarin M. **A multiple-task gait analysis approach: kinematic, kinetic and EMG reference data for healthy young and adult subjects.** *Gait & Posture* 33(1):6–13, 2011. https://www.sciencedirect.com/science/article/abs/pii/S0966636210002468 (PubMed: https://pubmed.ncbi.nlm.nih.gov/21123071/ )
12. Winter DA. **Biomechanics and Motor Control of Human Movement**, 4th ed., Wiley, 2009 (5th ed., Thomas SJ, Zeni JA, eds., 2022). Also *The Biomechanics and Motor Control of Human Gait: Normal, Elderly and Pathological*, 2nd ed., 1991. https://books.google.com/books/about/Biomechanics_and_Motor_Control_of_Human.html?id=_bFHL08IWfwC
13. Studenski S, Perera S, Patel K, et al. **Gait speed and survival in older adults.** *JAMA* 305(1):50–58, 2011. https://www.semanticscholar.org/paper/Gait-speed-and-survival-in-older-adults.-Studenski-Perera/25de2f67035336224c9f1e37f58c5b531d34b9ed
14. Tilson JK, Sullivan KJ, Cen SY, et al. **Meaningful gait speed improvement during the first 60 days poststroke: minimal clinically important difference (LEAPS).** *Physical Therapy* 90(2):196–208, 2010. https://pmc.ncbi.nlm.nih.gov/articles/PMC2816032/ (PubMed: https://pubmed.ncbi.nlm.nih.gov/20022995/ )
15. Bohannon RW, Glenney SS. **Minimal clinically important difference for change in comfortable gait speed of adults with pathology: a systematic review.** *Journal of Evaluation in Clinical Practice* 20(4):295–300, 2014. https://onlinelibrary.wiley.com/doi/abs/10.1111/jep.12158
16. Hewett TE, Myer GD, Ford KR, et al. **Biomechanical Measures of Neuromuscular Control and Valgus Loading of the Knee Predict Anterior Cruciate Ligament Injury Risk in Female Athletes: A Prospective Study.** *American Journal of Sports Medicine* 33(4):492–501, 2005. https://journals.sagepub.com/doi/10.1177/0363546504269591
17. Wellsandt E, Failla MJ, Snyder-Mackler L. **Limb Symmetry Indexes Can Overestimate Knee Function After Anterior Cruciate Ligament Injury.** *JOSPT* 47(5):334–338, 2017. https://www.jospt.org/doi/10.2519/jospt.2017.7285
18. **Return to Sport Following ACL Reconstruction: A Scoping Review of Criteria Determining RTS Readiness.** 2024. https://pubmed.ncbi.nlm.nih.gov/39565551/ ; related: https://pmc.ncbi.nlm.nih.gov/articles/PMC11887908/
19. **Does the 2D Frontal Plane Projection Angle Predict Frontal Plane Knee Moments…?** *International Journal of Sports Physical Therapy*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9718689/
20. Powers CM. **The Influence of Abnormal Hip Mechanics on Knee Injury: A Biomechanical Perspective.** *JOSPT* 40(2):42–51, 2010. https://www.jospt.org/doi/10.2519/jospt.2010.3337
21. **Dynamic knee valgus kinematics and their relationship to pain in women with patellofemoral pain.** *PMC.* https://pmc.ncbi.nlm.nih.gov/articles/PMC6738932/
22. **Runners with patellofemoral pain have altered biomechanics which targeted interventions can modify: systematic review and meta-analysis.** *Gait & Posture / ScienceDirect.* https://www.sciencedirect.com/science/article/abs/pii/S0966636215009674
23. **Three-dimensional biomechanical gait characteristics at baseline are associated with progression to total knee arthroplasty.** *PubMed* 25708360. https://pubmed.ncbi.nlm.nih.gov/25708360/
24. **External Knee Adduction and Flexion Moments during Gait and Medial Tibiofemoral Disease Progression in Knee Osteoarthritis.** *PMC.* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4470726/
25. **Biomechanical mechanisms behind the reduction of knee adduction moment in medial knee thrust gait.** *Scientific Reports*, 2025. https://www.nature.com/articles/s41598-025-21220-1
26. **Trendelenburg gait after total hip arthroplasty due to reduced muscle contraction of the hip abductors and extensors.** *Journal of Orthopaedics*, 2024. https://pubmed.ncbi.nlm.nih.gov/39351271/
27. **Aberrant pelvis and hip kinematics impair hip loading before and after total hip replacement.** *Gait & Posture.* https://www.sciencedirect.com/science/article/abs/pii/S0966636209001520
28. **Assessment Methods of Post-stroke Gait: A Scoping Review of Technology-Driven Approaches.** *Frontiers in Neurology* 12:650024, 2021. https://www.frontiersin.org/articles/10.3389/fneur.2021.650024/full (PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8217618/ )
29. **Post-stroke Stiff-Knee gait: are there different types or different severity levels?** *Journal of NeuroEngineering and Rehabilitation*, 2025. https://link.springer.com/article/10.1186/s12984-025-01582-3
30. Schwartz MH, Rozumalski A. **The Gait Deviation Index: a new comprehensive index of gait pathology.** *Gait & Posture* 28(3):351–357, 2008. https://pubmed.ncbi.nlm.nih.gov/18565753/
31. **Comparison of Gait Deviation Index (GDI) and Gait Variability Index (GVI) measured by marker-based and markerless motion capture systems in children with cerebral palsy.** *Gait & Posture*, 2024. https://www.sciencedirect.com/science/article/abs/pii/S0966636224006489
32. **Outcome tools used for ambulatory children with cerebral palsy: responsiveness and minimum clinically important differences.** *PMC.* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2990955/
33. **Gait analysis in children with cerebral palsy** (Rodda–Graham sagittal patterns). *EFORT Open Reviews* 1(12), 2016 / *PMC* 5489760. https://eor.bioscientifica.com/view/journals/eor/1/12/2058-5241.1.000052.xml ; https://pmc.ncbi.nlm.nih.gov/articles/PMC5489760/
34. **Gait Analysis in Parkinson's Disease: An Overview of the Most Accurate Markers for Diagnosis and Symptoms Monitoring.** *PMC* 7349580. https://pmc.ncbi.nlm.nih.gov/articles/PMC7349580/
35. **Gait and postural disorders in parkinsonism: a clinical approach.** *Journal of Neurology*, 2019 / *PMC* 7578144. https://link.springer.com/article/10.1007/s00415-019-09382-1
36. Barry E, Galvin R, Keogh C, Horgan F, Fahey T. **Is the Timed Up and Go test a useful predictor of risk of falls in community dwelling older adults: a systematic review and meta-analysis.** *BMC Geriatrics* 14:14, 2014. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3924230/ (PubMed: https://pubmed.ncbi.nlm.nih.gov/24484314/ )
37. **Five Times Sit to Stand Test** — norms and fall-risk cutoffs (Bohannon reference values; Buatois 2008 >15 s recurrent falls). RehabMeasures (SRALab): https://www.sralab.org/rehabilitation-measures/five-times-sit-stand-test ; reference-value meta-analysis: https://www.researchgate.net/publication/6757105
38. **Predictive Cutoff Values of the Five-Times Sit-to-Stand Test and the Timed Up & Go Test for Disability Incidence in Older People.** *Physical Therapy* 97(4):417–424, 2017. https://academic.oup.com/ptj/article/97/4/417/3078574
39. Schenkman M, et al. (sit-to-stand phase model) and **Sit-to-stand strategy / kinematics in older adults** studies: https://pmc.ncbi.nlm.nih.gov/articles/PMC12404020/ ; strength/trunk-use: https://pmc.ncbi.nlm.nih.gov/articles/PMC5948784/
40. **A Biomechanical Review of the Squat Exercise: Implications for Clinical Practice.** *International Journal of Sports Physical Therapy.* https://ijspt.scholasticahq.com/article/94600
41. **The relationship between the deep squat movement and hip, knee and ankle range of motion and muscle strength.** *PMC* 7276781. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7276781/
42. Perry J. **Gait Analysis: Normal and Pathological Function.** SLACK, 1992 (gait-cycle phase percentages, ambulator categories). Reference syntheses: https://www.bostonoandp.com/Customer-Content/www/CMS/files/GaitTerminology.pdf ; https://www.physio-pedia.com/The_Gait_Cycle
43. **Monocular 3D Human Pose Markerless Systems for Gait Assessment.** *PMC* 10295566. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10295566/
44. **A Survey of the State of the Art in Monocular 3D Human Pose Estimation: Methods, Benchmarks, and Challenges.** *PMC* 12031093. https://pmc.ncbi.nlm.nih.gov/articles/PMC12031093/
45. **Validation of Single-Camera MediaPipe BlazePose for Knee Joint Angle Measurement: Concurrent Validity, Inter-Rater and Test-Retest Reliability.** *JMIR Preprints* #102399. https://preprints.jmir.org/preprint/102399
46. **Validation of a 3D Markerless Motion Capture Tool Using Multiple Pose and Depth Estimations for Quantitative Gait Analysis.** *PMC* 11597901. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11597901/
47. **Accuracy Evaluation of 3D Pose Estimation with MediaPipe Pose for Physical Exercises** / stereo-fusion follow-up. *MDPI Sensors* 24(23):7772, 2024. https://www.mdpi.com/1424-8220/24/23/7772
48. **The Effect of Alignment Changes on Unilateral Transtibial Amputee's Gait: A Systematic Review.** *PLOS One* 11(12):e0167466, 2016. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0167466 (PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5140067/ )
49. **Biomechanical and Kinematic Gait Analysis in Lower Limb Amputees: Cross-Sectional Study.** *JMIR Rehabilitation and Assistive Technologies*, 2025. https://rehab.jmir.org/2025/1/e67022/
50. **3D pose estimation for scalable remote gait kinematics assessment.** *npj Digital Medicine*, 2025. https://www.nature.com/articles/s41746-025-02211-y

*Numbers flagged "(not verified)" in the body could not be confirmed from the retrieved sources and should be checked against the primary text before clinical use. No DOIs, citations, or error magnitudes were fabricated.*
