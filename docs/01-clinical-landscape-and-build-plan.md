# Markerless Gait Analysis Platform — Clinical Landscape & Cost‑Minimized Build Plan

> **Purpose.** A source‑cited research brief on how gait analysis is delivered in clinics today, followed by a specific, quantitative, cost‑minimized plan to build our own markerless gait‑analysis software that is *better than OpenCap* — for research use (no real patients), with a possible commercial pivot later.
>
> **Our constraints (drive every recommendation):**
> - **Hardware budget:** ≤ **$500** beyond gear we already own.
> - **We already own:** iPhone **16e** + iPhone **13**, and an **Apple Silicon (M‑series) Mac**.
> - **Software bar:** must **beat OpenCap** and **own the IP** (don't just fork it) because we may commercialize — so every component must be **commercial‑license‑safe**.
> - **Use cases:** understand gait and how injury / pain / neurological conditions / TBI / balance / vision affect it; asymmetry detection; mobility, ROM, fall‑risk, and (carefully) muscle‑loading reporting; elderly mobility; rehab. **Research only.**
>
> **Date:** 2026‑06‑12. Prices in USD. Many vendor prices in this market are quote‑only; figures are labeled *confirmed* / *used* / *estimate* with sources.

---

## TL;DR (the decisions)

1. **Hardware:** Stay on the two iPhones. Spend ~**$300–$430** of the $500 on **2–3 tripods + phone mounts, a printed ChArUco calibration board, and good lighting**, and *optionally* a **third used/cheap camera** to break the 2‑camera accuracy ceiling. **No LiDAR exists on either phone** (LiDAR is Pro‑only), so the architecture is **calibrated multi‑view RGB + strong 2D pose + triangulation**, not depth sensing. ([iPhone 16e has no LiDAR](https://www.apple.com/newsroom/2025/02/apple-debuts-iphone-16e-a-powerful-new-member-of-the-iphone-16-family/))
2. **Software stack (all commercial‑safe licenses):** iPhone capture → **RTMPose** (Apache‑2.0) 2D pose → **Pose2Sim** (BSD‑3) calibration + robust triangulation → **OpenSim / Moco** (Apache‑2.0) inverse kinematics + dynamics → **GRF‑from‑kinematics** model for force‑plate‑free kinetics → our **own reporting layer**. **Avoid OpenPose** ($25k/yr commercial license, *excludes sports*) and **Ultralytics YOLO** (AGPL/paid). This is the single most important strategic choice for a commercial future.
3. **How we beat OpenCap:** a **1‑phone "quick mode"** OpenCap can't do (it needs ≥2 phones — see 2.3b) *plus* a multi‑camera "accurate mode"; more cameras (3–4 vs 2), a **better/commercial‑safe pose model** (RTMPose vs OpenCap's OpenPose), **fully local on Apple Silicon** (no Stanford cloud dependency), **higher‑frame‑rate capture** (120/240 fps), and — the real differentiator — a **clinical reporting + normative‑comparison + asymmetry layer** that OpenCap simply does not have.
4. **Accuracy we can credibly target:** **~3–5° MAE** sagittal‑plane joint kinematics vs marker‑based (OpenCap is 4.5° mean; Pose2Sim 3.0° walking). Be honest that **transverse‑plane rotation is hard for *everyone*** (often >5°, even marker‑based) and that **muscle *strength/structure* cannot be inferred from gait** — only *relative model‑estimated loading patterns*.
5. **Roadmap:** MVP (single‑subject, 2–3 phones, spatiotemporal + sagittal kinematics) → validation study (concurrent validity vs a reference) → commercial‑ready (multi‑subject, kinetics, polished report).

---

# PART 1 — How clinics actually deliver gait analysis today

Gait analysis in practice spans a **huge cost/fidelity range**, from a $0 stopwatch 10‑meter walk test to a **$100k–$250k+** instrumented 3D gait lab. The four settings below use very different subsets of the same metric menu.

## The metric menu (what *can* be measured)

| Category | Parameters | Who measures it |
|---|---|---|
| **Spatiotemporal** | Gait speed (m/s), cadence (steps/min), step/stride length, stance/swing %, double‑support time, step width, gait variability | Everyone (easiest, cheapest) |
| **Kinematics** | Hip/knee/ankle (+pelvis) angles in **sagittal / frontal / transverse** planes; ROM | Sports (2D sagittal/frontal), neuro/ortho lab (full 3D) |
| **Kinetics** | Ground reaction force (GRF), joint **moments & powers** | Neuro/ortho lab (force plates), some research/sports |
| **Plantar pressure** | Peak pressure, force, center‑of‑pressure (CoP) trajectory by foot region | Podiatry (primary), neuro/ortho (adjunct) |
| **EMG** | Surface or fine‑wire muscle timing/intensity | Neuro/ortho lab, research |

---

## (a) PT / Rehabilitation

**What they measure:** Mostly **spatiotemporal + functional** — gait speed, cadence, step/stride length, stance/swing %, **double‑support time**, step width, gait variability, and **inter‑limb symmetry** (a primary post‑stroke target). Full 3D kinematics/kinetics/EMG generally live in the neuro lab, not the standard PT clinic.

**Tools:** Instrumented walkways (**GAITRite / Zeno**, ~**$10k–$15k**), functional tests (10‑Meter Walk Test, 6‑Minute Walk, Berg Balance, SPPB, Timed‑Up‑and‑Go), increasingly wearables. ([Physiopedia 10MWT](https://www.physio-pedia.com/10_Metre_Walk_Test))

**Workflow:** Referral → clinical exam → functional walk tests (add 1 m lead‑in/out to remove accel/decel) → optional walkway capture (auto‑computes footfall params in minutes) → same‑visit interpretation → set rehab goals & track progress.

**Normatives used:** Gait speed **≥1.0 m/s** ≈ healthy community ambulation; **<0.8 m/s** = limited ambulation + elevated fall risk; **<0.7 m/s** ≈ up to 1.5× fall risk. Symmetry‑ratio 95% cut‑offs (healthy adults): step length **1.08**, swing time **1.06**, stance time **1.05**. ([gait speed/fall risk](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9655734/), [symmetry ratios, Patterson](https://www.sciencedirect.com/science/article/abs/pii/S0966636209006493))

## (b) Running / Sports Performance

**What they measure:** **2D‑video‑dominant**, injury‑focused. Spatiotemporal (cadence/step rate, stride length, **ground contact time**, vertical oscillation) + sagittal/frontal kinematics: **foot‑strike pattern**, **overstride**, **contralateral pelvic drop**, hip adduction/internal rotation, knee flexion at contact. ~15 qualitative variables scored at gait events. Kinetics usually *inferred*, not measured. ([SimpliFaster gait analysis](https://simplifaster.com/articles/gait-analysis/), [JOSPT 2016](https://www.jospt.org/doi/10.2519/jospt.2016.6280))

**Workflow:** Treadmill run with ~4–5 min acclimation → high‑frame‑rate sagittal + frontal (sometimes posterior) cameras → slow‑motion frame‑by‑frame review → same‑session report with retraining cues. **Cash‑pay / bundled** into an eval — *not* billed via CPT 96000.

**Normatives:** Injured runners often present at ~**156–162 steps/min**; common retraining target **+5–10%** (the "180 spm is universal" claim is a **myth** — cadence scales with speed/height). A **+10% step rate** reduces vertical loading and hip/knee adduction → a leading injury‑prevention intervention. ([e3rehab cadence](https://e3rehab.com/running-cadence/), [PMC step‑rate](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7450991/))

## (c) Podiatry

**What they measure:** **Plantar‑pressure‑dominant** — dynamic pressure distribution (mat or in‑shoe), **peak pressure/force**, loading rate by **rearfoot/midfoot/forefoot**, **CoP trajectory**, foot‑strike, pronation/supination, arch. Often paired with 2D video and a 3D foot scan for orthotics. ([Tekscan](https://www.tekscan.com/blog/medical/how-pressure-mapping-complements-force-measurement-in-gait-analysis))

**Tools:** Pressure plates/insoles (**Tekscan**, **Novel Pedar/emed**, **RSscan**, **Zebris**) — wireless insoles ~**$10k**, wired ~**$5k**; plate systems quote‑only.

**Workflow:** Walk across pressure plate (± foot scan) → auto‑generated pressure maps/CoP report → drives orthotic prescription, footwear, surgical/rehab planning. Single visit, **cash‑pay/bundled**.

**Normative threshold:** **Peak plantar pressure >700 kPa** = "elevated" / diabetic foot‑ulcer risk. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7317473/))

## (d) Neuro / Ortho Gait Lab (CP, stroke, Parkinson's, TBI)

The **full instrumented 3D gait analysis** setting — and the main user of the CPT 96000 family.

**What they measure (all 5 elements):** (1) video exam, (2) spatiotemporal, (3) **3D kinematics** hip/knee/ankle+pelvis in all three planes, (4) **kinetics** (GRF via force plates, joint moments/powers), (5) **dynamic EMG** (surface/fine‑wire). ([Vicon clinical gait](https://www.vicon.com/resources/blog/what-is-clinical-gait-analysis/), [Physiopedia IGA](https://www.physio-pedia.com/Instrumented_Gait_Analysis))

**Condition‑specific targets:**
- **Cerebral palsy:** 3D mocap + EMG + force plate; drives **single‑event multilevel surgery (SEMLS)** planning; documented to change surgical decisions. ([EFORT review](https://eor.bioscientifica.com/view/journals/eor/1/12/2058-5241.1.000052.xml))
- **Stroke:** stiff‑knee gait, foot drop, circumduction, hip‑hiking; spatiotemporal **asymmetry** the key outcome (only ~14–21% improve over inpatient rehab). ([PMC](https://journals.sagepub.com/doi/10.1177/1545968314533614))
- **Parkinson's:** increasingly **IMU‑based** — reduced stride length, increased stride variability, festination, **freezing of gait**, impaired turning, and **reduced/asymmetric arm swing** (an *early* sign that precedes leg changes). ([npj PD 2025](https://www.nature.com/articles/s41531-025-00897-1), [arm swing PD](https://pmc.ncbi.nlm.nih.gov/articles/PMC2818433/))
- **TBI / concussion:** **dual‑task gait** is the sensitive paradigm — lower dual‑task gait speed/cadence/stride length and higher variability, sometimes persisting beyond symptom resolution. ([mTBI gait review](https://pubmed.ncbi.nlm.nih.gov/29550695/))

**Workflow:** Referral → physical exam → reflective marker placement (**Plug‑in‑Gait** model) → calibration → walking trials on an **~8–10 m walkway with 2 embedded force plates**, **4–10 trials** → processing + physician interpretation + written report (often days later). **Full session 2–4 hours.**

## Billing / reimbursement (mainly settings a & d)

CPT family **96000–96004** is the formal pathway for computerized dynamic gait analysis ([AAPC](https://www.aapc.com/codes/cpt-codes-range/96000-96004/)):

| Code | What |
|---|---|
| **96000** | Computer‑based 3D **kinematics** + video (technical) |
| **96001** | + dynamic **plantar pressure** during walking |
| **96002** | Dynamic **surface EMG**, 1–12 muscles |
| **96003** | Dynamic **fine‑wire EMG**, per muscle |
| **96004** | **Physician/QHP review + written report** (professional component) |

**Reimbursement reality:** No Medicare National Coverage Determination exists; an assigned PFS fee is **not** a guarantee of coverage. In practice, computerized 3D gait analysis is reimbursed mainly as **pre‑operative planning for cerebral palsy**; for most other indications payers (Cigna, Aetna, BCBS) often deem it **experimental/not medically necessary**. **Running and podiatry are typically cash‑pay/bundled.** *(Exact CPT dollar amounts shift yearly — confirm at the [CMS PFS lookup](https://www.cms.gov/medicare/physician-fee-schedule/search/overview); third‑party aggregators put 96004 ≈ $105.)*

> **Strategic read for us:** Reimbursement is a moat *and* a wall. A research/wellness/sports‑performance positioning (cash‑pay, screening, progress‑tracking) sidesteps the "experimental/investigational" coverage problem that blocks clinical 3D gait billing outside CP surgical planning. That is the cheaper and faster commercial wedge.

---

## Commercial systems & price ranges (the market we're undercutting)

> Vendor list prices are **almost universally quote‑only**. Figures below are labeled by confidence.

### Marker‑based 3D optical mocap (gold standard)
| System | Price | Confidence |
|---|---|---|
| **Vicon** full clinical gait lab (multi‑camera + Nexus + force plates) | **~$100k–$250k+** | Estimate (industry consensus; quote‑only) |
| **Qualisys** Miqus used bundle | **~$6,200 (used)** | Used listing |
| Qualisys new + QTM software | quote‑only | Not published |

Per‑camera and full‑system list prices are not published by Vicon/Qualisys/Motion Analysis; **$100k–$250k+** is the widely cited full‑lab range. ([Vicon cameras](https://www.vicon.com/hardware/cameras/), [Qualisys](https://www.qualisys.com/cameras/miqus/))

### Markerless video systems
| System | Price model | Notes |
|---|---|---|
| **Theia3D (Theia Markerless)** | Academic = **lifetime license**; commercial = **annual fee** ($ not published) | Software + ~8 cameras + support ([Theia budgeting](https://www.theiamarkerless.com/blog/budgeting-for-a-markerless-system)) |
| **DARI Motion** | quote‑only | FDA‑cleared, all‑in‑one, sub‑minute capture |
| **Simi**, **KinaTrax**, **Qualisys markerless**, **Contemplas** | quote‑only / "very high cost" | Simi noted as most expensive |
| **OpenCap** | **Free / open‑source** | 2–8 phones + Stanford cloud; the free baseline we must beat |

### Force plates & instrumented treadmills
| Item | Price | Confidence |
|---|---|---|
| Force plate (Bertec/AMTI/Kistler), per plate | **~$6k–$12k** | Estimate |
| **Kistler 9286B** portable 3D plate | **$23,921 new / $15,000 used** | Confirmed + used |
| Bertec FIT / Zebris FDM‑T instrumented treadmill | quote‑only (tens–hundreds of $k) | Not published |
| **GAITRite / Zeno** electronic walkway | **~$10k–$15k** (intl. up to ~$30k) | Estimate |

### Pressure mats / insoles
Wireless insoles (F‑Scan/Pedar/RSscan class) **~$10k**, wired ~**$5k**; Tekscan/Novel/Zebris plate systems quote‑only. ([ResearchGate](https://www.researchgate.net/post/Can_anyone_recommend_some_pressure_insoles_for_gait_analysis_in_adults))

### IMU / wearable suites (confirmed vendor numbers)
| System | Price | Confidence |
|---|---|---|
| **Xsens MVN Awinda Starter** | **$3,790** | Confirmed |
| **Xsens MVN Awinda (full ~17 sensors)** | **$6,990** | Confirmed |
| **Xsens MVN Link** (wired) | **$12,430** | Confirmed |
| **APDM Opal** per sensor / **Mobility Lab** system | **$2,399 / ~$20,032** | Estimate |
| **Delsys Trigno Lite / Mobile** (EMG) | **$1,900 / $2,300** | Estimate |
| **Noraxon Ultium** EMG/IMU leads | $95–$595 (accessories); full system quote‑only | Confirmed (parts) |

([Xsens pricing](https://www.cgchannel.com/2021/04/xsens-cuts-price-of-its-entry-level-inertial-mocap-systems/), [APDM Opal](https://web.fibion.com/articles/opal-movement-monitor-system-pricing/), [Delsys](https://delsys.com/product/trigno-avanti/))

**Takeaway:** The cheapest *credible* commercial gait setups start around **$6k–$20k** (used Qualisys, GAITRite, Xsens, APDM) and a real 3D lab is **$100k–$250k+**. **We are building a research‑grade pipeline for ~$0–$500 of new hardware.**

---

## Accuracy & validation norms (what "good" means, with numbers)

The clinically cited bar: **< 5° error is "good"** for **sagittal**‑plane joint angles; **frontal** is moderate; **transverse** (internal/external rotation) **routinely fails the <5° bar for *every* system, including marker‑based.**

| System | Sagittal (best) | Overall mean | Transverse / worst | Kinetics |
|---|---|---|---|---|
| **OpenCap** (Uhlrich 2023, *PLoS Comp Biol*) | — | **4.5° MAE** (per‑trial 3.7–10.2°) | rotational DOF higher | GRF **6.2% BW**; moments **1.2% BW·ht** |
| **Theia3D** (Kanko 2021) | knee FE **~3.3°** | segments **<5.5°** | **>5.5°** long‑axis rotation | — |
| **Pose2Sim** (Pagnon 2022) | CMC >0.9 | **3.0° walk / 4.1° run** | hip‑run **+15° offset** | n/a |
| **Marker‑based vs bone (fluoroscopy)** | — | **2.2–5.5°** | up to **14°**; hip rot. STA **6–22°** | reference |
| **Clinical "acceptable"** | **<5° = good** | — | usually fails | — |

Key facts that shape our targets:
- **OpenCap:** 2 iPhones, validated on 10 healthy subjects, walking/squat/STS/drop‑jump. **More than 2 cameras gave no major gain *for walking*** — but accuracy **degrades on fast sport tasks** (independent validations show larger errors in cutting/landing). Headline 4.5° **masks ~10° worst‑case DOFs.** ([OpenCap paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462))
- **Theia3D** showed **lower inter‑session variability than marker‑based** mocap — because it removes marker re‑placement error. ([Kanko reliability](https://www.sciencedirect.com/science/article/abs/pii/S0021929021004346))
- **The "gold standard" is not perfect:** skin markers differ from true bone motion by **2.2–5.5° on average, up to 14°**; soft‑tissue artifact gives **~6–22°** error in hip rotation. So "ground truth" itself is only ~2–5° accurate. ([markerless clinical review PMC8884063](https://pmc.ncbi.nlm.nih.gov/articles/PMC8884063/), [hip STA](https://pmc.ncbi.nlm.nih.gov/articles/PMC7405358/))
- **Pose‑model benchmarks (COCO AP / FPS):** **RTMPose‑m 75.8% AP, 90+ FPS CPU / 430+ FPS GPU**; RTMPose‑s 72.2%; AlphaPose ~71–73%; **OpenPose 61.8%** (improved 64.2%); **BlazePose is markedly weaker for biomechanics** (only ~65% keypoint detection in gait studies). COCO AP ≠ joint‑angle accuracy, but RTMPose dominates OpenPose on both accuracy *and* speed *and* license. ([RTMPose](https://arxiv.org/abs/2303.07399), [2D‑pose gait accuracy](https://www.sciencedirect.com/science/article/abs/pii/S0966636222004738), [GRF from 2D pose](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823796/))

---

# PART 2 — The build plan (cost‑minimized, "beat OpenCap")

## 2.0 Hardware reality check

- **Neither phone has LiDAR.** LiDAR ships **only on Pro/Pro Max**. The **iPhone 16e** (A18, single 48 MP camera, released 2025‑02‑28) and base **iPhone 13** both lack it. ([Apple newsroom 16e](https://www.apple.com/newsroom/2025/02/apple-debuts-iphone-16e-a-powerful-new-member-of-the-iphone-16-family/)) → Apple Vision's **metric (meters) 3D body pose requires LiDAR**, so we **cannot** rely on single‑phone depth. **Multi‑view triangulation with a checkerboard for scale is the correct architecture** and sidesteps the LiDAR gap entirely.
- **Both phones shoot 120/240 fps slo‑mo** — a real advantage over typical webcams for capturing fast gait events (initial contact, toe‑off).
- **Apple Silicon Mac** runs the whole pipeline locally via **MPS/CoreML/ONNX Runtime** — no cloud needed (unlike OpenCap's Stanford dependency). One caveat to verify: **OpenSim's native arm64 packaging** has historically been x86_64‑only on macOS; plan an **x86_64 conda env under Rosetta 2** as a fallback. ([OpenSim conda](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53116061/))

## 2.1 The capture rig — Bill of Materials (≤ $500)

| Item | Qty | Est. cost | Why |
|---|---|---|---|
| iPhone 16e + iPhone 13 | (own) | **$0** | Two calibrated views — matches OpenCap's baseline |
| Sturdy phone tripods (~1.5 m, e.g. UBeesize/Manfrotto‑class) | 2–3 | **$60–$120** | Stable, repeatable camera placement |
| Phone tripod mounts/clamps | 2–3 | **$15–$30** | Attach phones to tripods |
| **Printed ChArUco / checkerboard board** (rigid foam‑board or aluminum) | 1 | **$10–$40** | Intrinsic + extrinsic calibration & metric scale (the LiDAR replacement) |
| Work/LED lights (softbox or 2× LED panels) | 2 | **$40–$80** | Reduce motion blur → enables 120 fps; cleaner pose |
| Non‑slip floor markers / 8–10 m tape walkway | 1 | **$10–$20** | Define capture volume & a known‑distance ground truth for spatiotemporal validation |
| **Optional: 3rd/4th camera** — used iPhone/Android or a $40–$120 action cam (120 fps) | 1–2 | **$0–$240** | **Break the 2‑camera ceiling** → the easiest accuracy win over OpenCap |
| **Contingency** | — | remainder | Cables, SD cards, mounts |
| **Total** | | **~$135 (2 phones) → ~$430 (4 cameras)** | **Under $500** |

**Capture protocol:** 1080p @ **60 fps** (use **120/240 fps** for running) on all cameras; cameras at **~45° offsets** around an 8–10 m walkway covering ≥1 full gait cycle per view; **3–5 trials** each direction. Lock exposure/focus. **Sync** via OpenCap's web app (synchronizes simultaneous iOS recordings) *or* a **clap/flash** spike for post‑hoc alignment; interpolate from high‑fps to reduce sync error. Calibrate once per session: one ChArUco capture per camera for extrinsics. ([OpenCap iPhone app](https://github.com/stanfordnmbl/opencap-iphone))

## 2.2 Software architecture — the commercial‑safe stack

```
[2–4 iPhone/cam videos]
        │  (software sync: OpenCap web app / clap / flash)
        ▼
[Camera calibration]  ── OpenCV ChArUco (Apache‑2.0): intrinsics + extrinsics + metric scale
        ▼
[2D pose per view]    ── RTMPose via rtmlib (Apache‑2.0) on Apple Silicon (MPS / CoreML / ONNX)
        ▼
[Robust 3D triangulation] ── Pose2Sim (BSD‑3): multi‑view RANSAC triangulation, filtering, OpenSim export
        ▼
[Biomechanical model] ── OpenSim + Moco (Apache‑2.0): scaling, inverse kinematics (angles), inverse dynamics
        ▼
[Force‑plate‑free kinetics] ── GRF‑from‑kinematics (OpenGRF / trained LSTM) + Moco predictive sim
        ▼
[OUR reporting + normative + asymmetry layer]  ◄── the real product / differentiator / IP
```

**Component choices and *why* (license is decisive for a commercial pivot):**

| Layer | Choice | License | Why it beats the obvious alternative |
|---|---|---|---|
| 2D pose | **RTMPose** (MMPose / `rtmlib`) | **Apache‑2.0** ✅ | Higher COCO AP (75.8 vs 61.8) *and* faster *and* free for commercial use, vs **OpenPose** which is **$25k/yr and explicitly excludes Sports**. Avoid **Ultralytics YOLO** (AGPL → must open‑source your app, or pay Enterprise). |
| Mobile on‑device option | **MediaPipe** / **Apple Vision** | Apache‑2.0 / free SDK | For a phone‑only "lite" mode later; lower biomech accuracy, so not the primary path. |
| Calibration | **OpenCV ChArUco** | **Apache‑2.0** ✅ | Standard, robust, native arm64. |
| 3D triangulation | **Pose2Sim** | **BSD‑3** ✅ | Ships with RTMPose built in, OpenSim‑ready output, robust multi‑view RANSAC. **AniPose** (BSD‑3) is the fallback. |
| Biomechanics | **OpenSim + Moco** | **Apache‑2.0** ✅ | Gold‑standard musculoskeletal modeling; commercial‑safe. (Watch arm64 packaging.) |
| Kinetics w/o force plate | **OpenGRF** + trained **LSTM** | open | GRF from kinematics: vertical nRMSE ≤1.5%; med‑lateral is the hard axis (~18% error). ([OpenGRF](https://www.biorxiv.org/content/10.1101/2025.09.27.678739v1)) |
| Auto model fitting | **AddBiomechanics** *(dev only)* | free cloud | ⚠️ **requires sharing de‑identified data** — fine for research, **not** for a closed commercial pipeline. Use for prototyping, **replicate locally** for product. |

> **Commercial‑license bottom line:** every shipping component above is **Apache‑2.0 / BSD‑3 / MIT** or a free platform SDK. The only "traps" — **OpenPose** ($25k/yr, no sports), **Ultralytics AGPL**, and **AddBiomechanics' data‑sharing requirement** — are explicitly **excluded from the product path**. This is exactly where OpenCap (built on OpenPose) is *legally* hard to commercialize, and where we start ahead.

## 2.3 How we concretely exceed OpenCap

| OpenCap weakness | Our improvement | Expected effect |
|---|---|---|
| **Requires ≥2 phones** — no single‑camera option at all | **1‑phone "quick mode"** (monocular SMPL → OpenSim) *plus* multi‑cam "accurate mode" (see 2.3b) | The record‑one‑phone‑get‑a‑3D‑model experience OpenCap lacks, with an honest fidelity tier |
| **2‑camera default**, accuracy ceiling, worst‑case DOFs ~10° | **3–4 calibrated cameras**, RANSAC triangulation | Lower per‑DOF variance, better frontal/transverse, robustness to occlusion |
| **OpenPose dependency** (academic/$25k‑sports‑excluded; older 61.8 AP) | **RTMPose** (Apache, 75.8 AP, faster) | Better keypoints, commercial‑safe, faster |
| **Cloud dependency** (Stanford servers, academic‑use ToS) | **Fully local on Apple Silicon** | Privacy, no per‑use cost, no ToS limit, offline |
| **Standard 60 fps capture** | **120/240 fps** for running/fast events | Cleaner event detection, sport use cases |
| **Validated only on healthy + few tasks; degrades on sport** | **Task‑specific models + our own validation study** | Defensible accuracy claims per use case |
| **No clinical report** — outputs are raw kinematics/`.osim` | **Full normative‑comparison reporting + asymmetry indices + fall‑risk + neuro markers** | **The actual product.** OpenCap stops where we start. |

## 2.3b Single‑camera (1‑phone) mode — feasibility, accuracy, and design

> **Myth‑buster first:** OpenCap is **not** a single‑phone system. It requires a **minimum of two iPhones** — Stanford's own headline is *"movement analysis is now **two** smartphones away."* They use multi‑view triangulation precisely **because single‑view 3D is less accurate**, and because they wanted defensible **kinetics**. The 3D model you see in OpenCap is a *two‑camera* result. ([Stanford HPA](https://humanperformance.stanford.edu/news/human-movement-analysis-is-now-two-smartphones-away-with-opencap-software/), [OpenCap paper](https://mobl.mech.utah.edu/wp-content/uploads/2023/12/opencap.pdf))

**So can we do 3D from one phone? Yes — and OpenCap can't, which is our opening.** It's a different technique: instead of triangulating two views, a single view uses a **monocular 3D human‑mesh model** (SMPL‑family: e.g., CameraHMR/HybrIK/4D‑Humans) to *infer* a full 3D body, which then drives **OpenSim inverse kinematics** end‑to‑end. The cost is accuracy.

**What the literature says single‑camera 3D gait actually delivers:**

| Approach | Accuracy (vs marker‑based / OpenCap) | Notes |
|---|---|---|
| **Monocular SMPL → OpenSim IK** ("calibrationless monocular musculoskeletal simulation during gait") | **best mean 8.5° MAE across joints, range 3.7–21.6°**, from a **45° side view**; validated on 19 subjects + 4 gait patterns vs OpenCap + marker‑based | Frontal/transverse planes are unreliable; "IK cannot fully compensate" for single‑camera limits ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S240584402408109X), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11168395/)) |
| **Monocular 3D pose lifting** (MotionBERT etc.) | ~39 mm MPJPE on Human3.6M *lab benchmark* | Lab number, not clinical joint‑angle error; scale/depth ambiguous ([MotionBERT class](https://arxiv.org/pdf/2310.16288)) |
| **Sports2D** (single camera, **2D only**) | sagittal/frontal angles; subject must move **parallel** to the camera plane | True 2D — great for quick sagittal gait, **not** 3D ([Sports2D](https://github.com/davidpagnon/Sports2D)) |
| **MonoMSK** (2025, frontier) | monocular 3D *musculoskeletal dynamics* | Research‑grade, worth tracking ([arXiv](https://arxiv.org/html/2511.19326v1)) |

**The honest tradeoff (updated June 2026):** a *naive* single‑camera→IK pipeline lands around **~8.5° average error (up to ~21° in some DOFs)**. **But the ceiling is far higher:** **OpenCap Monocular** (Gilon, Miller & Uhlrich 2026, [arXiv 2603.24733](https://arxiv.org/abs/2603.24733)) reaches **4.8° MAE / 3.4 cm pelvis from one static smartphone video — matching 2‑camera OpenCap** — by refining a monocular pose estimate (**WHAM**) via optimization against a biomechanically‑constrained skeletal model, then estimating kinetics with physics + ML. So single‑camera 3D is no longer inherently ~2× worse; with the right refinement it is competitive. ⚠️ **Commercial caveat:** WHAM is **SMPL‑based (non‑commercial)** — for our commercial pivot we need a non‑SMPL monocular backbone or an SMPL licence (same class of landmine as OpenPose). Frontal/transverse and asymmetry remain harder single‑camera regardless; accurate mode (2 phones) is still the most defensible there.

**The design decision — ship BOTH, because we own two phones:**

| Mode | Cameras | Pipeline | Accuracy | Use it for |
|---|---|---|---|---|
| **Quick mode** | **1 phone** | iPhone → RTMPose + **monocular SMPL (CameraHMR‑class)** → OpenSim IK → report | ~8.5° MAE; sagittal OK, frontal weak | Screening, field/home capture, demos, the "record‑and‑get‑a‑model" wow factor, longitudinal self‑tracking |
| **Accurate mode** | **2–4 phones** | iPhone → RTMPose → **Pose2Sim triangulation** → OpenSim → kinetics | ~3–5° MAE; matches/beats OpenCap | Research‑grade kinematics, asymmetry indices, kinetics, any defensible claim |

This is strictly **better than OpenCap**: we offer the **1‑phone end‑to‑end experience OpenCap doesn't have** *and* a 2+‑phone mode that equals it — same app, same report, user picks the fidelity tier. The report clearly labels which mode produced it and **suppresses frontal‑plane/asymmetry metrics in quick mode** (or flags them low‑confidence) so we never over‑claim.

## 2.4 The reporting layer (the "full report" we want)

This is where the IP and differentiation live. The report computes, per session, against age/sex‑matched norms:

**1. Spatiotemporal panel** — gait speed, cadence, step/stride length, stance/swing %, double‑support, step width, variability; flagged vs norms (e.g., gait speed <0.8 m/s, the "sixth vital sign"). ([Studenski JAMA 2011: HR 0.88 per +0.1 m/s](https://pubmed.ncbi.nlm.nih.gov/21205966/))

**2. Kinematic panel** — hip/knee/ankle ROM & curves vs normative bands. Sagittal norms during gait: hip ROM ~40–48°, knee peak swing ~60°, ankle ROM ~25–30°. ([Physiopedia ROM](https://www.physio-pedia.com/Joint_Range_of_Motion_During_Gait))

**3. Asymmetry panel** — multiple indices, clearly defined:
- **Symmetry Index** (Robinson): `SI = (X_R − X_L)/(0.5(|X_R|+|X_L|)) × 100`.
- **Symmetry Ratio** (recommended for stroke by Patterson 2010).
- **Gait Asymmetry** `GA = 100·|ln(X_R/X_L)|`.
- **GDI** (Gait Deviation Index, Schwartz & Rozumalski 2008): 100 = healthy mean; every 10 pts below = 1 SD.
- **GPS / Movement Analysis Profile** (Baker 2009): RMS deviation from norm, per‑variable bar chart.
([GDI](https://pubmed.ncbi.nlm.nih.gov/18565753/), [GPS](https://pubmed.ncbi.nlm.nih.gov/19632117/), [Patterson](https://pubmed.ncbi.nlm.nih.gov/19932621/))

**4. Fall‑risk / mobility panel** — gait speed bands, TUG (≥13.5 s cut‑point), gait variability, step‑width variability.

**5. Neuro markers** (research) — stride‑time variability & arm‑swing asymmetry (Parkinson's), spatiotemporal asymmetry (stroke), **dual‑task gait** decrement (TBI/concussion).

**6. Kinetics panel** (when GRF model is enabled) — joint moments/powers and **inter‑limb moment asymmetry** (rigorous from inverse dynamics).

**7. Muscle panel — *carefully framed*** — OpenSim Static‑Optimization muscle **loading patterns**, labeled *"musculoskeletal‑model estimates, relative not absolute, most reliable for ankle plantar/dorsiflexors."*

**Normative reference datasets (public, CC‑BY — check each record):**
| Dataset | N | Contents | Best for |
|---|---|---|---|
| **Fukuchi 2018** (PeerJ) | 42 (young+older) | kinematics+kinetics, overground+treadmill, multi‑speed | **Age‑stratified, speed‑dependent norms** ([PeerJ](https://peerj.com/articles/4640/)) |
| **Schreiber & Moissenet 2019** (Sci Data) | 50 | kinematics+kinetics, 5 speeds | Speed‑modulation norms ([Nature](https://www.nature.com/articles/s41597-019-0124-4)) |
| **Lencioni 2019** (Sci Data) | ~50 | kinematics+kinetics+**EMG**, stairs/toe/heel | EMG activation templates ([Nature](https://www.nature.com/articles/s41597-019-0323-z)) |
| **AddBiomechanics 1.0** (2024) | **273 subj, >24M frames** | OpenSim‑ready kinematics+kinetics, 9 activities | **ML normative modeling** ([arXiv](https://arxiv.org/abs/2406.18537)) |

## 2.5 Scientific honesty guardrails (protect the brand & avoid FDA/false‑claims risk)

**Defensibly reportable from our pipeline:** joint angles, spatiotemporal parameters, ROM, GDI/GPS, symmetry indices, and — with the GRF model — net joint moments/powers and moment asymmetry.

**Report only with explicit caveats:** model‑estimated muscle **loading patterns** (relative, timing‑based, most reliable distally).

**Do NOT claim** (scientifically unsupported from gait kinematics): muscle **strength / maximal force capacity**, muscle **size / "muscular structure" / hypertrophy**, or accurate **co‑contraction** without EMG. Strength needs dynamometry; structure needs MRI/ultrasound. ([SO vs EMG, Heintz 2007](https://pubmed.ncbi.nlm.nih.gov/17071088/)) — *This boundary is non‑negotiable: it's both scientific integrity and, for a future product, regulatory/false‑advertising protection.*

## 2.6 Phased roadmap

**Phase 0 — Environment (week 1, $0).** Set up Mac: Python env, `rtmlib`/RTMPose, OpenCV, Pose2Sim, OpenSim (native arm64 if available, else Rosetta x86_64 conda). Smoke‑test RTMPose on a single iPhone clip.

**Phase 1 — MVP (weeks 2–5, $0–$150).** Build **both capture modes from day one** since they share everything downstream of pose:
- *Quick mode (1 phone):* iPhone clip → RTMPose → monocular SMPL (CameraHMR‑class) → OpenSim IK → sagittal report. This delivers the "record on one phone, get a 3D model + analysis" experience first — the fastest path to a demo.
- *Accurate mode (2 phones + 2 tripods):* ChArUco calibration → RTMPose → Pose2Sim triangulation → OpenSim IK.
**Output: spatiotemporal params + sagittal hip/knee/ankle angles** for one subject (a team member). Compare step length against tape‑measured ground truth, and quick‑mode vs accurate‑mode against each other. *Target: spatiotemporal within ~5% of tape; accurate‑mode sagittal angles ≤5°; quick‑mode sagittal ~8° with frontal flagged low‑confidence.*

**Phase 2 — Reporting + norms (weeks 6–9, $0).** Build the reporting layer: normative comparison (Fukuchi/Schreiber), asymmetry indices (SI/SR/GDI/GPS), fall‑risk & ROM panels, PDF/HTML report generator. Add 3rd/4th camera (+$0–$240) to improve frontal/transverse.

**Phase 3 — Kinetics (weeks 10–14, $0).** Integrate GRF‑from‑kinematics (OpenGRF + train an LSTM on AddBiomechanics) → inverse dynamics → joint moments/powers + Moco muscle loading. **Caveat med‑lateral GRF.**

**Phase 4 — Validation study (weeks 12–18).** *See 2.7.* Establish concurrent validity → defensible accuracy claims.

**Phase 5 — Commercial‑ready (post‑validation).** Multi‑subject batch, polished report, on‑device "lite" mode (MediaPipe/Apple Vision), de‑identification, local replication of any AddBiomechanics step, license audit, and a use‑case‑specific accuracy spec sheet.

## 2.7 Validation approach (how we earn the right to claim accuracy)

**Concurrent‑validity study design:** record subjects **simultaneously** with our system and a **reference**, then report agreement (RMSE in degrees, Bland‑Altman limits of agreement, CMC, ICC) per joint/plane.

**Reference options, cheapest first:**
1. **OpenCap itself** as a cross‑check (free) — shows we match/beat a published baseline on the same clips.
2. **A public benchmark dataset** with synchronized marker‑based ground truth (e.g., reprocess shared video+mocap pairs).
3. **Borrow time** on a university/partner **marker‑based or Theia3D lab** for a small N (gold‑standard concurrent capture) — the most credible, modest cost.

**Acceptance targets (pre‑registered):** sagittal hip/knee/ankle **MAE ≤ 5°**; spatiotemporal **≤ 5%** / step length **≤ 2–3 cm**; **report transverse‑plane honestly** as the known weak axis for all markerless systems. Use **≥10–15 subjects, multiple trials**, and report per‑DOF — *don't* hide behind a single mean like the "4.5°" headline.

## 2.8 Risks & mitigations
- **OpenSim arm64 packaging** → Rosetta x86_64 conda fallback; verify current channel first.
- **Transverse‑plane accuracy** → more cameras + honest reporting; never gate a clinical claim on hip rotation.
- **Sync error at high fps** → hardware‑free clap/flash + interpolation; or adopt OpenCap's web‑app sync.
- **Med‑lateral GRF error (~18%)** → report vertical/AP confidently, flag M‑L as estimate.
- **Scope creep into "muscle strength" claims** → hard guardrail (2.5).
- **License drift** (a dependency relicenses) → quarterly license audit; pin commercial‑safe versions.

---

## Sources (primary)

**Validation/accuracy:** OpenCap — Uhlrich 2023, *PLoS Comp Biol* e1011462 ([link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462)). Theia3D — Kanko 2021, *J Biomech* 110665 ([link](https://www.sciencedirect.com/science/article/abs/pii/S0021929021004346)). Pose2Sim — Pagnon 2022, *Sensors* 22:2712 ([link](https://pmc.ncbi.nlm.nih.gov/articles/PMC9002957/)). Markerless clinical review — Wade 2022, *PeerJ* ([link](https://pmc.ncbi.nlm.nih.gov/articles/PMC8884063/)). Hip soft‑tissue artifact — Fiorentino ([link](https://pmc.ncbi.nlm.nih.gov/articles/PMC7405358/)). 2D‑pose gait accuracy — Washabaugh 2022 ([link](https://www.sciencedirect.com/science/article/abs/pii/S0966636222004738)). RTMPose ([link](https://arxiv.org/abs/2303.07399)).

**Pipeline/licenses:** RTMPose/MMPose ([github](https://github.com/open-mmlab/mmpose)), Pose2Sim ([github](https://github.com/perfanalytics/pose2sim)), AniPose ([github](https://github.com/lambdaloop/anipose)), OpenSim ([github](https://github.com/opensim-org/opensim-core)), AddBiomechanics ([site](https://addbiomechanics.org)), OpenCap code ([github](https://github.com/stanfordnmbl/opencap-core)), OpenPose license ([link](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)), Ultralytics license ([link](https://www.ultralytics.com/license)), OpenGRF ([biorxiv](https://www.biorxiv.org/content/10.1101/2025.09.27.678739v1)), iPhone 16e no‑LiDAR ([Apple](https://www.apple.com/newsroom/2025/02/apple-debuts-iphone-16e-a-powerful-new-member-of-the-iphone-16-family/)).

**Clinical/normative:** CPT 96000‑96004 ([AAPC](https://www.aapc.com/codes/cpt-codes-range/96000-96004/)). Bohannon gait speed ([PubMed](https://pubmed.ncbi.nlm.nih.gov/9143432/)). Studenski 2011 *JAMA* ([link](https://pubmed.ncbi.nlm.nih.gov/21205966/)). GDI — Schwartz 2008 ([link](https://pubmed.ncbi.nlm.nih.gov/18565753/)). GPS — Baker 2009 ([link](https://pubmed.ncbi.nlm.nih.gov/19632117/)). Patterson stroke symmetry ([link](https://pubmed.ncbi.nlm.nih.gov/19932621/)). Fukuchi 2018 ([PeerJ](https://peerj.com/articles/4640/)). Schreiber & Moissenet 2019 ([Nature](https://www.nature.com/articles/s41597-019-0124-4)). Lencioni 2019 ([Nature](https://www.nature.com/articles/s41597-019-0323-z)). AddBiomechanics dataset ([arXiv](https://arxiv.org/abs/2406.18537)). SO vs EMG — Heintz 2007 ([link](https://pubmed.ncbi.nlm.nih.gov/17071088/)). mTBI gait ([link](https://pubmed.ncbi.nlm.nih.gov/29550695/)). Parkinson's digital biomarkers ([Nature](https://www.nature.com/articles/s41531-025-00897-1)).

> **Verification caveats:** Vendor prices are largely quote‑only (Vicon/Theia/AMTI/Bertec/Tekscan/Novel) — ranges are best‑available estimates, not list prices. CPT dollar amounts shift yearly — confirm at the CMS PFS lookup. OpenCap's journal is *PLoS Comp Biol* (not eLife). Confirm OpenSim's current native‑arm64 conda availability and the exact Lightning‑Pose LICENSE before locking the stack.
