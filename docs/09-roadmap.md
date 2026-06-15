# Roadmap / task list (living document)

Status: ✅ done · 🟡 in progress · ⬜ todo. This is the working backlog; update as we go.

## A. Capture & upload pipeline  *(the headline goal)*
- ✅ A1. Upload a video in the web app → background job → report (UI + job/status polling)
- 🟡 A2. **Monocular (1-phone) end-to-end**: video → MediaPipe 3D → marked OpenSim model → IK → `.mot` → report. Blocked on B2 (validated marked model).
- ⬜ A3. **2-phone (Pose2Sim) in the app**: multi-video upload + calibration handling → kinematics → report (Track A works on CLI; wrap it in the app).
- ⬜ A4. **Connect phone to computer** (research + build): live camera link (WebRTC/RTSP/USB). Decide iOS/Android path.
- ⬜ A5. **Remote record control**: start/stop phone recording *from the computer*; multi-phone sync for 2-camera capture.
- ⬜ A6. **Auto-ingest**: recorded clip flows straight into processing with no manual upload.

## B. Biomechanics engine
- ✅ B1. Pose2Sim accurate pipeline (Track A) — validated `.mot` produced
- 🟡 B2. **Marked model build + concurrent validation vs Track A** (`gait-build-model`, `gait-validate`; need base `.osim` + run on the Mac). Acceptance: sagittal RMSE ≤ ~5°.
- ⬜ B3. Monocular 3D quality: assess MediaPipe world-landmarks vs Pose2Sim; consider a better lifter.
- ⬜ B4. Frontal/transverse reliability — markerless weak axis; quantify and label honestly.

## C. Clinical analysis & report (task-specific)
- ✅ C1. Task detection (gait / squat / sit-to-stand)
- ✅ C2. Task-specific metric panels (each action shows what matters clinically)
- ✅ C3. Gait-cycle-normalized ensemble curves (mean ± SD across strides) + normal band
- ✅ C4. **Full spatiotemporal from the `.trc`**: walking speed, stride length (metric), stance/swing %, step width, symmetry (`spatiotemporal_3d.py`)
- ✅ C10. **OpenSim visuals — layer 1**: animated 3D motion playback in the report from the marker `.trc` (`viz3d.py`, three.js)
- ⬜ C11. **OpenSim visuals — layer 2**: render the full musculoskeletal **model geometry** (bones/muscles) driven by the `.mot` (export body transforms via the OpenSim API + load mesh geometry into the same three.js viewer)
- ⬜ C5. Squat depth: descent vs ascent split, depth-over-time, true frontal-plane valgus angle
- ⬜ C6. Sit-to-stand: triple-extension coordination timing, trunk-flexion momentum
- ⬜ C7. Summary indices: GDI / GPS (gait), composite squat/STS scores
- ⬜ C8. **Trial-over-time comparison** — overlay a patient's visits to track change (MCID-aware)
- ⬜ C9. Population-matched normative database (replace the approximate band)

## D. Deep clinical research & grounding  *(non-superficial — the "why")*
- ✅ D1+D2+D3. **Clinical evidence base** (`docs/11-clinical-evidence.md`, 50 refs): use-cases by condition (ACL/PFP, OA/TKA/THA, stroke, CP, Parkinson's, falls/aging, amputee), metric-by-metric reference values + MCIDs, and markerless validity per joint & plane (the numbers we hold ourselves to)
- ⬜ D4. Data & regulatory: research-use framing, consent/PHI, the line where it becomes clinical (BAA, etc.)

## E. UI / UX
- ⬜ E1. **Clear capture instructions** for users: camera placement, distance, height, lighting, framing, clothing, calibration object, walkway length, # of reps
- ✅ E2a. OpenSim *onboarding* (install/setup page + capability banner) — `/setup`, docs/10
- 🟡 E2b. **OpenSim *visuals*** (what "adding OpenSim" actually meant): 3D model in the report. Layer 1 done (C10); full model geometry is C11.
- 🟡 E3. Session / subject management UI (sessions list + report viewer exist; add subjects, visits, delete)
- 🟡 E4. Report polish (task panels in; keep refining wording, references, layout)

## F. Deployment & infra
- ✅ F1. Dockerfile + Azure scaffolding (Container Apps + processing-worker pattern) + docker-compose + CI template
- ⏸️ F2. Persistence (managed DB/storage/auth) — on hold per your call; on-disk store for now

---
### Suggested order
1. **E1 + E2** (capture instructions + OpenSim onboarding) — small, high-leverage, unblocks you actually capturing usable video.
2. **D2 + D3** (clinic use-cases + validity research) — the non-superficial grounding for what we measure and why.
3. **B2** (validate the marked model) — lights up A2 (monocular end-to-end).
4. **C4–C8** (spatiotemporal, squat/STS depth, trial-over-time) — clinical richness.
5. **A3 → A4/A5/A6** (2-phone in app → phone connection → remote record → auto-ingest).
