# Plan: real iPhone clip → joint angles, two tracks in parallel

**Objective:** get a real capture to flow all the way to the OpenCap-style joint-angle set
(`.mot`) and our clinical report/signatures, on two tracks at once:

- **Track A (accurate, 2 phones):** reuse **Pose2Sim**'s validated OpenSim model + IK to get a
  trustworthy result fast, and to serve as the **reference** that validates Track B.
- **Track B (quick, 1 phone, our IP):** build + validate **our own marked LaiUhlrich model** so a
  single BlazePose video produces joint angles we own end-to-end.

Both tracks converge on the same downstream we already shipped: `analysis/kinematics.py` →
`analysis/signatures.py` → report (the differentiator).

```
Track A: 2 videos → RTMPose → Pose2Sim (calib, triangulate, OpenSim IK) ─┐
Track B: 1 video  → MediaPipe 3D → our .trc → our marked .osim → OpenSim IK ─┤
                                                                            ▼
                                          coordinates.mot → kinematics → signatures → report
```

## Why this sequencing
Track A reuses assets that already exist and are validated, so it de-risks *our* downstream on
real joint angles before we invest in custom IP. Track B is the single-phone differentiator the
project is really about; Track A's output is the yardstick we validate Track B against
(concurrent comparison on the same subject/clip).

---

## Track A — accurate mode via Pose2Sim

**The crux is already solved by Pose2Sim:** it bundles an OpenSim model with markers named to its
keypoint format (HALPE_26 from RTMPose), the scale setup, and the IK setup, and runs OpenSim
scaling + IK for you. We delegate the heavy biomechanics to it and keep our analysis layer on top.

**Deliverables**
- `biomech/pose2sim_runner.py` — generates a Pose2Sim `Config.toml` tuned for 2 iPhones, drives the
  Pose2Sim steps (calibration → pose → sync → triangulation → filter → kinematics), returns the `.mot`.
- `pipeline.run_accurate()` — Track-A chain → our `report_from_mot()`.
- Capture/calibration runbook: 2 phones on tripods, ~45° apart; print a **ChArUco** board; record a
  short **calibration** clip (board visible in both views) + the **trial**; sync via OpenCap web app
  or a clap/flash. (BOM already in [docs/01 §2.1].)

**Run (on the Mac, once `pip install pose2sim` + OpenSim are present)**
```
gait-pipeline --accurate --project my_session --speed 1.2
```

## Track B — quick mode, our own marked model

**The crux we must solve:** the LaiUhlrich2022 `.osim` ships with *its own* anatomical marker set,
not markers named like our BlazePose keypoints. So we **inject** virtual markers named like our
keypoints, placed at the corresponding joint centers, producing a model our `.trc` can drive.

**Deliverables**
- `biomech/marker_placement.py` — spec: each active marker → (OpenSim body, placement reference).
  We place markers at **joint centers** (hip/knee/ankle/shoulder/elbow/wrist) rather than guessing
  skin-marker offsets, because video keypoints already approximate joint centers.
- `biomech/build_marked_model.py` — reads a base LaiUhlrich `.osim` and writes
  `LaiUhlrich2022_ga_markers.osim` with our named markers added (OpenSim API; lazy import).
- `opensim_setup.write_scale_setup_xml()` — generate the ScaleTool setup too (we already generate IK).
- Runbook: obtain the base model (OpenCap `opencap-core` / SimTK / Pose2Sim bundle, all open), run
  the marker-injection tool, then `gait-ik`.

**Run**
```
# one-time: build the marked model from a base .osim you downloaded
python -m gait_analysis.biomech.build_marked_model --base LaiUhlrich2022.osim --out LaiUhlrich2022_ga_markers.osim
# then the single-phone chain
gait-pipeline --video data/walk.mov --model LaiUhlrich2022_ga_markers.osim --speed 1.2
```

## Validation (how we earn accuracy claims)
1. **Track A vs literature:** Pose2Sim/OpenCap report ~3–5° sagittal — sanity-check our `.mot`
   against published norms (Fukuchi) and the OpenCap per-task error band.
2. **Track B vs Track A (concurrent):** capture the *same* subject with 1 phone (B) and 2 phones (A)
   simultaneously; compare per-coordinate RMSE / Bland-Altman. Target sagittal MAE ≤ ~5° (matching
   single-phone OpenCap Monocular's 4.8°); report transverse honestly as the weak axis.
3. **Refinement step (later):** adopt OpenCap-Monocular's *optimize-against-the-biomechanical-model*
   idea to push Track B from raw-landmark IK toward 4.8°. Resolve the SMPL/commercial backbone then.

## What runs offline now vs needs the Mac
- **Offline (tested in CI):** config/XML generation, marker-placement spec ↔ markerset consistency,
  `.trc`/`.mot` parsing, marker-validation guard, the whole analysis/report/signatures path from a `.mot`.
- **On the Mac (you run):** MediaPipe, Pose2Sim, OpenSim scaling/IK, and the marker-injection tool —
  none of which can run in the build sandbox. The validation guards fail loudly rather than fabricating.

## Risks
- **Marker correspondence (Track B):** joint-center markers ≠ the model's skin-marker convention →
  IK offsets. Mitigation: start sagittal-only, validate against Track A, iterate placement.
- **OpenSim arm64:** likely needs an x86_64 Rosetta conda env (already in `setup/setup_macos.sh`).
- **Pose2Sim API drift:** pin a version; wrap behind `pose2sim_runner.py` so upgrades are localized.
- **SMPL licence (refinement):** OpenCap-Monocular-grade accuracy via WHAM is non-commercial; the
  commercial-safe non-SMPL backbone is a separate, scheduled task.

---

## Track B end-to-end workflow (build + validate the marked model)

Run on the Mac in the conda `gait` env (OpenSim present). Steps 2-4 need a base `.osim`.

1. **Get a base full-body model** (`LaiUhlrich2022.osim` or similar) from an open source:
   OpenCap `opencap-core` (`opensimPipeline/Models`), SimTK, or the Pose2Sim bundle. All open.
2. **Inspect it** so we use its REAL names (don't trust guesses):
   ```
   gait-inspect-model --model LaiUhlrich2022.osim
   ```
   Paste the BODIES / JOINTS output back here; if a name differs from `biomech/marker_placement.py`
   (e.g. `walking_knee_r` vs `knee_r` -- already aliased), we reconcile the spec.
3. **Inject our markers** at the joint centres:
   ```
   gait-build-model --base LaiUhlrich2022.osim --out LaiUhlrich2022_ga.osim
   ```
4. **Produce a Track-B `.mot`** from a single video (quick mode): MediaPipe 3D -> `.trc` ->
   scale (`gait-scale`) -> IK (`gait-ik`), or `gait-pipeline --video ... --model LaiUhlrich2022_ga.osim`.
5. **Validate against Track A** (the Pose2Sim `.mot` you already produced), ideally the SAME trial:
   ```
   gait-validate --ref pose2sim_trial.mot --test trackB_trial.mot
   ```
   **Acceptance:** sagittal RMSE mean <= ~5 deg (matches single-phone OpenCap Monocular's 4.8 deg).
   Report frontal/transverse honestly as the weak axis. If a coordinate has a large *bias* (constant
   offset) but high *r*, it's a marker-placement offset -> nudge that marker in `marker_placement.py`
   and rebuild; iterate.
6. **Ship it:** once sagittal RMSE passes, set `GAIT_OSIM_MODEL=LaiUhlrich2022_ga.osim` so the web
   app's single-video flow produces real angles end-to-end.

The `gait-validate` tool runs offline today; it's the objective gate that turns "the marked model
exists" into "the marked model is trustworthy."
