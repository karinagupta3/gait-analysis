# Gait Analysis

Research project to build a **markerless gait‑analysis platform** that is better than
[OpenCap](https://www.opencap.ai/) — for research use (no real patients), with a possible
commercial pivot later. Capture runs on two iPhones (16e + 13) and an Apple Silicon Mac,
with a target of **≤ $500** in additional hardware.

## Goal

Understand how gait is affected by injury, pain, neurological conditions, TBI, balance,
vision, and aging — producing a full report covering **mobility, joint kinematics/ROM,
asymmetry, fall risk, and (carefully framed) muscle‑loading estimates**.

## Where to start

- **[`docs/01-clinical-landscape-and-build-plan.md`](docs/01-clinical-landscape-and-build-plan.md)** —
  the full source‑cited research brief (how clinics deliver gait analysis today, by setting;
  commercial systems & pricing; accuracy/validation norms) **and** the cost‑minimized,
  commercial‑license‑safe build plan, BOM, reporting layer, roadmap, and validation design.
- **[`docs/02-phase1-quickstart.md`](docs/02-phase1-quickstart.md)** — how to run the Phase 1
  code (iPhone video → 2D pose → spatiotemporal report).

## Repo layout

```
src/gait_analysis/
  pose/rtmpose_runner.py        RTMPose 2D keypoints (accurate-mode 2D)  (gait-pose CLI)
  pose/mediapipe3d.py           BlazePose 3D world landmarks, 1 cam  (quick mode)
  biomech/blazepose_to_trc.py   MediaPipe 3D -> OpenSim .trc markers
  biomech/markerset.py          keypoint -> OpenSim marker map + IK weights
  biomech/opensim_setup.py      auto-generate OpenSim IK setup XML from the markerset
  biomech/opensim_ik.py         OpenSim IK -> all joint angles (validates markers; Phase-1b)
  analysis/spatiotemporal.py    Zeni-2008 gait events, cadence, symmetry  (gait-spatiotemporal CLI)
  analysis/kinematics.py        OpenSim .mot -> ROM, L/R symmetry, per-coordinate graphs
  analysis/signatures.py        clinical-signature flags (tightness/weakness/neuro/pain) from kinematics
  pipeline.py                   end-to-end orchestrator  (gait-pipeline CLI; --from-mot runs today)
  config.py                     COCO-17 layout + defaults
tests/                          offline synthetic tests (no video/network/OpenSim)  — 19 passing
setup/setup_macos.sh            Apple Silicon environment setup
docs/                           research brief, quickstart, quick-mode 3D, clinical signatures
```

## Clinical interpretation

The system is built to read **tightness, weakness, injury-risk and neuro/concussion/pain patterns** from
the joint-angle data — not just step length. See
**[`docs/04-clinical-signatures.md`](docs/04-clinical-signatures.md)** for the source-cited signature
library (coordinate, normal vs abnormal threshold, implicated structure, confirming clinical test), the
honest limits (deviation ≠ cause; speed confound; markerless plane reliability), and how
`analysis/signatures.py` encodes it.

## Planned stack (all commercial‑license‑safe)

iPhone capture → **RTMPose** (Apache‑2.0) 2D pose → **Pose2Sim** (BSD‑3) calibration +
triangulation → **OpenSim/Moco** (Apache‑2.0) kinematics + dynamics →
GRF‑from‑kinematics → **our own clinical reporting + normative‑comparison layer**.

> Deliberately avoids OpenPose ($25k/yr commercial license, excludes sports) and
> Ultralytics YOLO (AGPL) so the pipeline stays commercializable.

## Status

🛠️ **Phase 1 in progress.** Working today: iPhone video → RTMPose 2D → spatiotemporal
report (cadence, step/stride time, temporal symmetry) with passing synthetic tests.
Next: monocular 3D (quick mode) + Pose2Sim triangulation (accurate mode) → OpenSim IK →
joint angles. See [the build plan](docs/01-clinical-landscape-and-build-plan.md) for the
full roadmap (MVP → reporting → kinetics → validation → commercial‑ready).
