# Quick Mode: single-phone 3D → all joint angles

Goal: the OpenCap-style experience — **record on one phone, get a 3D model + the full
joint-angle dataset** (pelvis tilt/list/rotation + tx/ty/tz, hip flexion/adduction/
rotation, knee, ankle, subtalar, mtp, lumbar extension/bending/rotation, arm flex/add/
rot, elbow flex, pro/sup — both sides). Those coordinates come out of **OpenSim inverse
kinematics**; we don't compute them by hand.

## Why this exact stack (commercial-safe)

| Stage | Tool | License | Why |
|---|---|---|---|
| 3D landmarks (1 cam) | **MediaPipe BlazePose GHUM** | Apache-2.0, commercial-OK | Metric 3D (metres), trained on Google data — **no SMPL / Human3.6M license trap** |
| Markers → OpenSim | our `blazepose_to_trc` | — | writes `.trc` |
| Scale + IK | **OpenSim** (LaiUhlrich2022) | Apache-2.0 | yields ALL model coordinates |
| Report + graphs | our `analysis/kinematics` | — | ROM, L/R symmetry, per-coordinate plots |

We avoid **SMPL** (non-commercial; Meshcapade licence) and **OpenPose/Ultralytics**.

## Pipeline

```
walk.mov ─► mediapipe3d ─► world_landmarks.npz ─► blazepose_to_trc ─► markers.trc
                                                                          │
                                          OpenSim Scale + IK  ◄───────────┘
                                                   │
                                            coordinates.mot ─► analysis.kinematics ─► report + plots
```

## Run it (what works today vs Phase-1b)

**1. Single-phone 3D landmarks** (runs now on your Mac; `pip install mediapipe`):
```bash
python -m gait_analysis.pose.mediapipe3d --video data/walk.mov \
    --out outputs/walk_mp3d.npz --overlay outputs/walk_mp3d_overlay.mp4
open outputs/walk_mp3d_overlay.mp4
```

**2. Markers for OpenSim** (runs now):
```bash
python -m gait_analysis.biomech.blazepose_to_trc --npz outputs/walk_mp3d.npz \
    --out outputs/walk.trc
```

**3. OpenSim Scale + IK → all joint angles** (Phase-1b — needs OpenSim installed):
```bash
python -m gait_analysis.biomech.opensim_ik --model models/LaiUhlrich2022_markers.osim \
    --trc outputs/walk.trc --ik-setup setups/ik_tasks.xml --out-mot outputs/walk_ik.mot
```

**4. Report + graphs of every coordinate** (runs now, given any `.mot`):
```bash
python -m gait_analysis.analysis.kinematics --mot outputs/walk_ik.mot \
    --out-plot outputs/walk_angles.png
```
Prints ROM + L/R symmetry per coordinate and writes a grid of curves (R vs L overlaid).

## Open engineering tasks for step 3 (honest status)

OpenSim IK needs a model whose **markers are named to match our 33 BlazePose markers**
at the right anatomical spots. Building/validating that marker set on LaiUhlrich2022 (the
way OpenCap does, but from a single-camera marker cloud) is the open task. Until it's
validated we do **not** print joint angles from real captures — the code fails loudly
rather than fabricating numbers.

## Honesty / accuracy

Single-camera depth (z) is the weakest axis, so **sagittal angles** (flexion/extension:
hip_flexion, knee_angle, ankle_angle) are usable, while **frontal/transverse** (adduction,
rotation, subtalar) are **low-confidence** (~8.5° typical single-cam error, worse out of
plane). The report flags these. For trustworthy frontal/transverse + clinical asymmetry,
use **accurate mode (2 phones)** — see [build plan §2.3b](01-clinical-landscape-and-build-plan.md).
