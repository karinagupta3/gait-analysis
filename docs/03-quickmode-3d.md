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

**3. OpenSim IK → all joint angles** (Phase-1b — needs OpenSim installed). The IK setup
XML is **auto-generated** from `biomech/markerset.py` (no hand-written XML), and the runner
**validates** that the model's markers exist in the .trc before running:
```bash
gait-ik --model models/LaiUhlrich2022_markers.osim --trc outputs/walk.trc --out-mot outputs/walk_ik.mot
```

**4. Report + signature flags from every coordinate** (runs now, given any `.mot`):
```bash
gait-pipeline --from-mot outputs/walk_ik.mot --speed 1.2 --plot outputs/walk_angles.png
```
Prints ROM + L/R symmetry per coordinate, writes a grid of curves (R vs L overlaid), AND
emits the clinical signature flags (see [docs/04](04-clinical-signatures.md)).

**Whole chain in one call** (once MediaPipe + OpenSim + a marked model are installed):
```bash
gait-pipeline --video data/walk.mov --model models/LaiUhlrich2022_markers.osim --outdir outputs --speed 1.2
```

## Open engineering tasks for step 3 (honest status)

OpenSim IK needs a model whose **markers are named to match our 33 BlazePose markers**
at the right anatomical spots. Building/validating that marker set on LaiUhlrich2022 (the
way OpenCap does, but from a single-camera marker cloud) is the open task. Until it's
validated we do **not** print joint angles from real captures — the code fails loudly
rather than fabricating numbers.

## Honesty / accuracy

Single-camera depth (z) is the weakest axis. With a **naive** landmark-to-IK approach
(raw MediaPipe → OpenSim) expect ~8.5° MAE and unreliable frontal/transverse angles.
**But the ceiling is much higher than that:** **OpenCap Monocular** (Gilon, Miller &
Uhlrich, 2026, [arXiv 2603.24733](https://arxiv.org/abs/2603.24733)) reaches **4.8° MAE
rotational / 3.4 cm pelvis from a single static smartphone video** — matching 2-camera
OpenCap — by refining a monocular pose estimate (**WHAM**) via optimization against a
biomechanically-constrained skeletal model, then estimating kinetics with physics + ML.

So our quick mode should target ~4.8°, not 8.5°, by adopting that *refine-against-the-
biomechanical-model* idea rather than feeding raw landmarks straight to IK.

⚠️ **Commercial-license caveat:** WHAM (and most monocular 3D-mesh models) are **SMPL**-
based, and SMPL is **non-commercial** (Meshcapade licence). For a commercial pivot we need
a non-SMPL monocular backbone or an SMPL licence — the same landmine as OpenPose. MediaPipe
BlazePose (Apache-2.0) stays clean but is less accurate; this is the trade-off to resolve.

The report flags frontal/transverse coordinates as low-confidence regardless. For the most
trustworthy frontal/transverse + clinical asymmetry, use **accurate mode (2 phones)** —
see [build plan §2.3b](01-clinical-landscape-and-build-plan.md).
