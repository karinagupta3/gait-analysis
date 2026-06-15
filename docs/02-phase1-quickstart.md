# Phase 1 Quickstart

Phase 1 goal (see [build plan §2.6](01-clinical-landscape-and-build-plan.md)): get
**spatiotemporal parameters + sagittal joint angles** from iPhone video, in **both**
capture modes, and sanity-check against tape-measured ground truth.

This scaffold currently implements the **2D pose → spatiotemporal** slice end-to-end.
The 3D lifting (monocular SMPL for quick mode; Pose2Sim triangulation for accurate
mode) and OpenSim IK are the next commits — they're stubbed with clear TODOs.

## What works today

```
iPhone .mov ──> gait-pose ──> keypoints .npz ──> gait-spatiotemporal ──> cadence/symmetry report
              (RTMPose 2D)                       (Zeni 2008 events)
```

## Setup (Apple Silicon)

```bash
bash setup/setup_macos.sh
source .venv/bin/activate
pytest -q            # synthetic gait-event tests, no video or network needed
```

## Run on a real clip

1. **Record** one iPhone clip of someone walking ~8–10 m **across** the frame
   (sagittal view), phone on a tripod, 1080p @ 60 fps. Save it to `data/walk.mov`.
2. **Extract 2D pose** (downloads RTMPose ONNX weights on first run):
   ```bash
   gait-pose --video data/walk.mov --out outputs/walk.npz --overlay outputs/walk_overlay.mp4
   ```
   Open `outputs/walk_overlay.mp4` to eyeball skeleton quality.
3. **Spatiotemporal report:**
   ```bash
   gait-spatiotemporal --keypoints outputs/walk.npz
   ```
   Example output:
   ```
   === Spatiotemporal gait report (2D, temporal only) ===
   Cadence         : 112.4 steps/min
   Mean step time  : 0.534 s
   Stride symmetry : L/R = 1.02  [OK]
   ```

## Honesty guardrails baked in

- **Temporal** parameters (cadence, step/stride time, symmetry) are valid from a
  single 2D sagittal view and are **scale-free** — reported now.
- **Spatial** parameters (step length in metres) need calibration/triangulation —
  **not** reported yet (only flagged), to avoid a false metric claim.
- **Frontal/transverse** kinematics and **kinetics** are reserved for accurate
  (multi-camera) mode + the GRF model; quick mode will flag them low-confidence.

## Next commits (Phase 1 continuation)

- [ ] `pose/monocular3d.py` — SMPL mesh (CameraHMR-class) → 3D joints (quick mode)
- [ ] `triangulation/pose2sim_runner.py` — multi-view → OpenSim-ready 3D (accurate mode)
- [ ] `biomech/opensim_ik.py` — scaling + inverse kinematics → joint angles
- [ ] `analysis/kinematics.py` — hip/knee/ankle ROM vs normative bands (Fukuchi 2018)
- [ ] `report/` — the clinical PDF/HTML report (Phase 2)
