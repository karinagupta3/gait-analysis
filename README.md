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

## Planned stack (all commercial‑license‑safe)

iPhone capture → **RTMPose** (Apache‑2.0) 2D pose → **Pose2Sim** (BSD‑3) calibration +
triangulation → **OpenSim/Moco** (Apache‑2.0) kinematics + dynamics →
GRF‑from‑kinematics → **our own clinical reporting + normative‑comparison layer**.

> Deliberately avoids OpenPose ($25k/yr commercial license, excludes sports) and
> Ultralytics YOLO (AGPL) so the pipeline stays commercializable.

## Status

📋 Planning complete — see the docs. Implementation phases (MVP → reporting → kinetics →
validation → commercial‑ready) are described in the build plan.
