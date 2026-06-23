"""End-to-end orchestration: video -> 3D -> OpenSim IK -> report -> signature flags.

Two entry points:
  * report_from_mot(): runs TODAY on any OpenSim .mot -- kinematics report + clinical
    signature flags. This is the analysis half and needs no video/MediaPipe/OpenSim.
  * run_quick(): the full single-phone (quick-mode) chain. Stages lazy-import their
    heavy deps and fail loudly with guidance; nothing is fabricated.

Quick-mode chain (see docs/03):
    video --MediaPipe3D--> world_landmarks.npz --blazepose_to_trc--> markers.trc
          --OpenSim scale+IK--> coordinates.mot --kinematics--> report --signatures--> flags
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import gait_cycle, kinematics, signatures


def report_from_mot(mot_path: str | Path, gait_speed_m_s: float | None = None,
                    plot_path: str | Path | None = None,
                    html_path: str | Path | None = None) -> dict:
    """Kinematics report + signature flags from an existing OpenSim .mot. Runs now."""
    time, coords, meta = kinematics.read_storage(mot_path)
    summary = kinematics.summarize(time, coords, meta)
    print(kinematics.format_report(summary))
    if plot_path:
        print(f"\nWrote plot: {kinematics.plot_coordinates(time, coords, plot_path)}")

    # Phase-windowed features (swing/stance) -> clinically-correct, less noisy flags.
    phase = gait_cycle.compute_phase_features(time, coords)
    ctx = signatures.Context(gait_speed_m_s=gait_speed_m_s, phase=phase)
    from .analysis import tasks
    task, findings, _metrics = tasks.route(time, coords, summary, ctx)
    print(f"\nDetected task: {task}")
    print("\n" + signatures.format_findings(findings, ctx))

    if html_path:
        from .analysis import report
        out = report.build_html_report(mot_path, html_path, gait_speed_m_s=gait_speed_m_s)
        print(f"\nWrote HTML report: {out}")
    return {"summary": summary, "findings": findings, "phase": phase}


def run_quick(video: str | Path, model: str | Path, outdir: str | Path,
              gait_speed_m_s: float | None = None,
              height_m: float | None = None, mass_kg: float | None = None) -> dict:
    """Full single-phone quick-mode pipeline (OpenCap-style marker augmentation).

    video -> MediaPipe BlazePose-33 world landmarks -> the Stanford marker-augmenter
    LSTM (43 anatomical "_study" markers) -> OpenSim scale + IK on the LaiUhlrich2022
    model. Driving IK from the augmented markers (not raw keypoints) is what keeps the
    trunk upright instead of hunched. `model` is kept for signature compatibility but
    is unused -- the bundled LaiUhlrich2022 model is used. Requires mediapipe + OpenSim.

    height_m / mass_kg: subject anthropometry for scaling + augmentation. If height is
    None it is estimated from the pose (less reliable -- prefer the UI-entered value).
    """
    video, outdir = Path(video), Path(outdir)
    if not video.exists():
        raise FileNotFoundError(video)
    outdir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    # Stage 1: MediaPipe 3D (lazy import; needs `pip install mediapipe`).
    from .pose import mediapipe3d
    print("[1/5] MediaPipe 3D world landmarks ...")
    npz = outdir / "world_landmarks.npz"
    d = mediapipe3d.extract_world_landmarks(video)
    np.savez_compressed(npz, **d)

    # Stage 2: landmarks -> OpenSim frame, then marker augmentation -> 43 anatomical
    # markers written to a .trc the LaiUhlrich2022 model is built around.
    from .biomech import blazepose_to_trc, marker_augmentation as MA
    print("[2/5] Marker augmentation (Stanford LSTM -> anatomical markers) ...")
    vis = d["visibility"].astype(float)
    world_raw = blazepose_to_trc.remap_axes(d["world_landmarks"].astype(float))
    masked = world_raw.copy()
    masked[vis < 0.3] = np.nan
    # Determine facing from the RAW (pre-gap-fill) landmarks + visibility, so the
    # estimate is confidence-weighted (hips fall back to shoulders on hard clips).
    basis = MA.facing_basis(masked, vis)
    conf = float(np.median(vis[:, [11, 12, 23, 24, 25, 26, 27, 28]]))   # shoulders/hips/knees/ankles
    if basis is None:
        print("[note] low-confidence pose: facing undetermined; skeleton may be unreliable")
    world = blazepose_to_trc._fill_gaps(masked)
    fps = float(d["fps"]) or 30.0
    world = MA.smooth_world(world, fps)                 # Butterworth low-pass (de-jitter)
    height_m = float(height_m) if height_m else MA.estimate_height_m(world)
    mass_kg = float(mass_kg) if mass_kg else 70.0
    print(f"      subject: height={height_m:.2f} m, mass={mass_kg:.0f} kg, "
          f"tracking confidence={conf:.2f}")
    names, pos = MA.augment(world, height_m, mass_kg, basis=basis)
    times = np.arange(world.shape[0]) / fps
    trc = blazepose_to_trc.write_trc(
        outdir / "markers.trc", names, pos.astype(np.float32), times)

    # Stage 3: OpenSim scale + IK on the augmented markers (bundled OpenCap setups).
    from .biomech import augmented_ik
    print("[3/5] OpenSim scaling + inverse kinematics ...")
    mot, scaled_model = augmented_ik.scale_and_ik(trc, outdir, height_m, mass_kg)
    import shutil
    shutil.copyfile(mot, outdir / "coordinates.mot")   # tier-2 contract name
    mot = outdir / "coordinates.mot"

    # Stages 4-5: report + signatures.
    print("[4/5] Kinematics report ...")
    print("[5/5] Clinical signature flags ...")
    result = report_from_mot(mot, gait_speed_m_s, plot_path=outdir / "joint_angles.png")
    result["mot"] = mot
    result["scaled_model"] = str(scaled_model)
    result["tracking_confidence"] = round(conf, 2)

    # Side-by-side viewer: video+markers (left) synced with the SCALED OpenSim model.
    try:
        from .analysis import synced_viewer
        synced_viewer.build(video, npz, outdir / "synced", model=scaled_model, mot=mot)
        result["synced_viewer"] = str(outdir / "synced" / "viewer.html")
    except Exception as exc:
        print(f"[note] synced viewer skipped: {exc}")
    return result


def run_screening(video: str | Path, outdir: str | Path, subject: str = "",
                  task: str = "gait", height_cm=None, weight_kg=None) -> dict:
    """Single-phone 2D SAGITTAL screening: video -> pose -> metrics -> report.

    task = "gait" (cadence/strides/per-stride peaks), "squat", or "sit_to_stand"
    (rep counting, depth/timing). No depth, no scaling, no OpenSim — just a
    side-view video. Writes a self-contained HTML screening report.
    """
    import numpy as np
    from .pose import mediapipe3d

    video, outdir = Path(video), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] MediaPipe pose (task={task}) ...")
    d = mediapipe3d.extract_world_landmarks(video)
    np.savez_compressed(outdir / "pose.npz", **d)
    args = (d["image_landmarks"], d["visibility"], int(d["width"]), int(d["height"]), float(d["fps"]))
    report_out = outdir / "screening_report.html"
    if task == "tug":
        from .analysis import tug_2d
        print("[2/3] TUG metrics (total time, phases, gait speed) ...")
        metrics = tug_2d.compute_tug_metrics(*args)
        print("[3/3] TUG report ...")
        report_path = tug_2d.build_tug_report(metrics, report_out, subject=subject)
    elif task in ("squat", "sit_to_stand"):
        from .analysis import movement_2d, movement_report
        print(f"[2/3] {task} metrics (reps, depth/timing, symmetry) ...")
        metrics = movement_2d.compute_movement_metrics(
            *args, task, height_cm=height_cm, weight_kg=weight_kg)
        print("[3/3] Movement report ...")
        report_path = movement_report.build_movement_report(metrics, report_out, subject=subject)
    else:
        from .analysis import gait_metrics_2d, screening_report
        print("[2/3] Gait metrics (cadence, strides, per-stride peaks, symmetry) ...")
        metrics = gait_metrics_2d.compute_gait_metrics(*args)
        print("[3/3] Screening report ...")
        report_path = screening_report.build_screening_report(metrics, report_out, subject=subject)

    # Per-frame signals for the interactive grapher.
    try:
        from .analysis import series_export
        series_export.write_series(
            series_export.from_screening_metrics(task, metrics), outdir / "series.json")
    except Exception as exc:
        print(f"[note] series.json skipped: {exc}")

    # Synced viewer: video with 2D pose overlay (left) + 3D world-landmark skeleton (right).
    viewer_path = None
    try:
        from .analysis import synced_viewer
        print("[+] Building synced viewer ...")
        synced_viewer.build(video, outdir / "pose.npz", outdir / "synced")
        viewer_path = str(outdir / "synced" / "viewer.html")
    except Exception as exc:
        print(f"[note] synced viewer skipped: {exc}")

    return {"mode": "screening", "task": task, "metrics": metrics,
            "report": str(report_path), "viewer": viewer_path}


def run_accurate(project_dir: str | Path, gait_speed_m_s: float | None = None) -> dict:
    """Track A: 2-phone accurate mode via Pose2Sim -> our report. Needs Pose2Sim + OpenSim."""
    from .biomech import pose2sim_runner
    print("[1/2] Pose2Sim (calibration -> triangulation -> OpenSim IK) ...")
    mot = pose2sim_runner.run(project_dir)
    print("[2/2] Kinematics report + signature flags ...")
    result = report_from_mot(mot, gait_speed_m_s)
    result["mot"] = mot
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gait pipeline: report from .mot, quick-mode, or accurate-mode.")
    ap.add_argument("--from-mot", help="Run analysis only on an existing OpenSim .mot")
    ap.add_argument("--video", help="Quick-mode (1 phone): input video")
    ap.add_argument("--model", help="Quick-mode: full-body .osim with matching markers")
    ap.add_argument("--outdir", default="outputs", help="Quick-mode: output directory")
    ap.add_argument("--accurate", help="Accurate-mode (2 phones): Pose2Sim project directory")
    ap.add_argument("--speed", type=float, default=None, help="Gait speed (m/s) for context")
    ap.add_argument("--plot", default=None, help="Optional joint-angle PNG (analysis-only mode)")
    ap.add_argument("--html", default=None, help="Optional self-contained HTML report path")
    args = ap.parse_args(argv)

    if args.from_mot:
        report_from_mot(args.from_mot, args.speed, plot_path=args.plot, html_path=args.html)
        return 0
    if args.accurate:
        run_accurate(args.accurate, args.speed)
        return 0
    if args.video and args.model:
        run_quick(args.video, args.model, args.outdir, args.speed)
        return 0
    ap.error("Provide --from-mot <file>, --accurate <project>, or both --video and --model.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
