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
              gait_speed_m_s: float | None = None) -> dict:
    """Full single-phone quick-mode pipeline. Requires mediapipe + OpenSim + a model."""
    video, model, outdir = Path(video), Path(model), Path(outdir)
    if not video.exists():
        raise FileNotFoundError(video)
    outdir.mkdir(parents=True, exist_ok=True)

    # Stage 1: MediaPipe 3D (lazy import; needs `pip install mediapipe`).
    from .pose import mediapipe3d
    print("[1/5] MediaPipe 3D world landmarks ...")
    npz = outdir / "world_landmarks.npz"
    import numpy as np
    np.savez_compressed(npz, **mediapipe3d.extract_world_landmarks(video))

    # Stage 2: landmarks -> OpenSim .trc.
    from .biomech import blazepose_to_trc
    print("[2/5] BlazePose -> OpenSim markers (.trc) ...")
    trc = blazepose_to_trc.npz_to_trc(npz, outdir / "markers.trc")

    # Stage 3: OpenSim scale + IK (needs OpenSim + a marked model).
    from .biomech import opensim_ik
    print("[3/5] OpenSim inverse kinematics ...")
    mot = opensim_ik.run_ik_from_trc(model, trc, outdir / "coordinates.mot")

    # Stages 4-5: report + signatures.
    print("[4/5] Kinematics report ...")
    print("[5/5] Clinical signature flags ...")
    result = report_from_mot(mot, gait_speed_m_s, plot_path=outdir / "joint_angles.png")
    result["mot"] = mot

    # Side-by-side viewer: video+markers (left) synced with the OpenSim model render (right).
    try:
        from .analysis import synced_viewer
        synced_viewer.build(video, npz, outdir / "synced", model=model, mot=mot)
        result["synced_viewer"] = str(outdir / "synced" / "viewer.html")
    except Exception as exc:
        print(f"[note] synced viewer skipped: {exc}")
    return result


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
