"""Track A: drive Pose2Sim (RTMPose -> triangulation -> OpenSim IK) for accurate mode.

Pose2Sim (BSD-3) already bundles an OpenSim model whose markers match its HALPE_26
keypoints, plus scale + IK setups, and runs OpenSim scaling/IK for you. We delegate the
heavy biomechanics to it and keep our analysis/report/signatures layer on top.

This module scaffolds a Pose2Sim project (folders + a starter Config.toml tuned for two
iPhones) and drives the pipeline steps. The Config.toml is version-sensitive -- start
from Pose2Sim's bundled demo config and let this fill in the iPhone-specific fields.
Lazy-imports Pose2Sim; fails loudly if absent. Untestable in the build sandbox (no
Pose2Sim/OpenSim) beyond config/scaffold generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Starter config tuned for 2 phones + RTMPose/HALPE_26. Minimal subset; the user should
# reconcile it with their installed Pose2Sim's demo Config.toml (fields drift by version).
STARTER_CONFIG = """\
[project]
multi_person = false
participant_height = 1.75
participant_mass = 70.0
frame_rate = 'auto'

[pose]
pose_framework = 'rtmlib'        # RTMPose backend (Apache-2.0; no OpenPose)
pose_model = 'HALPE_26'          # has feet -> better ankle/foot for gait
mode = 'balanced'
det_frequency = 1
display_detection = false        # headless server -> never pop a GUI window

[synchronization]
synchronization_type = 'sound'   # clap sync; or set up OpenCap-app simultaneous capture

[calibration]
calibration_type = 'calculate'   # compute from a printed checkerboard (vs 'convert' an existing file)
   [calibration.calculate]
   save_debug_images = false      # headless server -> no debug image popups
      [calibration.calculate.intrinsics]
      overwrite_intrinsics = false
      intrinsics_extension = 'jpg' # we extract jpg frames from the calibration clip
      extract_every_N_sec = 1
      intrinsics_corners_nb = [4, 7]   # INNER corners (a 5x8-square board) -> see capture guide
      intrinsics_square_size = 25.0    # mm
      show_detection_intrinsics = false
      [calibration.calculate.extrinsics]
      calculate_extrinsics = true
      extrinsics_method = 'board'  # automated checkerboard detection (no manual point clicking)
      extrinsics_extension = 'png' # one shared-board frame per camera (camN_ext.png)
      show_reprojection_error = false
      moving_cameras = false
         [calibration.calculate.extrinsics.board]
         board_position = 'horizontal'   # board lies flat on the floor at the walk start
         extrinsics_corners_nb = [4, 7]
         extrinsics_square_size = 25.0   # mm

[triangulation]
reproj_error_threshold_triangulation = 15
likelihood_threshold_triangulation = 0.3
interpolation = 'cubic'

[filtering]
type = 'butterworth'
display_figures = false          # headless server -> never pop a GUI window
[filtering.butterworth]
order = 4
cut_off_frequency = 6

[kinematics]
use_augmentation = true          # OpenCap-style marker augmenter (improves IK)
right_left_symmetry = true
"""

# Pose2Sim project subfolders (per its expected layout).
SUBDIRS = ["calibration", "videos", "pose", "pose-3d", "kinematics"]


def prepare_project(project_dir: str | Path) -> Path:
    """Create the Pose2Sim folder layout + a starter Config.toml. Runs offline."""
    project_dir = Path(project_dir)
    for sub in SUBDIRS:
        (project_dir / sub).mkdir(parents=True, exist_ok=True)
    cfg = project_dir / "Config.toml"
    if not cfg.exists():
        cfg.write_text(STARTER_CONFIG)
    return project_dir


def _require_pose2sim():
    try:
        from Pose2Sim import Pose2Sim  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "Pose2Sim is not installed. Run:  pip install pose2sim\n"
            "(BSD-3; brings RTMPose + OpenSim model/setup for accurate mode)"
        ) from exc
    from Pose2Sim import Pose2Sim
    return Pose2Sim


def run(project_dir: str | Path) -> Path:
    """Run the Pose2Sim accurate-mode steps; return the OpenSim IK .mot.

    Expects a prepared project with calibration video(s) + the trial videos in place.

    Invocation matches what Pose2Sim actually expects of THIS installed version (proven
    on its bundled multi-camera demo): each stage reads Config.toml from the current
    working directory, so we chdir into the project and call the stages with no args
    (passing a path is version-fragile). We also force the 'fork' multiprocessing start
    method -- Pose2Sim pools break under macOS 'spawn'; Linux (the worker) already forks.
    synchronization + personAssociation are best-effort (they're for clap-syncing
    independently-started clips / multi-person and shouldn't abort an otherwise-good run).
    """
    import os

    project_dir = Path(project_dir).resolve()
    if not (project_dir / "Config.toml").exists():
        raise FileNotFoundError(f"No Config.toml in {project_dir}; call prepare_project first.")

    # Headless worker (no display): Pose2Sim is configured to SHOW figures/detections,
    # which pops GUI windows and crashes with "TclError: no display". Two guards:
    # (1) force the Agg matplotlib backend, (2) flip every display_* flag in Config.toml
    # to false (the real fix -- a backend alone won't stop an explicit plt.show()).
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    import re as _re
    _cfg = project_dir / "Config.toml"
    _txt = _cfg.read_text()
    for _key in ("display_detection", "display_figures", "show_realtime_results",
                 "show_plots", "save_vid"):
        _txt = _re.sub(rf"({_key}\s*=\s*)[Tt]rue", r"\1false", _txt)
    _cfg.write_text(_txt)

    try:
        import multiprocessing as _mp
        _mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        pass

    p2s = _require_pose2sim()
    cwd = os.getcwd()

    def _stage(name, fn, optional=False):
        # Log each stage (so the worker logs show progress) and treat Pose2Sim's
        # sys.exit() as a normal failure -- several stages call sys.exit on a bad
        # config/data path, which would otherwise silently kill the worker (SystemExit
        # is NOT an Exception, so the worker's `except Exception` never sees it).
        print(f"[pose2sim] {name} ...", flush=True)
        try:
            fn()
            print(f"[pose2sim] {name} done", flush=True)
        except BaseException as exc:               # noqa: BLE001 (incl. SystemExit)
            if optional:
                print(f"[pose2sim] {name} skipped ({type(exc).__name__}: {exc})", flush=True)
                return
            if isinstance(exc, SystemExit):
                raise RuntimeError(f"Pose2Sim {name} exited: {exc}") from exc
            raise

    try:
        os.chdir(project_dir)
        _stage("calibration", p2s.calibration)
        _stage("poseEstimation", p2s.poseEstimation)
        _stage("synchronization", p2s.synchronization, optional=True)
        _stage("personAssociation", p2s.personAssociation, optional=True)
        _stage("triangulation", p2s.triangulation)
        # NOTE: Pose2Sim's filtering step (butterworth smoothing of the 3D) kills the
        # worker process uncatchably on the headless server (clean os._exit mid-run,
        # even with display off), so we SKIP it. It's optional smoothing -- marker
        # augmentation falls back to the raw (unfiltered) triangulated .trc, and our
        # own pipeline smooths separately. Revisit if smoothing quality matters.
        _stage("markerAugmentation", p2s.markerAugmentation)
        _stage("kinematics", p2s.kinematics)        # OpenSim scaling + IK
    finally:
        os.chdir(cwd)

    mots = sorted((project_dir / "kinematics").glob("*.mot"))
    if not mots:
        raise RuntimeError("Pose2Sim finished but no .mot found in kinematics/.")
    return mots[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pose2Sim accurate-mode driver")
    ap.add_argument("--project", required=True)
    ap.add_argument("--prepare-only", action="store_true",
                    help="Just scaffold folders + Config.toml, then stop")
    args = ap.parse_args(argv)

    prepare_project(args.project)
    if args.prepare_only:
        print(f"Prepared project at {args.project}. Add calibration + trial videos, then re-run.")
        return 0
    out = run(args.project)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
