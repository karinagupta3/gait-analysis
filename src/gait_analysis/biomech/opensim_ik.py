"""OpenSim scaling + inverse kinematics (the stage that yields ALL joint angles).

SCAFFOLD / Phase-1b. This wraps OpenSim's ScaleTool and InverseKinematicsTool so a
.trc of markers becomes a .mot of model coordinates (pelvis_tilt/list/rotation +
tx/ty/tz, hip_flexion/adduction/rotation, knee_angle, ankle_angle, subtalar_angle,
mtp_angle, lumbar_*, arm_*, elbow_flex, pro_sup -- both sides). Those coordinates
are exactly what analysis/kinematics.py then graphs and what the clinical
"signature" layer interprets (docs/04).

NOT runnable until you install OpenSim and supply a model + setup files:
  * OpenSim Python package. On Apple Silicon a native osx-arm64 build may be
    unavailable; use an x86_64 (Rosetta) conda env (see setup/setup_macos.sh).
  * A full-body .osim model whose markers are NAMED to match the 33 BlazePose
    markers we write (see blazepose_to_trc.BLAZEPOSE_33). The LaiUhlrich2022 model
    (as used by OpenCap) is the target; it needs a marker set placed at those
    landmarks. Building/validating that marker set is the open task here.
  * ScaleTool setup XML and an IK tasks XML.

This module deliberately does NOT fabricate results. It documents the interface and
fails loudly if OpenSim or inputs are missing, so we never ship an untested number.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _require_opensim():
    try:
        import opensim  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "OpenSim Python is not installed (Phase-1b).\n"
            "Apple Silicon: create an x86_64 Rosetta conda env, e.g.\n"
            "  CONDA_SUBDIR=osx-64 conda create -n opensim python=3.11\n"
            "  conda activate opensim && conda install -c opensim-org opensim\n"
        ) from exc
    import opensim
    return opensim


def scale_model(model_path, marker_trc, scale_setup_xml, out_model, time_range=None):
    """Run OpenSim ScaleTool. Returns path to the scaled .osim."""
    osim = _require_opensim()
    for p in (model_path, marker_trc, scale_setup_xml):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    scale_tool = osim.ScaleTool(str(scale_setup_xml))
    # Caller's setup XML should reference model_path + marker_trc; we run it as-is.
    if not scale_tool.run():
        raise RuntimeError("OpenSim ScaleTool failed.")
    return Path(out_model)


def run_ik(model_path, marker_trc, ik_setup_xml, out_mot, time_range=None):
    """Run OpenSim InverseKinematicsTool -> coordinates .mot."""
    osim = _require_opensim()
    for p in (model_path, marker_trc, ik_setup_xml):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    model = osim.Model(str(model_path))
    ik = osim.InverseKinematicsTool(str(ik_setup_xml))
    ik.setModel(model)
    ik.setMarkerDataFileName(str(marker_trc))
    ik.setOutputMotionFileName(str(out_mot))
    if time_range is not None:
        ik.setStartTime(float(time_range[0]))
        ik.setEndTime(float(time_range[1]))
    if not ik.run():
        raise RuntimeError("OpenSim InverseKinematicsTool failed.")
    return Path(out_mot)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OpenSim scale + IK (Phase-1b scaffold)")
    ap.add_argument("--model", required=True, help="Full-body .osim (markers named to match TRC)")
    ap.add_argument("--trc", required=True, help="Marker .trc (from blazepose_to_trc)")
    ap.add_argument("--ik-setup", required=True, help="IK tasks setup XML")
    ap.add_argument("--out-mot", required=True)
    ap.add_argument("--scale-setup", default=None, help="Optional ScaleTool setup XML")
    ap.add_argument("--scaled-model-out", default=None)
    args = ap.parse_args(argv)

    model = args.model
    if args.scale_setup:
        model = scale_model(args.model, args.trc, args.scale_setup,
                            args.scaled_model_out or "scaled.osim")
    out = run_ik(model, args.trc, args.ik_setup, args.out_mot)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
