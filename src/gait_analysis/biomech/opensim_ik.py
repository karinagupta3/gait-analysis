"""OpenSim scaling + inverse kinematics: markers (.trc) -> joint angles (.mot).

This is the stage that yields ALL the OpenCap-style coordinates (pelvis tilt/list/
rotation + tx/ty/tz, hip flexion/adduction/rotation, knee, ankle, subtalar, mtp,
lumbar, arm, elbow, pro/sup -- both sides). Those drive analysis/kinematics.py and
analysis/signatures.py.

Requirements to RUN (not to import):
  * OpenSim Python package. On Apple Silicon a native osx-arm64 build may be
    unavailable; use an x86_64 (Rosetta) conda env (see setup/setup_macos.sh).
  * A full-body .osim whose markers are NAMED to match our active markers
    (biomech/markerset.active_markers()). Target: LaiUhlrich2022 (the OpenCap model)
    with a marker set placed at those landmarks -- BUILDING/VALIDATING that marker
    set is the remaining open task; until validated, treat outputs as provisional.

The module fails loudly if OpenSim or inputs are missing -- it never fabricates angles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .markerset import validate_against_trc_markers
from .opensim_setup import write_ik_tool_setup_xml, write_scale_setup_xml


def read_trc_marker_names(trc_path: str | Path) -> list[str]:
    """Read marker names from a .trc header (no OpenSim needed)."""
    lines = Path(trc_path).read_text().splitlines()
    if len(lines) < 4:
        raise ValueError(f"TRC too short: {trc_path}")
    # Row 4 (index 3): Frame#  Time  M1 '' '' M2 '' '' ...
    cols = lines[3].split("\t")
    names = [c.strip() for c in cols[2:] if c.strip()]
    return names


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


def run_ik_from_trc(
    model_path: str | Path,
    marker_trc: str | Path,
    out_mot: str | Path,
    time_range: tuple[float, float] | None = None,
    setup_out: str | Path | None = None,
):
    """Generate an IK setup from our marker weights and run OpenSim IK.

    Validates that the model's expected markers exist in the TRC first.
    """
    for p in (model_path, marker_trc):
        if not Path(p).exists():
            raise FileNotFoundError(p)

    missing = validate_against_trc_markers(read_trc_marker_names(marker_trc))
    if missing:
        raise ValueError(
            f"TRC is missing IK markers required by the model: {missing}. "
            "Check the pose->TRC mapping (biomech/markerset.py)."
        )

    setup_out = Path(setup_out) if setup_out else Path(out_mot).with_suffix(".ik_setup.xml")
    write_ik_tool_setup_xml(
        setup_out, model_file=str(model_path), marker_file=str(marker_trc),
        output_motion_file=str(out_mot), time_range=time_range,
    )

    osim = _require_opensim()
    ik = osim.InverseKinematicsTool(str(setup_out))
    if not ik.run():
        raise RuntimeError("OpenSim InverseKinematicsTool failed.")
    return Path(out_mot)


def run_scale(
    model_path: str | Path,
    static_trc: str | Path,
    out_model: str | Path,
    mass_kg: float = 70.0,
    height_m: float = 1.75,
    time_range: tuple[float, float] = (0.0, 1.0),
    setup_out: str | Path | None = None,
):
    """Generate a measurement-based ScaleTool setup and size the model to the subject.

    `static_trc` is a short standing/neutral-pose capture. Validates markers first.
    """
    for p in (model_path, static_trc):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    missing = validate_against_trc_markers(read_trc_marker_names(static_trc))
    if missing:
        raise ValueError(f"Static TRC missing IK markers: {missing}")

    setup_out = Path(setup_out) if setup_out else Path(out_model).with_suffix(".scale_setup.xml")
    write_scale_setup_xml(
        setup_out, model_file=str(model_path), static_trc=str(static_trc),
        output_model_file=str(out_model), mass_kg=mass_kg, height_m=height_m,
        time_range=time_range,
    )
    osim = _require_opensim()
    scale_tool = osim.ScaleTool(str(setup_out))
    if not scale_tool.run():
        raise RuntimeError("OpenSim ScaleTool failed.")
    return Path(out_model)


def main_scale(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OpenSim scaling: model + static .trc -> scaled .osim")
    ap.add_argument("--model", required=True, help="Marked full-body .osim")
    ap.add_argument("--static", required=True, help="Static-pose .trc")
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--mass", type=float, default=70.0, help="Subject mass (kg)")
    ap.add_argument("--height", type=float, default=1.75, help="Subject height (m)")
    args = ap.parse_args(argv)
    out = run_scale(args.model, args.static, args.out_model, args.mass, args.height)
    print(f"Wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OpenSim IK: .trc -> .mot (generates its own setup)")
    ap.add_argument("--model", required=True, help="Full-body .osim (markers named to match markerset)")
    ap.add_argument("--trc", required=True, help="Marker .trc (from blazepose_to_trc / pose2sim)")
    ap.add_argument("--out-mot", required=True)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    args = ap.parse_args(argv)

    tr = (args.start, args.end) if args.start is not None and args.end is not None else None
    out = run_ik_from_trc(args.model, args.trc, args.out_mot, time_range=tr)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
