"""Print an OpenSim model's bodies, joints (with connected bodies), coordinates, markers.

Run this on a base .osim (on the Mac, where OpenSim is installed) so we use the model's
REAL names rather than guesses -- then marker_placement.py is reconciled to them.

    python -m gait_analysis.biomech.inspect_model --model LaiUhlrich2022.osim
"""

from __future__ import annotations

import argparse


def _require_opensim():
    try:
        import opensim  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("OpenSim not installed (use the conda 'gait' env).") from exc
    import opensim
    return opensim


def inspect(model_path: str) -> dict:
    osim = _require_opensim()
    model = osim.Model(str(model_path))
    model.initSystem()

    bodies = [model.getBodySet().get(i).getName() for i in range(model.getBodySet().getSize())]

    joints = []
    js = model.getJointSet()
    for i in range(js.getSize()):
        j = js.get(i)
        try:
            p = j.getParentFrame().findBaseFrame().getName()
            c = j.getChildFrame().findBaseFrame().getName()
        except Exception:
            p = c = "?"
        joints.append((j.getName(), p, c))

    coords = [model.getCoordinateSet().get(i).getName()
              for i in range(model.getCoordinateSet().getSize())]
    markers = [model.getMarkerSet().get(i).getName()
               for i in range(model.getMarkerSet().getSize())]
    return {"bodies": bodies, "joints": joints, "coordinates": coords, "markers": markers}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dump an OpenSim model's bodies/joints/coords/markers")
    ap.add_argument("--model", required=True)
    args = ap.parse_args(argv)
    info = inspect(args.model)
    print("BODIES:", ", ".join(info["bodies"]))
    print("\nJOINTS (name: parent -> child):")
    for name, p, c in info["joints"]:
        print(f"  {name}: {p} -> {c}")
    print("\nCOORDINATES:", ", ".join(info["coordinates"]))
    print("\nMARKERS:", ", ".join(info["markers"]) or "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
