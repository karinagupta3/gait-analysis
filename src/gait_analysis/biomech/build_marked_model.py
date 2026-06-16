"""Inject our named markers into a base LaiUhlrich2022 .osim (Track B, Phase-1b).

Reads a base full-body OpenSim model and writes a copy with virtual markers named like
our keypoints, placed at joint centres per marker_placement.PLACEMENTS. The resulting
model is what our .trc (from blazepose_to_trc) drives through OpenSim IK.

Needs OpenSim installed + a base model you obtain from an open source (OpenCap
opencap-core, SimTK, or the Pose2Sim bundle). UNTESTED against a live model in CI --
the marker-placement spec is iterated/validated against Track A (docs/05). Fails loudly
rather than guessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .marker_placement import PLACEMENTS, validate

# Joint names vary between LaiUhlrich/Rajagopal variants; try aliases before falling back.
JOINT_ALIASES = {
    # LaiUhlrich2022 names the tibiofemoral joint "walker_knee_*"; older Rajagopal
    # variants use "walking_knee_*". Try both plus the plain name.
    "knee_r": ["knee_r", "walker_knee_r", "walking_knee_r"],
    "knee_l": ["knee_l", "walker_knee_l", "walking_knee_l"],
    "acromial_r": ["acromial_r", "shoulder_r", "GlenoHumeral_r"],
    "acromial_l": ["acromial_l", "shoulder_l", "GlenoHumeral_l"],
}


def _require_opensim():
    try:
        import opensim  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "OpenSim Python is not installed. Create an x86_64 Rosetta conda env on "
            "Apple Silicon (see setup/setup_macos.sh)."
        ) from exc
    import opensim
    return opensim


def _joint_location_in_body(model, osim, state, joint_name: str, target_body):
    """Return the joint centre expressed in `target_body`'s frame, or None.

    The joint centre is where the parent/child frames coincide at the default pose, so we
    take the joint's child-frame origin and express it in the target body via OpenSim's
    frame math. (The earlier approach read a PhysicalOffsetFrame's raw translation, which
    is 0 whenever the offset lives on the opposite frame -- that put every marker at the
    body origin.) `state` must already be realized to Position.
    """
    try:
        joint = model.getJointSet().get(joint_name)
    except Exception:
        return None
    try:
        child = joint.getChildFrame()
        return child.findStationLocationInAnotherFrame(state, osim.Vec3(0, 0, 0), target_body)
    except Exception:
        return None


def build(base_model_path: str | Path, out_model_path: str | Path) -> Path:
    base_model_path, out_model_path = Path(base_model_path), Path(out_model_path)
    if not base_model_path.exists():
        raise FileNotFoundError(base_model_path)

    spec = validate()
    if spec["missing"] or spec["extra"]:
        raise ValueError(f"marker_placement spec inconsistent with markerset: {spec}")

    osim = _require_opensim()
    model = osim.Model(str(base_model_path))
    state = model.initSystem()
    model.realizePosition(state)
    bodyset = model.getBodySet()

    # Pass 1: resolve every marker location FIRST, against the clean realized state.
    # (addMarker() mutates the model and invalidates `state`, so we must not add markers
    # while still resolving joint centres -- doing so left every marker after the first
    # at the body origin.)
    resolved = []  # (marker_name, body, osim.Vec3, fell_back)
    for p in PLACEMENTS:
        try:
            body = bodyset.get(p.body)
        except Exception as exc:
            raise ValueError(
                f"Body '{p.body}' for marker '{p.marker}' not in model. "
                f"Check body names match this .osim."
            ) from exc

        loc = None
        if p.at_joint:
            for jname in JOINT_ALIASES.get(p.at_joint, [p.at_joint]):
                loc = _joint_location_in_body(model, osim, state, jname, body)
                if loc is not None:
                    break
        if loc is None:
            resolved.append((p.marker, body, osim.Vec3(*p.offset), True))
        else:
            resolved.append((p.marker, body, osim.Vec3(
                loc.get(0) + p.offset[0],
                loc.get(1) + p.offset[1],
                loc.get(2) + p.offset[2]), False))

    # Pass 2: now add the markers (mutates the model).
    added, fellback = [], []
    for marker_name, body, loc, fell_back in resolved:
        model.addMarker(osim.Marker(marker_name, body, loc))
        added.append(marker_name)
        if fell_back:
            fellback.append(marker_name)

    model.finalizeConnections()
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.printToXML(str(out_model_path))

    print(f"Added {len(added)} markers -> {out_model_path}")
    if fellback:
        print(f"  (placed at body origin + offset, joint centre unresolved: {fellback})")
    return out_model_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inject named markers into a base LaiUhlrich .osim")
    ap.add_argument("--base", required=True, help="Base full-body .osim")
    ap.add_argument("--out", required=True, help="Output marked .osim")
    args = ap.parse_args(argv)
    build(args.base, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
