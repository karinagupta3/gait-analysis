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
    "knee_r": ["knee_r", "walking_knee_r"],
    "knee_l": ["knee_l", "walking_knee_l"],
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


def _joint_location_in_body(model, osim, joint_name: str, body_name: str):
    """Best-effort: return the joint's offset-frame translation expressed on `body`.

    Returns an osim.Vec3, or None if it can't be resolved (caller falls back).
    """
    try:
        joint = model.getJointSet().get(joint_name)
    except Exception:
        return None
    # A Joint has two PhysicalOffsetFrames (parent/child); pick the one based on `body`.
    for getter in ("getParentFrame", "getChildFrame"):
        try:
            frame = getattr(joint, getter)()
            base = frame.findBaseFrame().getName()
            if base == body_name:
                return frame.get_translation()
        except Exception:
            continue
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
    model.initSystem()
    bodyset = model.getBodySet()

    added, fellback = [], []
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
                loc = _joint_location_in_body(model, osim, jname, p.body)
                if loc is not None:
                    break
        if loc is None:
            loc = osim.Vec3(*p.offset)
            fellback.append(p.marker)
        else:
            loc = osim.Vec3(loc.get(0) + p.offset[0],
                            loc.get(1) + p.offset[1],
                            loc.get(2) + p.offset[2])

        marker = osim.Marker(p.marker, body, loc)
        model.addMarker(marker)
        added.append(p.marker)

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
