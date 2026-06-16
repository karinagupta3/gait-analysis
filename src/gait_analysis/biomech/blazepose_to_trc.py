"""Convert MediaPipe BlazePose 3D world landmarks to an OpenSim .trc marker file.

The .trc is what OpenSim's Scale + Inverse Kinematics tools consume. We write the
33 BlazePose world landmarks (metres) as named virtual markers; the OpenSim model
used for IK must carry markers with the SAME names at the corresponding anatomical
locations (see biomech/opensim_ik.py and docs/03).

COORDINATE FRAMES (read this -- it is the #1 source of bad IK):
  MediaPipe world landmarks: +x image-right, +y image-down, +z toward camera.
  OpenSim convention:        +x anterior (forward), +y up, +z subject's right.
We apply a fixed remap suitable for a SAGITTAL-view capture (subject walking across
the frame). Orientation/handedness still depends on which way the subject faces, so
the result MUST be validated against a known pose before trusting frontal/transverse
angles. This is the documented limitation of single-camera quick mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..pose.mediapipe3d import BLAZEPOSE_33


def remap_axes(world: np.ndarray) -> np.ndarray:
    """MediaPipe world frame -> OpenSim-like frame for a sagittal capture.

    world: (..., 3) [mp_x, mp_y, mp_z].
    OpenSim: X_ant = mp_x (walking direction), Y_up = -mp_y, Z_right = mp_z.
    """
    x, y, z = world[..., 0], world[..., 1], world[..., 2]
    out = np.empty_like(world)
    out[..., 0] = x        # anterior (walking direction)
    out[..., 1] = -y       # up (MediaPipe y points down)
    out[..., 2] = z        # lateral (depth) -- weakest axis from one camera
    return out


def write_trc(
    path: str | Path,
    marker_names: list[str],
    positions: np.ndarray,   # (T, M, 3) in metres
    times: np.ndarray,       # (T,)
    units: str = "m",
):
    """Write a minimal, OpenSim-compatible .trc file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    T, M, _ = positions.shape
    if len(marker_names) != M:
        raise ValueError(f"{len(marker_names)} names vs {M} markers")
    rate = 1.0 / np.median(np.diff(times)) if T > 1 else 30.0

    header_cols = ["Frame#", "Time"]
    for name in marker_names:
        header_cols += [name, "", ""]
    axis_row = ["", ""]
    for i in range(1, M + 1):
        axis_row += [f"X{i}", f"Y{i}", f"Z{i}"]

    lines = []
    lines.append(f"PathFileType\t4\t(X/Y/Z)\t{path.name}")
    lines.append("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                 "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames")
    lines.append(f"{rate:.6f}\t{rate:.6f}\t{T}\t{M}\t{units}\t{rate:.6f}\t1\t{T}")
    lines.append("\t".join(header_cols))
    lines.append("\t".join(axis_row))

    for f in range(T):
        row = [str(f + 1), f"{times[f]:.6f}"]
        for m in range(M):
            x, y, z = positions[f, m]
            # OpenSim tolerates blanks for missing markers; emit blanks for NaN.
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                row += ["", "", ""]
            else:
                row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
        lines.append("\t".join(row))

    path.write_text("\n".join(lines) + "\n")
    return path


def _fill_gaps(world: np.ndarray) -> np.ndarray:
    """Trim no-detection lead-in/out and linearly interpolate interior gaps.

    OpenSim's TRC reader trims trailing empty columns, so rows that end in a missing
    marker lose a column and IK fails. Gap-filling (standard mocap practice) removes all
    NaNs: drop frames where <half the markers are present, then interpolate per marker/axis.
    """
    T, M, _ = world.shape
    present = np.isfinite(world).all(axis=2)            # (T, M)
    valid_frame = present.sum(axis=1) >= (M // 2)
    if not valid_frame.any():
        return world
    lo = int(np.argmax(valid_frame))
    hi = T - int(np.argmax(valid_frame[::-1]))
    out = world[lo:hi].copy()
    n = out.shape[0]
    idx = np.arange(n)
    for m in range(M):
        for a in range(3):
            col = out[:, m, a]
            ok = np.isfinite(col)
            if ok.sum() >= 2:
                col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
            elif ok.sum() == 1:
                col[~ok] = col[ok][0]
            else:
                col[:] = 0.0
            out[:, m, a] = col
    return out


def npz_to_trc(npz_path: str | Path, out_trc: str | Path, min_visibility: float = 0.3):
    """Load a mediapipe3d .npz and write a .trc with axis remap, low-conf masking, gap-fill."""
    data = np.load(npz_path)
    world = data["world_landmarks"].astype(float)   # (T,33,3)
    vis = data["visibility"].astype(float)           # (T,33)
    fps = float(data["fps"]) or 30.0

    world = remap_axes(world)
    world[vis < min_visibility] = np.nan             # blank low-confidence markers
    world = _fill_gaps(world)                         # trim + interpolate -> no NaNs for OpenSim
    times = np.arange(world.shape[0]) / fps
    return write_trc(out_trc, list(BLAZEPOSE_33), world.astype(np.float32), times)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MediaPipe 3D .npz -> OpenSim .trc")
    ap.add_argument("--npz", required=True, help="mediapipe3d output .npz")
    ap.add_argument("--out", required=True, help="Output .trc")
    ap.add_argument("--min-visibility", type=float, default=0.3)
    args = ap.parse_args(argv)
    path = npz_to_trc(args.npz, args.out, args.min_visibility)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
