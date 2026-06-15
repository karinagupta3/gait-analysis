"""Kinematics reporting from OpenSim inverse-kinematics output (.mot / .sto).

This is the layer that turns a solved OpenSim motion into the OpenCap-style
"lots of data" report: every model coordinate (pelvis tilt/list/rotation + tx/ty/tz,
hip flexion/adduction/rotation, knee, ankle, subtalar, mtp, lumbar
extension/bending/rotation, arm flex/add/rot, elbow flex, pro/sup) -- both sides --
with range-of-motion, left/right symmetry, and per-coordinate graphs.

It parses the OpenSim STO/MOT storage format directly (no OpenSim install needed
to read results), so it is fully testable offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Coordinate display groups for the LaiUhlrich2022 / Rajagopal full-body model.
# (These are the OpenSim coordinate names IK solves for -- the same set OpenCap reports.)
PELVIS = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
]
LOWER_LIMB = [
    "hip_flexion", "hip_adduction", "hip_rotation",
    "knee_angle", "ankle_angle", "subtalar_angle", "mtp_angle",
]
TRUNK = ["lumbar_extension", "lumbar_bending", "lumbar_rotation"]
UPPER_LIMB = ["arm_flex", "arm_add", "arm_rot", "elbow_flex", "pro_sup"]

# Bilateral bases get _r/_l suffixes; PELVIS/TRUNK are midline (single column).
BILATERAL = LOWER_LIMB + UPPER_LIMB
MIDLINE = PELVIS + TRUNK


def read_storage(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Parse an OpenSim .mot/.sto file.

    Returns (time, {column_name: values}, meta). meta includes 'inDegrees'.
    Format: free-text header lines, a line equal to 'endheader', then a
    tab-separated column-name row, then numeric rows.
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    meta: dict[str, str] = {}
    header_end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower() == "endheader":
            header_end = i
            break
        if "=" in stripped:
            k, _, v = stripped.partition("=")
            meta[k.strip()] = v.strip()
    if header_end is None:
        raise ValueError(f"No 'endheader' found in {path}")

    col_line = lines[header_end + 1]
    names = col_line.split("\t")
    names = [n.strip() for n in names if n.strip() != ""]

    data_rows = []
    for line in lines[header_end + 2:]:
        if not line.strip():
            continue
        data_rows.append([float(x) for x in line.split("\t") if x.strip() != ""])
    arr = np.asarray(data_rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError(
            f"Column/data mismatch in {path}: {len(names)} names vs {arr.shape} data"
        )

    time = arr[:, 0]
    coords = {name: arr[:, j] for j, name in enumerate(names) if name.lower() != "time"}
    meta["inDegrees"] = meta.get("inDegrees", "yes")
    return time, coords, meta


def compute_rom(coords: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Range of motion (min/max/range/mean) per coordinate."""
    out = {}
    for name, v in coords.items():
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out[name] = {
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "range": float(np.max(v) - np.min(v)),
            "mean": float(np.mean(v)),
        }
    return out


def bilateral_pairs(coords: dict[str, np.ndarray]) -> list[tuple[str, str, str]]:
    """Find (base, right_col, left_col) triples present in the data."""
    pairs = []
    for base in BILATERAL:
        r, l = f"{base}_r", f"{base}_l"
        if r in coords and l in coords:
            pairs.append((base, r, l))
    return pairs


def symmetry(coords: dict[str, np.ndarray]) -> dict[str, float]:
    """ROM-based left/right symmetry ratio (L_range / R_range; 1.0 = symmetric).

    See docs/01 s2.4 for the index definitions. ROM symmetry is robust to the
    sign conventions that differ between left and right OpenSim coordinates.
    """
    rom = compute_rom(coords)
    out = {}
    for base, r, l in bilateral_pairs(coords):
        rr, lr = rom[r]["range"], rom[l]["range"]
        if rr > 1e-9:
            out[base] = lr / rr
    return out


def plot_coordinates(time, coords, out_png: str | Path, max_cols: int = 4):
    """Grid of per-coordinate curves; left/right overlaid where bilateral."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = {base: (r, l) for base, r, l in bilateral_pairs(coords)}
    midline = [c for c in MIDLINE if c in coords]
    panels = list(pairs.keys()) + midline
    if not panels:
        panels = sorted(coords.keys())

    n = len(panels)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), squeeze=False)

    for idx, name in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        if name in pairs:
            r, l = pairs[name]
            ax.plot(time, coords[r], label="R", color="tab:blue")
            ax.plot(time, coords[l], label="L", color="tab:red")
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.plot(time, coords[name], color="tab:green")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("time (s)", fontsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def summarize(time, coords, meta) -> dict:
    rom = compute_rom(coords)
    sym = symmetry(coords)
    return {
        "n_frames": int(len(time)),
        "duration_s": float(time[-1] - time[0]) if len(time) > 1 else 0.0,
        "in_degrees": str(meta.get("inDegrees", "yes")).lower() == "yes",
        "n_coordinates": len(coords),
        "rom": rom,
        "symmetry_LR": sym,
    }


def format_report(summary: dict) -> str:
    lines = ["=== Kinematics report (OpenSim IK) ==="]
    units = "deg" if summary["in_degrees"] else "rad"
    lines.append(
        f"Frames: {summary['n_frames']}  duration: {summary['duration_s']:.2f}s  "
        f"coordinates: {summary['n_coordinates']}  (angles in {units})"
    )
    lines.append("")
    lines.append(f"{'coordinate':<22}{'min':>9}{'max':>9}{'range':>9}")
    for name in sorted(summary["rom"]):
        r = summary["rom"][name]
        lines.append(f"{name:<22}{r['min']:>9.1f}{r['max']:>9.1f}{r['range']:>9.1f}")
    if summary["symmetry_LR"]:
        lines.append("")
        lines.append("L/R ROM symmetry (1.0 = symmetric; flag if outside 0.90-1.10):")
        for base, ratio in sorted(summary["symmetry_LR"].items()):
            flag = "" if 0.90 <= ratio <= 1.10 else "  <-- ASYMMETRIC"
            lines.append(f"  {base:<20} {ratio:>6.3f}{flag}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Kinematics report from an OpenSim .mot/.sto")
    ap.add_argument("--mot", required=True, help="OpenSim IK output (.mot or .sto)")
    ap.add_argument("--out-plot", default=None, help="Optional PNG of coordinate curves")
    args = ap.parse_args(argv)

    time, coords, meta = read_storage(args.mot)
    summary = summarize(time, coords, meta)
    print(format_report(summary))
    if args.out_plot:
        path = plot_coordinates(time, coords, args.out_plot)
        print(f"\nWrote plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
