"""Local numeric benchmark of the monocular pipeline vs OpenCap's published numbers.

Without a synchronized video+mocap ground-truth dataset we cannot compute true
joint-angle RMSE, but we CAN measure two things OpenCap also reports and compare:
  * marker tracking RMS (OpenSim IK residual)  -- OpenCap target ~1-2 cm
  * gait kinematic ranges vs literature norms   -- knee swing peak ~60-65 deg
and we quantify the monocular bilateral asymmetry (near vs far leg), which exposes
the single-camera occlusion limitation directly.
"""
import sys
from pathlib import Path

import numpy as np

from gait_analysis.biomech import (augmented_ik as AIK, blazepose_to_trc as B,
                                   marker_augmentation as MA)

CLIPS = {
    "mixkit_man_profile": "data/stock/mixkit_man_profile.mp4",
    "pexels_woman_side": "data/stock/pexels_woman_side_1080.mp4",
    "pexels_woman_curly": "data/stock/pexels_woman_curly.mp4",
}
CACHE = {"mixkit_man_profile": "outputs/run_quick_test/world_landmarks.npz"}


def world_for(name, path, work):
    npz = work / "world.npz"
    if name in CACHE and Path(CACHE[name]).exists():
        d = dict(np.load(CACHE[name]))
    elif npz.exists():
        d = dict(np.load(npz))
    else:
        from gait_analysis.pose import mediapipe3d
        d = mediapipe3d.extract_world_landmarks(path)
        np.savez_compressed(npz, **d)
    return d


def marker_rms_cm(out_dir: Path) -> float:
    sto = out_dir / "_ik_marker_errors.sto"
    if not sto.exists():
        return float("nan")
    lines = sto.read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.lower().startswith("time"))
    cols = lines[hdr].split("\t")
    ri = next((k for k, c in enumerate(cols) if "RMS" in c), None)
    if ri is None:
        return float("nan")
    vals = [float(l.split("\t")[ri]) for l in lines[hdr + 1:] if l.strip()]
    return float(np.mean(vals) * 100.0)   # m -> cm


def bench_clip(name, path):
    work = Path("outputs/bench") / name
    work.mkdir(parents=True, exist_ok=True)
    d = world_for(name, path, work)
    vis = d["visibility"].astype(float)
    world_raw = B.remap_axes(d["world_landmarks"].astype(float))
    masked = world_raw.copy(); masked[vis < 0.3] = np.nan
    basis = MA.facing_basis(masked, vis)          # robust confidence-weighted facing
    world = B._fill_gaps(masked)
    fps = float(d["fps"]) or 30.0
    world = MA.smooth_world(world, fps)            # Butterworth de-jitter
    h = MA.estimate_height_m(world)
    names, pos = MA.augment(world, h, 70.0, basis=basis)
    trc = B.write_trc(work / "markers.trc", names, pos.astype(np.float32),
                      np.arange(world.shape[0]) / fps)
    mot, scaled = AIK.scale_and_ik(trc, work, h, 70.0)
    import shutil
    shutil.copyfile(mot, work / "coordinates.mot")

    lines = (work / "coordinates.mot").read_text().splitlines()
    i = next(i for i, l in enumerate(lines) if l.startswith("time"))
    cols = lines[i].split("\t")
    data = np.array([[float(x) for x in l.split("\t")] for l in lines[i+1:] if l.strip()])

    def rom(n):
        if n not in cols:
            return float("nan")
        v = data[:, cols.index(n)]
        return float(np.percentile(v, 97.5) - np.percentile(v, 2.5))
    kr, kl = rom("knee_angle_r"), rom("knee_angle_l")
    return {
        "frames": data.shape[0], "height_est": round(h, 2),
        "marker_rms_cm": round(marker_rms_cm(work), 2),
        "knee_rom_r": round(kr, 1), "knee_rom_l": round(kl, 1),
        "knee_peak": round(max(kr, kl), 1),
        "hip_flex_rom": round(rom("hip_flexion_r"), 1),
        "pelvis_rot_mean": round(float(np.mean(data[:, cols.index("pelvis_rotation")])), 1)
                            if "pelvis_rotation" in cols else None,
        "leg_symmetry": round(min(kr, kl) / max(kr, kl), 2) if max(kr, kl) > 0 else 0,
    }


def main():
    which = sys.argv[1:] or list(CLIPS)
    print(f"\n{'clip':22s} {'frm':>4} {'h(m)':>5} {'mRMS(cm)':>8} "
          f"{'kneeR':>6} {'kneeL':>6} {'peak':>5} {'hipROM':>6} {'pelvROT':>7} {'sym':>5}")
    print("-" * 92)
    for name in which:
        try:
            r = bench_clip(name, CLIPS[name])
            print(f"{name:22s} {r['frames']:>4} {r['height_est']:>5} "
                  f"{r['marker_rms_cm']:>8} {r['knee_rom_r']:>6} {r['knee_rom_l']:>6} "
                  f"{r['knee_peak']:>5} {r['hip_flex_rom']:>6} "
                  f"{str(r['pelvis_rot_mean']):>7} {r['leg_symmetry']:>5}")
        except Exception as e:
            print(f"{name:22s}  ERROR: {e}")
    print("\nOpenCap reference (2-camera, validated): marker RMS ~1-2 cm; "
          "walking knee swing peak ~60-65 deg; joint MAE ~4.5 deg.")
    print("sym=1.0 is perfect L/R symmetry; low sym = far-leg occlusion (monocular limit).")


if __name__ == "__main__":
    main()
