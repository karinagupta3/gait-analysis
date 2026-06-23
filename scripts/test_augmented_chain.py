"""Local end-to-end test of the marker-augmentation -> scale -> IK chain.

Proves (without the cloud worker) that augmented anatomical markers drive a
sane OpenSim model: subject stands (~pelvis_ty), trunk isn't hunched
(lumbar_extension near upright), knees/hips plausible.
"""
import sys
from pathlib import Path

import numpy as np

from gait_analysis.biomech import blazepose_to_trc, marker_augmentation as MA

OUT = Path("outputs/aug_test")
OUT.mkdir(parents=True, exist_ok=True)


def get_world(video: str) -> dict:
    npz = OUT / "world.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    from gait_analysis.pose import mediapipe3d
    print(f"[mediapipe] extracting world landmarks from {video} ...")
    d = mediapipe3d.extract_world_landmarks(video)
    np.savez_compressed(npz, **d)
    return d


def main(video: str):
    d = get_world(video)
    world = blazepose_to_trc.remap_axes(d["world_landmarks"].astype(float))  # (T,33,3) Y-up
    vis = d["visibility"].astype(float)
    world[vis < 0.3] = np.nan
    world = blazepose_to_trc._fill_gaps(world)
    fps = float(d["fps"]) or 30.0
    T = world.shape[0]
    print(f"[world] frames={T} fps={fps:.1f}")

    height = MA.estimate_height_m(world)
    mass = 70.0
    print(f"[subject] est height={height:.2f} m, mass={mass} kg")

    names, pos = MA.augment(world, height, mass)
    print(f"[augment] {len(names)} markers, positions {pos.shape}")
    # sanity: study marker vertical spread should look like a standing person
    ys = pos[..., 1]
    print(f"[augment] marker Y range: {np.nanmin(ys):.3f} .. {np.nanmax(ys):.3f} m "
          f"(span {np.nanmax(ys)-np.nanmin(ys):.3f} m)")

    times = np.arange(T) / fps
    trc = blazepose_to_trc.write_trc(OUT / "markers_LSTM.trc", names,
                                     pos.astype(np.float32), times)
    print(f"[trc] wrote {trc}")

    # confirm Pose2Sim can parse our TRC
    from Pose2Sim.common import read_trc
    Q, frames_col, time_col, markers, header = read_trc(trc)
    print(f"[pose2sim read_trc] OK: {len(markers)} markers, {len(Q)} frames")

    # scale + IK with the bundled LaiUhlrich2022 model (pose_model='LSTM')
    import opensim
    from Pose2Sim import kinematics as K
    osim_setup = K.get_opensim_setup_dir()
    print(f"[opensim] setup dir: {osim_setup}")
    opensim.ModelVisualizer.addDirToGeometrySearchPaths(str(osim_setup / "Geometry"))

    print("[scale] perform_scaling ...")
    K.perform_scaling(Path(trc), "LSTM", OUT, osim_setup,
                      use_simple_model=False, subject_height=height, subject_mass=mass,
                      remove_scaling_setup=False)
    scaled = OUT / (Path(trc).stem + ".osim")
    print(f"[scale] scaled model -> {scaled} exists={scaled.exists()}")

    print("[ik] perform_IK ...")
    K.perform_IK(Path(trc), OUT, osim_setup, "LSTM", remove_IK_setup=False)
    mot = OUT / (Path(trc).stem + ".mot")
    print(f"[ik] motion -> {mot} exists={mot.exists()}")

    # inspect key coordinates
    if mot.exists():
        lines = mot.read_text().splitlines()
        hdr_i = next(i for i, l in enumerate(lines) if l.startswith("time"))
        cols = lines[hdr_i].split("\t")
        data = np.array([[float(x) for x in l.split("\t")] for l in lines[hdr_i+1:] if l.strip()])
        def col(name):
            return data[:, cols.index(name)] if name in cols else None
        print("\n=== key joint angles (deg unless _t*) ===")
        for c in ["pelvis_ty", "pelvis_tx", "lumbar_extension", "hip_flexion_r",
                  "knee_angle_r", "ankle_angle_r", "hip_flexion_l", "knee_angle_l"]:
            v = col(c)
            if v is not None:
                print(f"  {c:18s} mean={v.mean():8.2f}  min={v.min():8.2f}  max={v.max():8.2f}")
        print("\nSANITY: standing person -> pelvis_ty ~0.8-1.0 m, "
              "lumbar_extension within ~±20°, knee 0..70°.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/stock/mixkit_man_profile.mp4")
