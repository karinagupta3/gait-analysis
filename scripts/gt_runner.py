"""Run our monocular pipeline on ONE video with a chosen backend -> OpenSim .mot.

Reusable unit for the ground-truth benchmark: given a video + subject height/mass,
produce joint angles via either backend, so we can compare each against the lab
mocap ground truth (scripts/compare_mot.py).

  backend="mediapipe": MediaPipe world-landmarks (current production path)
  backend="hybrid":    RTMPose 2D (x,y) + MediaPipe depth (z)  [biomech/hybrid3d]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gait_analysis.biomech import (augmented_ik as AIK, blazepose_to_trc as B,
                                   hybrid3d, marker_augmentation as MA)


def _mp(video, cache: Path | None):
    if cache and cache.exists():
        return {k: v for k, v in np.load(cache).items()}
    from gait_analysis.pose import mediapipe3d
    d = mediapipe3d.extract_world_landmarks(video)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **d)
    return d


def _rtm(video, cache: Path | None):
    if cache and cache.exists():
        return {k: v for k, v in np.load(cache).items()}
    from gait_analysis.pose import rtmpose2d
    d = rtmpose2d.extract_rtmpose(video)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **d)
    return d


def process_video(video, height_m: float, mass_kg: float, backend: str,
                  out_dir, cache_dir: Path | None = None) -> Path:
    """video -> coordinates.mot in out_dir, via `backend`. Returns the .mot path."""
    video = str(video)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mp = _mp(video, (cache_dir / "mp.npz") if cache_dir else None)

    if backend == "hybrid":
        rtm = _rtm(video, (cache_dir / "rtm.npz") if cache_dir else None)
        d = hybrid3d.build_hybrid(mp, rtm, height_m)
    elif backend == "mediapipe":
        d = mp
    else:
        raise ValueError(f"unknown backend {backend!r}")

    vis = d["visibility"].astype(float)
    raw = B.remap_axes(d["world_landmarks"].astype(float))
    masked = raw.copy(); masked[vis < 0.3] = np.nan
    basis = MA.facing_basis(masked, vis)
    world = B._fill_gaps(masked)
    fps = float(d["fps"]) or 30.0
    world = MA.smooth_world(world, fps)
    names, pos = MA.augment(world, height_m, mass_kg, basis=basis)
    trc = B.write_trc(out_dir / "markers.trc", names, pos.astype(np.float32),
                      np.arange(world.shape[0]) / fps)
    mot, _ = AIK.scale_and_ik(trc, out_dir, height_m, mass_kg)
    import shutil
    shutil.copyfile(mot, out_dir / "coordinates.mot")
    return out_dir / "coordinates.mot"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--mass", type=float, default=70.0)
    ap.add_argument("--backend", default="mediapipe", choices=["mediapipe", "hybrid"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    p = process_video(a.video, a.height, a.mass, a.backend, a.out,
                      Path(a.cache) if a.cache else None)
    print("wrote", p)
