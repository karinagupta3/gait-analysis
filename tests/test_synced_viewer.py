"""Tests for the synced video+markers / 3D viewer scene building."""

import numpy as np

from gait_analysis.analysis.synced_viewer import kpts2d_from_npz, synced_html


def _save_rtmpose_npz(path, T=30):
    rng = np.random.default_rng(0)
    kp = rng.uniform(0, 640, (T, 17, 2)).astype(np.float32)
    kp[5, 0] = np.nan                       # a missing detection
    sc = np.full((T, 17), 0.9, np.float32)
    sc[6, 0] = 0.1                           # a low-confidence point
    np.savez(path, keypoints=kp, scores=sc, fps=np.float32(30), width=np.int32(640),
             height=np.int32(480))


def _save_mediapipe_npz(path, T=30):
    rng = np.random.default_rng(1)
    il = rng.uniform(0, 1, (T, 33, 2)).astype(np.float32)   # normalized
    np.savez(path, world_landmarks=rng.uniform(-1, 1, (T, 33, 3)).astype(np.float32),
             visibility=np.full((T, 33), 0.9, np.float32), image_landmarks=il,
             fps=np.float32(60), width=np.int32(1280), height=np.int32(720))


def test_rtmpose_overlay_scene(tmp_path):
    p = tmp_path / "rt.npz"
    _save_rtmpose_npz(p)
    s = kpts2d_from_npz(p)
    assert s["w"] == 640 and s["h"] == 480 and s["fps"] == 30
    assert len(s["frames"]) == 30 and len(s["frames"][0]) == 17
    assert s["frames"][5][0] is None                         # NaN -> None
    assert s["frames"][6][0] is None                         # low score -> None
    assert (5, 11) in s["links"]                             # COCO link present


def test_mediapipe_overlay_scaled_to_pixels(tmp_path):
    p = tmp_path / "mp.npz"
    _save_mediapipe_npz(p)
    s = kpts2d_from_npz(p)
    assert s["w"] == 1280 and len(s["frames"][0]) == 33
    pts = [pt for fr in s["frames"] for pt in fr if pt]
    assert max(x for x, _ in pts) <= 1280 and max(y for _, y in pts) <= 720   # scaled to pixels
    assert (23, 24) in s["links"]                            # BlazePose hip link


def test_synced_html_injection(tmp_path):
    p = tmp_path / "rt.npz"
    _save_rtmpose_npz(p)
    html = synced_html("clip.mp4", kpts2d_from_npz(p), None, "none")
    for ph in ("__KP__", "__VIDEO__", "__SCENE__", "__MODE__"):
        assert ph not in html
    assert "clip.mp4" in html and "OrbitControls" in html


def test_synced_html_model_mode_uses_vtk(tmp_path):
    p = tmp_path / "rt.npz"
    _save_rtmpose_npz(p)
    scene = {"fps": 30, "bodies": [{"name": "pelvis", "meshes": []}], "frames": [{"pelvis": [0, 0, 0, 0, 0, 0, 1]}]}
    html = synced_html("clip.mp4", kpts2d_from_npz(p), scene, "model")
    assert "VTKLoader" in html and '"model"' in html and "musculoskeletal model" in html
