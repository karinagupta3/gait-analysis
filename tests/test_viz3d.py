"""Tests for the 3D motion playback scene builder."""

import numpy as np

from gait_analysis.analysis import viz3d
from gait_analysis.biomech.blazepose_to_trc import write_trc


def _walk_trc(path, T=240):
    t = np.arange(T) / 60.0
    w = 2 * np.pi * 0.9 * t
    names = ["RHip", "LHip", "RKnee", "LKnee", "RAnkle", "LAnkle", "RShoulder", "LShoulder"]
    pos = np.zeros((T, len(names), 3))
    px = 1.2 * t
    heights = {"RHip": 0.95, "LHip": 0.95, "RKnee": 0.5, "LKnee": 0.5,
               "RAnkle": 0.08, "LAnkle": 0.08, "RShoulder": 1.4, "LShoulder": 1.4}
    lat = {"RHip": 0.1, "LHip": -0.1, "RKnee": 0.1, "LKnee": -0.1,
           "RAnkle": 0.1, "LAnkle": -0.1, "RShoulder": 0.15, "LShoulder": -0.15}
    for i, n in enumerate(names):
        swing = 0.3 * np.sin(w + (np.pi if n.startswith("L") else 0)) if "Ankle" in n else 0
        pos[:, i] = np.column_stack([px + swing, np.full(T, heights[n]), np.full(T, lat[n])])
    write_trc(path, names, pos, t)


def test_scene_structure(tmp_path):
    p = tmp_path / "w.trc"
    _walk_trc(p)
    scene = viz3d.trc_to_scene(p, max_frames=100)
    assert len(scene["frames"]) <= 100
    assert scene["fps"] > 0
    assert [0, 1] in scene["links"]                       # RHip-LHip link resolved
    assert all(len(fr) == len(scene["names"]) for fr in scene["frames"])
    # figure is centered near origin
    pts = np.array([p for fr in scene["frames"] for p in fr if p is not None])
    assert abs(pts.mean(axis=0)).max() < 1.5


def test_report_section_embeds_viewer(tmp_path):
    p = tmp_path / "w.trc"
    _walk_trc(p)
    html = viz3d.report_section(p)
    assert "3D motion playback" in html
    assert "from 'three'" in html and "__SCENE__" not in html   # bare import (map in report head); scene injected


def test_report_section_graceful_on_bad_trc(tmp_path):
    bad = tmp_path / "bad.trc"
    bad.write_text("not a trc\n")
    assert viz3d.report_section(bad) == ""
