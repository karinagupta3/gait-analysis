"""Tests for the layer-2 OpenSim model viewer HTML generation (no OpenSim needed)."""

from gait_analysis.biomech.export_model_motion import _viewer_html


def test_viewer_html_injects_scene():
    scene = {
        "fps": 30.0,
        "bodies": [{"name": "pelvis", "meshes": [
            {"file": "geometry/pelvis.vtp", "found": True, "scale": [1, 1, 1],
             "offset_pos": [0, 0, 0], "offset_quat": [0, 0, 0, 1]}]}],
        "frames": [{"pelvis": [0, 0.9, 0, 0, 0, 0, 1]}],
    }
    html = _viewer_html(scene)
    assert "__SCENE__" not in html                 # placeholder replaced
    assert "VTKLoader" in html and "OrbitControls" in html
    assert "pelvis" in html and "geometry/pelvis.vtp" in html
