"""Tests for Track A (Pose2Sim wrapper) and Track B (marked-model build) scaffolds."""

import pytest

from gait_analysis.biomech import build_marked_model, marker_placement, pose2sim_runner


# --- Track B: marker placement spec ---

def test_placement_spec_matches_markerset():
    spec = marker_placement.validate()
    assert spec["missing"] == [], f"active markers with no placement: {spec['missing']}"
    assert spec["extra"] == [], f"placements not in markerset: {spec['extra']}"


def test_placement_bodies_are_named():
    # Every placement names a body and a marker; lower-limb ones target a joint centre.
    for p in marker_placement.PLACEMENTS:
        assert p.marker and p.body
    by_marker = {p.marker: p for p in marker_placement.PLACEMENTS}
    assert by_marker["left_knee"].at_joint == "knee_l"
    assert by_marker["right_ankle"].at_joint == "ankle_r"


def test_build_marked_model_missing_base(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_marked_model.build(tmp_path / "nope.osim", tmp_path / "out.osim")


def test_build_marked_model_needs_opensim(tmp_path):
    base = tmp_path / "base.osim"
    base.write_text("<OpenSimDocument/>")          # spec is consistent, so it reaches OpenSim
    with pytest.raises(SystemExit, match="OpenSim"):
        build_marked_model.build(base, tmp_path / "out.osim")


# --- Track A: Pose2Sim project scaffold + config ---

def test_prepare_project_creates_layout(tmp_path):
    proj = pose2sim_runner.prepare_project(tmp_path / "session1")
    for sub in pose2sim_runner.SUBDIRS:
        assert (proj / sub).is_dir()
    cfg = (proj / "Config.toml").read_text()
    assert "[project]" in cfg
    assert "pose_model = 'HALPE_26'" in cfg     # feet keypoints for gait
    assert "rtmlib" in cfg                      # RTMPose, not OpenPose


def test_run_without_config_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        pose2sim_runner.run(tmp_path / "empty")
