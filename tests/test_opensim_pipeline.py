"""Tests for OpenSim setup generation, marker validation, and the pipeline (offline)."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gait_analysis.biomech import opensim_ik, opensim_setup
from gait_analysis.biomech.blazepose_to_trc import write_trc
from gait_analysis.biomech.markerset import (
    IK_MARKER_WEIGHTS, active_markers, validate_against_trc_markers,
)
from gait_analysis.pipeline import report_from_mot, run_quick


def _write_synthetic_mot(path, n=51):
    t = np.linspace(0, 1.0, n)
    cols = ["time", "knee_angle_r", "knee_angle_l", "hip_flexion_r", "ankle_angle_r"]
    data = np.column_stack([
        t,
        35 + 0 * t,                       # stiff-knee (max ~35)
        30 + 30 * np.sin(2 * np.pi * t),  # normal-ish
        5 + 10 * np.sin(2 * np.pi * t),   # min ~ -5? actually min 5-10... keep >0 region
        2 + 0 * t,                        # foot drop (max ~2)
    ])
    lines = ["synthetic", "nColumns=5", "inDegrees=yes", "endheader", "\t".join(cols)]
    for row in data:
        lines.append("\t".join(f"{v:.5f}" for v in row))
    path.write_text("\n".join(lines) + "\n")


# --- markerset ---

def test_active_markers_excludes_zero_weight():
    active = active_markers()
    assert "left_hip" in active
    assert "left_eye" not in active           # not in weights at all
    assert all(IK_MARKER_WEIGHTS[m] > 0 for m in active)


def test_validate_against_trc_finds_missing():
    missing = validate_against_trc_markers(["left_hip", "right_hip"])
    assert "left_knee" in missing             # required but absent
    assert "left_hip" not in missing


# --- opensim_setup XML generation ---

def test_ik_task_set_xml_structure(tmp_path):
    path = opensim_setup.write_ik_task_set_xml(tmp_path / "tasks.xml")
    root = ET.parse(path).getroot()
    assert root.tag == "OpenSimDocument"
    tasks = root.findall(".//IKMarkerTask")
    names = {t.get("name") for t in tasks}
    assert names == set(active_markers())     # exactly the active markers
    # weights match
    for t in tasks:
        assert float(t.find("weight").text) == IK_MARKER_WEIGHTS[t.get("name")]


def test_scale_setup_xml_structure(tmp_path):
    path = opensim_setup.write_scale_setup_xml(
        tmp_path / "scale.xml", model_file="m.osim", static_trc="static.trc",
        output_model_file="scaled.osim", mass_kg=72, height_m=1.8,
    )
    root = ET.parse(path).getroot()
    tool = root.find("ScaleTool")
    assert tool.find("mass").text == "72"
    assert tool.find("height").text == "1800"               # m -> mm
    assert tool.find("GenericModelMaker/model_file").text == "m.osim"
    # measurement-based scaling with marker pairs from our markerset
    measurements = root.findall(".//Measurement")
    names = {m.get("name") for m in measurements}
    assert {"femur_r", "tibia_l", "torso"} <= names
    femur = next(m for m in measurements if m.get("name") == "femur_r")
    assert femur.find(".//markers").text == "right_hip right_knee"
    # MarkerPlacer reuses the IK task set
    assert root.find(".//MarkerPlacer/IKTaskSet") is not None


def test_run_scale_rejects_bad_static_trc(tmp_path):
    pos = np.zeros((2, 2, 3), dtype=np.float32)
    static = write_trc(tmp_path / "s.trc", ["left_hip", "right_hip"], pos, np.arange(2) / 30.0)
    model = tmp_path / "m.osim"
    model.write_text("<x/>")
    with pytest.raises(ValueError, match="missing IK markers"):
        opensim_ik.run_scale(model, static, tmp_path / "scaled.osim")


def test_ik_tool_setup_xml_has_files(tmp_path):
    path = opensim_setup.write_ik_tool_setup_xml(
        tmp_path / "ik.xml", model_file="m.osim", marker_file="m.trc",
        output_motion_file="out.mot", time_range=(0.0, 1.0),
    )
    root = ET.parse(path).getroot()
    tool = root.find("InverseKinematicsTool")
    assert tool.find("model_file").text == "m.osim"
    assert tool.find("marker_file").text == "m.trc"
    assert tool.find("output_motion_file").text == "out.mot"
    assert tool.find("time_range").text == "0 1"
    assert tool.find("IKTaskSet") is not None


# --- TRC marker reader + IK validation guard ---

def test_run_ik_rejects_trc_missing_markers(tmp_path):
    # TRC with only two markers -> IK must refuse before needing OpenSim.
    pos = np.zeros((3, 2, 3), dtype=np.float32)
    times = np.arange(3) / 30.0
    trc = write_trc(tmp_path / "m.trc", ["left_hip", "right_hip"], pos, times)
    model = tmp_path / "model.osim"
    model.write_text("<placeholder/>")
    with pytest.raises(ValueError, match="missing IK markers"):
        opensim_ik.run_ik_from_trc(model, trc, tmp_path / "out.mot")


def test_read_trc_marker_names(tmp_path):
    pos = np.zeros((2, 3, 3), dtype=np.float32)
    trc = write_trc(tmp_path / "m.trc", ["a", "b", "c"], pos, np.arange(2) / 30.0)
    assert opensim_ik.read_trc_marker_names(trc) == ["a", "b", "c"]


# --- pipeline ---

def test_report_from_mot_runs(tmp_path, capsys):
    mot = tmp_path / "ik.mot"
    _write_synthetic_mot(mot)
    result = report_from_mot(mot, gait_speed_m_s=1.0, plot_path=tmp_path / "p.png")
    out = capsys.readouterr().out
    assert "Kinematics report" in out
    assert "signature flags" in out
    ids = {f.rule_id for f in result["findings"]}
    assert "stiff_knee_swing" in ids          # knee max ~35 < 45
    assert (tmp_path / "p.png").exists()


def test_run_quick_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_quick(tmp_path / "nope.mov", tmp_path / "m.osim", tmp_path / "out")
