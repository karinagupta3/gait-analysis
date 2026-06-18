"""Tests for the graphable-series export (2D metrics + OpenSim .mot)."""

from gait_analysis.analysis import series_export as SE


def test_from_screening_metrics_gait_both_legs_and_nan():
    m = {"fps": 30.0, "sides": {
        "right": {"_series": {"knee": [0, 30, 60, float("nan")], "hip": [0, 10, 20, 30]}},
        "left": {"_series": {"knee": [0, 20, 40, 60], "hip": None}}}}
    p = SE.from_screening_metrics("gait", m)
    assert set(p["signals"]) == {"knee_right", "hip_right", "knee_left"}  # hip_left None dropped
    assert p["signals"]["knee_right"]["data"][3] is None                 # NaN -> None
    assert p["signals"]["knee_left"]["data"] == [0, 20, 40, 60]
    assert p["n"] == 4 and p["fps"] == 30.0


def test_from_screening_metrics_squat():
    m = {"fps": 60.0, "_series": {"knee": [1, 2, 3], "hip": [4, 5, 6], "trunk": [7, 8, 9]}}
    p = SE.from_screening_metrics("squat", m)
    assert set(p["signals"]) == {"knee", "hip", "trunk"}


def test_from_mot(tmp_path):
    mot = tmp_path / "c.mot"
    mot.write_text("Coordinates\nversion=1\ninDegrees=yes\nendheader\n"
                   "time\tknee_angle_r\thip_flexion_r\n0\t10\t5\n0.1\t20\t6\n0.2\t30\t7\n")
    p = SE.from_mot(mot)
    assert p["task"] == "3d"
    assert "knee_angle_r" in p["signals"] and "hip_flexion_r" in p["signals"]
    assert p["signals"]["knee_angle_r"]["data"] == [10.0, 20.0, 30.0]
    assert p["t"] == [0.0, 0.1, 0.2]
    assert abs(p["fps"] - 10.0) < 0.01
