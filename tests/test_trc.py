"""Tests for the BlazePose -> OpenSim .trc writer (offline)."""

import numpy as np

from gait_analysis.biomech.blazepose_to_trc import npz_to_trc, remap_axes, write_trc
from gait_analysis.pose.mediapipe3d import BLAZEPOSE_33


def test_remap_axes():
    w = np.array([[1.0, 2.0, 3.0]])     # mp_x, mp_y(down), mp_z
    out = remap_axes(w)
    assert out[0, 0] == 1.0             # anterior = mp_x
    assert out[0, 1] == -2.0            # up = -mp_y
    assert out[0, 2] == 3.0             # lateral = mp_z


def test_write_trc_structure(tmp_path):
    T, M = 10, len(BLAZEPOSE_33)
    pos = np.random.rand(T, M, 3).astype(np.float32)
    times = np.arange(T) / 30.0
    path = write_trc(tmp_path / "m.trc", list(BLAZEPOSE_33), pos, times)

    lines = path.read_text().splitlines()
    # Row 3 holds counts: NumFrames, NumMarkers.
    meta = lines[2].split("\t")
    assert int(meta[2]) == T
    assert int(meta[3]) == M
    # 5 header lines + T data rows.
    data_rows = [ln for ln in lines[5:] if ln.strip()]
    assert len(data_rows) == T


def test_npz_to_trc_roundtrip(tmp_path):
    T, M = 8, len(BLAZEPOSE_33)
    npz = tmp_path / "mp.npz"
    np.savez(
        npz,
        world_landmarks=np.random.rand(T, M, 3).astype(np.float32),
        visibility=np.ones((T, M), dtype=np.float32),
        image_landmarks=np.zeros((T, M, 2), dtype=np.float32),
        fps=np.float32(30.0),
        width=np.int32(1920),
        height=np.int32(1080),
    )
    trc = npz_to_trc(npz, tmp_path / "out.trc")
    assert trc.exists()
    # Low-visibility markers become blanks: set all vis=0 -> all-blank data fields.
    np.savez(
        npz,
        world_landmarks=np.random.rand(T, M, 3).astype(np.float32),
        visibility=np.zeros((T, M), dtype=np.float32),
        image_landmarks=np.zeros((T, M, 2), dtype=np.float32),
        fps=np.float32(30.0), width=np.int32(1920), height=np.int32(1080),
    )
    trc2 = npz_to_trc(npz, tmp_path / "out2.trc")
    body = [ln for ln in trc2.read_text().splitlines()[5:] if ln.strip()]
    # Each data row should still exist (frame# + time present), markers blank.
    assert len(body) == T
