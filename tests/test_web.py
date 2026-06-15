"""Test the FastAPI web app (skipped if the web extra isn't installed)."""

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient   # noqa: E402

from gait_analysis.web.app import create_app   # noqa: E402


def _mot_bytes(n=200):
    t = np.linspace(0, 2.0, n)
    w = 2 * np.pi * 1.0 * t
    cols = ["time", "hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l"]
    data = np.column_stack([t, 20 * np.sin(w), 20 * np.sin(w + np.pi),
                            30 + 30 * np.sin(w), 30 + 30 * np.sin(w + np.pi)])
    lines = ["t", "nColumns=5", "inDegrees=yes", "endheader", "\t".join(cols)]
    for r in data:
        lines.append("\t".join(f"{v:.5f}" for v in r))
    return ("\n".join(lines) + "\n").encode()


def test_health_and_index():
    c = TestClient(create_app())
    assert c.get("/health").json() == {"ok": True}
    r = c.get("/")
    assert r.status_code == 200 and "Gait Analysis" in r.text


def test_upload_generates_report():
    c = TestClient(create_app())
    r = c.post("/report",
               files={"mot": ("trial.mot", _mot_bytes(), "text/plain")},
               data={"subject": "Tester", "trial": "walk", "speed": "1.2"})
    # redirected to the session report page
    assert r.status_code == 200
    assert "Clinical signature flags" in r.text
    assert "Tester" in r.text
    # session now appears in the list + API
    assert any(s["subject"] == "Tester" for s in c.get("/api/sessions").json())
