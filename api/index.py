"""Vercel serverless entrypoint for the gait-analysis web tier (report viewer + upload).

Heavy video->.mot processing (OpenSim/MediaPipe) is NOT available here -- it stays local.
This tier serves the .mot/.trc upload -> clinical report (with 3D playback) flow.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")          # matplotlib needs a writable cache
os.environ.setdefault("GAIT_STORE_DIR", "/tmp/gait-store")  # serverless: only /tmp is writable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gait_analysis.web.app import create_app  # noqa: E402

app = create_app()
