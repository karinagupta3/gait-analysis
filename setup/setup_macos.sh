#!/usr/bin/env bash
# Phase-1 environment setup for Apple Silicon (M-series) macOS.
# Creates a venv and installs the commercial-license-safe core stack.
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
echo ">> Using interpreter: $($PY --version)"

if [ ! -d .venv ]; then
  echo ">> Creating .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
echo ">> Installing core package (editable) + dev extras"
pip install -e ".[dev]"

cat <<'EOF'

Core install done. Quick check:
    source .venv/bin/activate
    pytest -q                       # runs synthetic gait-event tests (no video needed)

Run on a real clip (downloads RTMPose weights once):
    gait-pose --video data/walk.mov --out outputs/walk.npz --overlay outputs/walk_overlay.mp4
    gait-spatiotemporal --keypoints outputs/walk.npz

ACCURATE MODE (multi-camera) extras -- install separately when you start Phase 1b:
    pip install pose2sim            # BSD-3, multi-view triangulation -> OpenSim
    # OpenSim: install via conda. Native osx-arm64 may be unavailable; if so,
    # create an x86_64 (Rosetta) conda env:
    #   CONDA_SUBDIR=osx-64 conda create -n opensim python=3.11
    #   conda activate opensim && conda install -c opensim-org opensim
EOF
