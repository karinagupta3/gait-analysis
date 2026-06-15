# Lightweight web app (report viewer + .mot/.trc upload -> clinical report + 3D playback).
# Heavy video->.mot processing (Pose2Sim + OpenSim) is NOT in this image -- it runs where
# OpenSim is installed and feeds .mot files to this app.
FROM python:3.11-slim
WORKDIR /app

# Web tier needs only fastapi/uvicorn/numpy/scipy/matplotlib -- NOT opencv/mediapipe/onnx.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt uvicorn

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e . --no-deps      # package only; deps already pinned above

ENV GAIT_STORE_DIR=/tmp/gait-store MPLCONFIGDIR=/tmp/mpl
EXPOSE 8000
# --factory because create_app() builds the FastAPI instance; bind 0.0.0.0 for the container.
CMD uvicorn gait_analysis.web.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}
