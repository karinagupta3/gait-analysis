# Web app + single-phone 2D screening (MediaPipe pose -> sagittal angles -> report).
# Heavy video->.mot 3D (Pose2Sim + OpenSim) is still NOT in this image -- that's the
# two-phone "accurate" mode, run where OpenSim is installed.
FROM python:3.11-slim
WORKDIR /app

# System libs MediaPipe/OpenCV need at runtime (libGL + glib).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt uvicorn

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e . --no-deps      # package only; deps already pinned above

# Bake the MediaPipe pose-landmarker model into the image (screening needs it).
RUN python -c "import urllib.request,os; os.makedirs('/app/models',exist_ok=True); \
urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task','/app/models/pose_landmarker_heavy.task')"

ENV GAIT_STORE_DIR=/tmp/gait-store MPLCONFIGDIR=/tmp/mpl \
    GAIT_POSE_TASK_MODEL=/app/models/pose_landmarker_heavy.task
EXPOSE 8000
# --factory because create_app() builds the FastAPI instance; bind 0.0.0.0 for the container.
CMD uvicorn gait_analysis.web.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}
