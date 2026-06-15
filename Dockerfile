# Lightweight web app (report viewer + session store). Containerizes the part that
# is deployable anywhere. Heavy video->.mot processing (Pose2Sim + OpenSim) is NOT in
# this image -- it runs where OpenSim is installed and feeds .mot files to this app.
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[web]"

EXPOSE 8000
# --factory because create_app() builds the FastAPI instance.
CMD ["uvicorn", "gait_analysis.web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
