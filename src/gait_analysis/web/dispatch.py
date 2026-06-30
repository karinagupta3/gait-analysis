"""Tier-1 side of dispatching a 3D job to the tier-2 OpenSim worker.

WHY THIS EXISTS
---------------
OpenSim is conda-only and too heavy for the slim tier-1 web image, so the 3D
modes ("quick" / "accurate") are handed off to the tier-2 worker
(`gait_analysis.worker.queue_worker`) running on a conda image. This module is
the *only* tier-1 code that talks to Azure Storage: it uploads the input video,
enqueues the job, polls progress, and pulls back the worker's outputs.

THE CONTRACT (must match queue_worker.py EXACTLY)
-------------------------------------------------
  Storage Queue : "gait-jobs"
  Input blob    : gait-in/<session_id>/video.<ext>
  Output blobs  : gait-out/<session_id>/{coordinates.mot, markers.trc,
                                         report.html, status.json}
  Queue message : {"session_id": "...", "mode": "quick"|"accurate",
                   "ext": "mp4", "speed": <float|null>}

The worker downloads gait-in/<sid>/video.<ext>, runs the pipeline, and writes the
output blobs above. `coordinates.mot` is the canonical hand-off; `status.json`
is what tier-1 polls for {"state": "processing"|"done"|"error", ...}.

ENVIRONMENT
-----------
  GAIT_STORAGE_CONNECTION : storage account connection string (required).
                            Same variable the worker reads — one account, one
                            queue, two containers, shared by both tiers.

DESIGN NOTES
------------
azure-storage-blob / azure-storage-queue are imported lazily *inside* the
functions so tier-1 (and `import gait_analysis.web.dispatch`) works in
environments that never dispatch 3D jobs and don't have the azure SDKs
installed. Nothing at module import time touches Azure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Fixed names — copied verbatim from worker/queue_worker.py. These three strings
# ARE the wire contract; if they drift from the worker, jobs silently vanish.
QUEUE_NAME = "gait-jobs"
IN_CONTAINER = "gait-in"
OUT_CONTAINER = "gait-out"

# Valid modes for the queue message. "quick" = 1-phone 3D, "accurate" = 2-phone.
VALID_MODES = ("quick", "accurate")


# --------------------------------------------------------------------------- #
# Connection / client helpers (lazy-imported azure SDKs)
# --------------------------------------------------------------------------- #
def storage_configured() -> bool:
    """True iff GAIT_STORAGE_CONNECTION is set (i.e. 3D dispatch is possible).

    Tier-1 calls this to decide whether to offer the 3D modes at all, so the app
    degrades gracefully (2D screening only) when no storage account is wired up.
    """
    return bool(os.environ.get("GAIT_STORAGE_CONNECTION"))


def _conn_str() -> str:
    """Return the storage connection string or fail loudly (never fabricate)."""
    conn = os.environ.get("GAIT_STORAGE_CONNECTION")
    if not conn:
        raise RuntimeError(
            "GAIT_STORAGE_CONNECTION is not set. Tier-1 needs the storage "
            "account connection string to dispatch 3D jobs to the tier-2 "
            "worker. (Set the same value the worker uses.)"
        )
    return conn


def _blob_service():
    """Lazy-import azure-storage-blob and return a BlobServiceClient."""
    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient.from_connection_string(_conn_str())


def _queue_client():
    """Lazy-import azure-storage-queue and return a QueueClient for gait-jobs.

    The worker enqueues nothing — it only receives — but its messages are plain
    JSON strings, so we configure this client to send raw text (no base64). If
    you ever change the encoding here, change the worker's decode side too.
    """
    from azure.storage.queue import QueueClient
    # Match the worker EXACTLY: it uses bare from_connection_string (default no-op
    # encoding) and json.loads(message.content). Using the same call on both sides
    # guarantees the encode/decode policies agree across SDK versions.
    return QueueClient.from_connection_string(_conn_str(), QUEUE_NAME)


def _ensure_container(blob_service, container: str) -> None:
    """Create the container if it doesn't exist (idempotent, best-effort).

    The worker assumes the containers already exist; on a fresh account the very
    first dispatch would otherwise 404 on upload. Creating here makes tier-1
    self-sufficient without a separate provisioning step.
    """
    from azure.core.exceptions import ResourceExistsError
    try:
        blob_service.create_container(container)
    except ResourceExistsError:
        pass


def _ensure_queue(queue) -> None:
    """Create the gait-jobs queue if absent (idempotent, best-effort)."""
    from azure.core.exceptions import ResourceExistsError
    try:
        queue.create_queue()
    except ResourceExistsError:
        pass


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def dispatch_job(
    session_id: str,
    local_video_path: str | Path,
    mode: str,
    ext: str,
    speed: float | None = None,
    height_m: float | None = None,
    mass_kg: float | None = None,
) -> dict:
    """Upload the video and enqueue a 3D job for the tier-2 worker.

    Steps (the tier-1 half of the contract):
      1. Upload `local_video_path` to gait-in/<session_id>/video.<ext>.
      2. Enqueue {"session_id", "mode", "ext", "speed"} on the gait-jobs queue.

    Args:
        session_id:       opaque session id; namespaces all blobs for this job.
        local_video_path: path to the already-saved upload on the tier-1 box.
        mode:             "quick" (1-phone) or "accurate" (2-phone).
        ext:              video extension WITHOUT the dot, e.g. "mp4".
        speed:            optional gait speed in m/s for clinical signatures, or
                          None to let the pipeline infer / skip it.

    Returns the queue message dict that was sent (handy for logging/tests).

    Raises RuntimeError if storage isn't configured, ValueError on a bad mode,
    or FileNotFoundError if the local video is missing.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"mode must be one of {VALID_MODES!r}, got {mode!r}.")

    src = Path(local_video_path)
    if not src.is_file():
        raise FileNotFoundError(f"video not found: {src}")

    # Normalize ext to the bare extension (no leading dot, no surrounding ws).
    ext = ext.lstrip(".").strip()

    blob_service = _blob_service()
    _ensure_container(blob_service, IN_CONTAINER)
    # Make sure the OUTPUT container exists too, so the worker's status/output
    # writes (and our later polling) never race a missing container.
    _ensure_container(blob_service, OUT_CONTAINER)

    # 1) Upload the input video to gait-in/<sid>/video.<ext>, overwriting any
    #    prior attempt for this session (re-dispatch is safe / idempotent).
    blob_name = f"{session_id}/video.{ext}"
    bc = blob_service.get_blob_client(container=IN_CONTAINER, blob=blob_name)
    with open(src, "rb") as fh:
        bc.upload_blob(fh, overwrite=True)

    # 2) Enqueue the job. Shape MUST match what the worker's process_message
    #    reads: session_id, mode, ext, speed (float | None).
    message = {
        "session_id": session_id,
        "mode": mode,
        "ext": ext,
        "speed": speed,
        "height_m": height_m,
        "mass_kg": mass_kg,
    }
    queue = _queue_client()
    _ensure_queue(queue)
    queue.send_message(json.dumps(message))

    return message


def dispatch_project(session_id: str, project_dir: str | Path,
                     speed: float | None = None) -> int:
    """Upload an assembled Pose2Sim project tree and enqueue a 2-phone accurate job.

    Uploads every file under `project_dir` to gait-in/<sid>/project/<relpath> (the
    worker downloads that prefix, runs the engine, and writes gait-out/<sid>/...),
    then enqueues {"session_id", "mode":"accurate", "speed"}. Returns the file count.
    """
    project_dir = Path(project_dir)
    if not project_dir.is_dir():
        raise FileNotFoundError(f"project dir not found: {project_dir}")

    blob_service = _blob_service()
    _ensure_container(blob_service, IN_CONTAINER)
    _ensure_container(blob_service, OUT_CONTAINER)

    n = 0
    for p in sorted(project_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(project_dir).as_posix()
        bc = blob_service.get_blob_client(
            container=IN_CONTAINER, blob=f"{session_id}/project/{rel}")
        with open(p, "rb") as fh:
            bc.upload_blob(fh, overwrite=True)
        n += 1
    if n == 0:
        raise RuntimeError(f"project dir {project_dir} has no files to upload")

    queue = _queue_client()
    _ensure_queue(queue)
    queue.send_message(json.dumps(
        {"session_id": session_id, "mode": "accurate", "speed": speed}))
    return n


def poll_status(session_id: str) -> dict:
    """Return the worker's status for a session, or a synthetic 'queued' state.

    Reads gait-out/<session_id>/status.json (the worker writes this as it moves
    through processing -> done/error). If the blob doesn't exist yet, the worker
    hasn't picked the job up, so we report {"state": "queued"} — matching the
    vocabulary tier-1's JobManager already uses (queued|running|done|error;
    the worker uses "processing" for running).

    Raises RuntimeError if storage isn't configured. Any other read error is
    re-raised so callers can distinguish "not started" from "storage is broken".
    """
    from azure.core.exceptions import ResourceNotFoundError

    blob_service = _blob_service()
    bc = blob_service.get_blob_client(
        container=OUT_CONTAINER, blob=f"{session_id}/status.json")
    try:
        raw = bc.download_blob().readall()
    except ResourceNotFoundError:
        # No status yet => the worker hasn't started this job.
        return {"state": "queued"}
    return json.loads(raw)


def fetch_outputs(session_id: str, dest_dir: str | Path) -> list[str]:
    """Download the worker's outputs for a session into `dest_dir`.

    Pulls report.html and coordinates.mot (both expected on success), plus
    markers.trc if the worker produced it. status.json is intentionally NOT
    fetched here — poll_status() handles that separately.

    Args:
        session_id: the session whose gait-out/<sid>/ prefix to read.
        dest_dir:   local directory to write into (created if needed).

    Returns the list of filenames actually downloaded (e.g.
    ["report.html", "coordinates.mot", "markers.trc"]), in fetch order.

    Raises RuntimeError if storage isn't configured. coordinates.mot is the
    contract artifact, but this function does not *enforce* its presence — it
    simply reports what was available, leaving policy to the caller.
    """
    from azure.core.exceptions import ResourceNotFoundError

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    blob_service = _blob_service()

    # Order matters for the caller's UX: the human-viewable report first, then
    # the canonical .mot, then the optional marker trajectories.
    wanted = ["report.html", "coordinates.mot", "markers.trc"]
    fetched: list[str] = []
    for name in wanted:
        bc = blob_service.get_blob_client(
            container=OUT_CONTAINER, blob=f"{session_id}/{name}")
        try:
            data = bc.download_blob().readall()
        except ResourceNotFoundError:
            # markers.trc is optional; for report/mot a miss just means the job
            # isn't done (or failed) — the caller decides via poll_status().
            continue
        (dest / name).write_bytes(data)
        fetched.append(name)

    return fetched


def fetch_synced(session_id: str, dest_dir: str | Path) -> int:
    """Download the worker's synced-viewer folder (gait-out/<sid>/synced/**) into
    dest_dir, preserving sub-paths (viewer.html, overlay.mp4, geometry/*.vtp). This
    is how the cloud OpenSim BONE rendering reaches tier-1 to be served. Returns the
    number of files downloaded (0 if the worker produced no scene)."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    cc = _blob_service().get_container_client(OUT_CONTAINER)
    prefix = f"{session_id}/synced/"
    n = 0
    for b in cc.list_blobs(name_starts_with=prefix):
        rel = b.name[len(prefix):]
        if not rel:
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
        n += 1
    return n
