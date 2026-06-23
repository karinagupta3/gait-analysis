"""Tier-2 OpenSim cloud worker: video -> OpenSim .mot for the 3D modes.

WHY THIS IS A SEPARATE TIER
---------------------------
OpenSim is conda-only and large, so it CANNOT live in the slim tier-1 web image
(see Dockerfile vs Dockerfile.worker). Tier-1 (the FastAPI app) handles uploads,
session storage, and rendering reports; tier-2 (this worker) does the heavy
video -> 3D -> OpenSim IK number-crunching on a conda image that actually has
OpenSim installed.

THE .mot IS THE CONTRACT WITH TIER-1
------------------------------------
Tier-1 dispatches a job by (a) uploading the video to the input blob and (b)
enqueueing a JSON message on the Storage Queue. This worker produces an OpenSim
`coordinates.mot` (plus `markers.trc` and a `report.html`) under the output blob
prefix. The `.mot` is the canonical hand-off: it is exactly what tier-1 can feed
back into `pipeline.report_from_mot()` to render kinematics + clinical signatures
without ever needing OpenSim itself. Everything else is auxiliary.

LAYOUT (names are fixed and shared with tier-1)
-----------------------------------------------
  Storage Queue : "gait-jobs"
  Input blob    : gait-in/<session_id>/video.<ext>
  Output blobs  : gait-out/<session_id>/{coordinates.mot, markers.trc,
                                         report.html, status.json}

  Queue message : {"session_id": "...", "mode": "quick"|"accurate",
                   "ext": "mp4", "speed": <float|null>}

  - mode "quick"    : 1-phone 3D -> pipeline.run_quick(video, model, outdir)
  - mode "accurate" : 2-phone    -> pipeline.run_accurate(project_dir)

ENVIRONMENT
-----------
  GAIT_STORAGE_CONNECTION : storage account connection string (required)
  GAIT_OSIM_MODEL         : path to the baked marked model
                            (default /app/models/LaiUhlrich2022_ga.osim)

The worker is a console-style loop. It pulls messages with a long (~30 min)
visibility timeout so a slow OpenSim run does not let the message reappear and
get double-processed, writes a `status.json` so tier-1 can poll progress, and is
robustly wrapped per-message so one bad job never kills the loop.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path

# Fixed names — must match tier-1 and setup/deploy_worker_azure.sh exactly.
QUEUE_NAME = "gait-jobs"
IN_CONTAINER = "gait-in"
OUT_CONTAINER = "gait-out"

# OpenSim IK runs can take many minutes; keep the message invisible for ~30 min so
# it is not redelivered mid-processing. (Storage Queue max visibility is 7 days.)
VISIBILITY_TIMEOUT_S = 30 * 60

# Default model path matches Dockerfile.worker's ENV + the baked-in marked model.
DEFAULT_OSIM_MODEL = "/app/models/LaiUhlrich2022_ga.osim"


def _conn_str() -> str:
    """Return the storage connection string or fail loudly (never fabricate)."""
    conn = os.environ.get("GAIT_STORAGE_CONNECTION")
    if not conn:
        raise SystemExit(
            "GAIT_STORAGE_CONNECTION is not set. This worker needs the storage "
            "account connection string (set by setup/deploy_worker_azure.sh)."
        )
    return conn


def _osim_model() -> str:
    """Path to the marked OpenSim model baked into the worker image."""
    return os.environ.get("GAIT_OSIM_MODEL", DEFAULT_OSIM_MODEL)


def _blob_service():
    """Lazy-import azure-storage-blob and return a BlobServiceClient."""
    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient.from_connection_string(_conn_str())


def _queue_client():
    """Lazy-import azure-storage-queue and return a QueueClient for gait-jobs."""
    from azure.storage.queue import QueueClient
    return QueueClient.from_connection_string(_conn_str(), QUEUE_NAME)


def _download_blob(blob_service, container: str, name: str, dest: Path) -> Path:
    """Download a single blob to `dest` (creating parents). Returns dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    bc = blob_service.get_blob_client(container=container, blob=name)
    with open(dest, "wb") as fh:
        fh.write(bc.download_blob().readall())
    return dest


def _upload_blob(blob_service, container: str, name: str, src: Path,
                 content_type: str | None = None) -> None:
    """Upload a local file to gait-out/<name>, overwriting any prior version."""
    from azure.storage.blob import ContentSettings
    bc = blob_service.get_blob_client(container=container, blob=name)
    settings = ContentSettings(content_type=content_type) if content_type else None
    with open(src, "rb") as fh:
        bc.upload_blob(fh, overwrite=True, content_settings=settings)


def _write_status(blob_service, session_id: str, payload: dict) -> None:
    """Write gait-out/<sid>/status.json so tier-1 can poll job progress."""
    bc = blob_service.get_blob_client(
        container=OUT_CONTAINER, blob=f"{session_id}/status.json")
    from azure.storage.blob import ContentSettings
    bc.upload_blob(
        json.dumps(payload).encode("utf-8"), overwrite=True,
        content_settings=ContentSettings(content_type="application/json"))


def _upload_outputs(blob_service, session_id: str, outdir: Path) -> list[str]:
    """Upload the standard outputs that exist under `outdir` to gait-out/<sid>/.

    The .mot is the contract; .trc and report.html are uploaded when present.
    Returns the list of blob names actually uploaded.
    """
    uploaded: list[str] = []
    wanted = [
        ("coordinates.mot", "text/plain"),
        ("markers.trc", "text/plain"),
        ("report.html", "text/html"),
    ]
    for fname, ctype in wanted:
        local = outdir / fname
        if local.exists():
            _upload_blob(blob_service, OUT_CONTAINER,
                         f"{session_id}/{fname}", local, content_type=ctype)
            uploaded.append(fname)
    return uploaded


def process_message(blob_service, msg: dict, workroot: Path) -> dict:
    """Run one job end to end. Returns the success status payload.

    Reuses the existing pipeline UNCHANGED:
      * quick    -> pipeline.run_quick(video, model, outdir, gait_speed_m_s=speed)
                    writes coordinates.mot / markers.trc into outdir.
      * accurate -> pipeline.run_accurate(project_dir, gait_speed_m_s=speed)
                    Pose2Sim writes the .mot under the project dir; run_accurate
                    returns it as result["mot"].
    """
    from gait_analysis import pipeline

    session_id = msg["session_id"]
    mode = msg.get("mode", "quick")
    ext = msg.get("ext", "mp4")
    speed = msg.get("speed")  # float | None — gait speed context for signatures

    # Mark the job as processing so tier-1 sees movement immediately.
    _write_status(blob_service, session_id, {"state": "processing", "mode": mode})

    sid_dir = workroot / session_id
    outdir = sid_dir / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    # Pull the input video: gait-in/<sid>/video.<ext>
    video = _download_blob(
        blob_service, IN_CONTAINER, f"{session_id}/video.{ext}",
        sid_dir / f"video.{ext}")

    if mode == "quick":
        # 1-phone 3D: MediaPipe -> marker augmentation -> OpenSim IK -> coordinates.mot.
        pipeline.run_quick(
            video, _osim_model(), outdir, gait_speed_m_s=speed,
            height_m=msg.get("height_m"), mass_kg=msg.get("mass_kg"))
        report_src = outdir / "report.html"
    elif mode == "accurate":
        # 2-phone: Pose2Sim expects a project directory; the downloaded video is
        # the seed input. run_accurate triangulates + runs OpenSim IK and returns
        # the .mot path; copy it into outdir under the canonical name.
        result = pipeline.run_accurate(sid_dir, gait_speed_m_s=speed)
        mot = Path(result["mot"])
        if mot.resolve() != (outdir / "coordinates.mot").resolve():
            import shutil
            (outdir).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mot, outdir / "coordinates.mot")
        report_src = outdir / "report.html"
    else:
        raise ValueError(f"Unknown mode {mode!r} (expected 'quick' or 'accurate').")

    # Render a self-contained HTML report from the .mot (the tier-1 contract),
    # so the worker hands back a viewable artifact alongside the raw .mot.
    mot_path = outdir / "coordinates.mot"
    if mot_path.exists() and not report_src.exists():
        try:
            pipeline.report_from_mot(
                mot_path, gait_speed_m_s=speed, html_path=report_src)
        except Exception as exc:  # report is best-effort; .mot is the real product
            print(f"[note] report.html generation skipped: {exc}")

    uploaded = _upload_outputs(blob_service, session_id, outdir)
    if "coordinates.mot" not in uploaded:
        raise RuntimeError(
            "Pipeline finished but no coordinates.mot was produced — the .mot is "
            "the contract with tier-1, so this job is treated as failed.")

    # Upload the synced viewer folder (video overlay + OpenSim BONE scene + geometry)
    # so tier-1 can serve the 3D rendering inline. Best-effort: failure here doesn't
    # fail the job (the .mot/report are the contract).
    synced = outdir / "synced"
    if synced.is_dir():
        try:
            cnt = _upload_dir(blob_service, OUT_CONTAINER, synced, f"{session_id}/synced")
            print(f"[tier-2] uploaded synced viewer ({cnt} files)")
        except Exception as exc:
            print(f"[tier-2] synced upload skipped: {exc}")

    return {"state": "done", "mode": mode, "outputs": uploaded}


def _upload_dir(blob_service, container: str, local_dir, prefix: str) -> int:
    """Upload every file under local_dir to <prefix>/<relpath>. Returns file count."""
    from pathlib import Path as _P
    local_dir = _P(local_dir)
    ctypes = {".html": "text/html", ".mp4": "video/mp4", ".json": "application/json",
              ".vtp": "application/octet-stream", ".trc": "text/plain"}
    n = 0
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            _upload_blob(blob_service, container, f"{prefix}/{rel}", p,
                         content_type=ctypes.get(p.suffix.lower()))
            n += 1
    return n


def main() -> int:
    """Console loop: receive jobs from gait-jobs, process, write status, repeat."""
    blob_service = _blob_service()
    queue = _queue_client()
    print(f"[tier-2] worker up. queue={QUEUE_NAME} in={IN_CONTAINER} "
          f"out={OUT_CONTAINER} model={_osim_model()}")

    while True:
        # Pull a small batch; long visibility so OpenSim has time to finish.
        messages = queue.receive_messages(
            messages_per_page=1, visibility_timeout=VISIBILITY_TIMEOUT_S)
        got_any = False
        for message in messages:
            got_any = True
            session_id = "<unparsed>"
            with tempfile.TemporaryDirectory(prefix="gait-job-") as tmp:
                workroot = Path(tmp)
                try:
                    body = json.loads(message.content)
                    session_id = body.get("session_id", "<missing>")
                    print(f"[tier-2] processing session={session_id} "
                          f"mode={body.get('mode')}")
                    status = process_message(blob_service, body, workroot)
                    _write_status(blob_service, session_id, status)
                    print(f"[tier-2] done session={session_id} -> {status['outputs']}")
                except Exception as exc:  # one failure must not kill the loop
                    err = f"{type(exc).__name__}: {exc}"
                    print(f"[tier-2] ERROR session={session_id}: {err}")
                    traceback.print_exc()
                    try:
                        _write_status(blob_service, session_id,
                                      {"state": "error", "error": err})
                    except Exception as status_exc:
                        print(f"[tier-2] could not write error status: {status_exc}")
                finally:
                    # Always remove the message so a poison job is not retried forever.
                    # (status.json already records success/failure for tier-1.)
                    try:
                        queue.delete_message(message)
                    except Exception as del_exc:
                        print(f"[tier-2] could not delete message: {del_exc}")

        if not got_any:
            # Idle: nothing on the queue. In an Event-triggered Container Apps Job
            # the replica exits here; the queue-length scale rule starts a fresh
            # replica when new messages arrive. Returning keeps the job from
            # spinning a busy-loop and racking up cost.
            print("[tier-2] queue empty; exiting (scale rule restarts on demand).")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
