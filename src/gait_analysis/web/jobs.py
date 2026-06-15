"""Tiny in-process background job manager for slow processing (video -> .mot -> report).

Processing takes minutes (MediaPipe/Pose2Sim/OpenSim), so it runs on a daemon thread
and the UI polls for status. In-process and file-backed results -- fine for a local /
single-instance app; swap for a real queue (Celery/Azure jobs) when scaling out.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    state: str = "queued"        # queued | running | done | error
    error: str = ""
    session_id: str = ""
    log: list[str] = field(default_factory=list)


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def submit(self, target) -> str:
        """target(job) -> session_id. Runs on a daemon thread."""
        jid = uuid.uuid4().hex[:8]
        job = Job(jid)
        self._jobs[jid] = job

        def run():
            job.state = "running"
            try:
                job.session_id = target(job) or ""
                job.state = "done"
            except Exception as exc:
                job.error = f"{type(exc).__name__}: {exc}"
                job.log.append(traceback.format_exc().splitlines()[-1])
                job.state = "error"

        threading.Thread(target=run, daemon=True).start()
        return jid

    def get(self, jid: str) -> Job | None:
        return self._jobs.get(jid)
