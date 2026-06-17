"""Server-side TWO-PHONE session staging + matching for accurate (Pose2Sim) mode.

The accurate pipeline (Track A) needs FOUR clips that belong to ONE capture session:

    cam1 calibration  +  cam2 calibration   (the ChArUco/checkerboard board)
    cam1 trial        +  cam2 trial         (the actual walk)

Two phones can't upload as a single multipart form, so we let each phone POST its
own clip independently and stitch them together by a short shared SESSION CODE that
both operators type in. This module is the staging + matching + assembly layer that
turns those four independent uploads into the on-disk project tree that Pose2Sim's
``pose2sim_runner`` expects, then drives the existing accurate pipeline + report.

Directory layout this module produces (learned from
``biomech/pose2sim_runner.py`` ``prepare_project``/``run`` and the task spec):

    <project>/
      Config.toml                 # written by pose2sim_runner.prepare_project
      calibration/
        cam1/<calibration clip>   # board video for camera 1
        cam2/<calibration clip>   # board video for camera 2
      videos/
        cam1/<trial clip>         # walk video for camera 1
        cam2/<trial clip>         # walk video for camera 2
      pose/ pose-3d/ kinematics/  # created empty by prepare_project; Pose2Sim fills them

Staging layout (before assembly), under the same on-disk store as the web app:

    <DATA_DIR>/twophone/<code>/
      meta.json                   # session metadata + per-clip records
      staged/<role>_<kind>.<ext>  # e.g. cam1_trial.mov, cam2_calibration.mov
      project/                    # assembled Pose2Sim project (built lazily)

Design notes / honesty:
  * Heavy deps (Pose2Sim, OpenSim, MediaPipe, the report's matplotlib path) are all
    lazy-imported -- importing this module is cheap and works without the [web]
    extra or any biomech stack installed.
  * This does NOT edit app.py. Wiring it into routes is left to the caller; see the
    StructuredOutput / module docstring for the exact functions to call.
  * UNVALIDATED end-to-end. The assemble/match/staging logic is exercisable today,
    but ``run_session`` calls Pose2Sim, which needs OpenSim + real synchronized
    two-phone footage WITH a calibration board to produce a meaningful .mot. There
    is no such footage in this repo, so the full path has only been reasoned through
    against pose2sim_runner.run(), not actually run. Two open questions that ONLY
    real footage can answer: (1) whether Pose2Sim picks up per-camera subfolders
    named exactly ``cam1``/``cam2`` under both ``videos/`` and ``calibration/`` for
    this installed version (folder naming has drifted across Pose2Sim releases --
    some versions expect ``calibration/intrinsics/<cam>`` + ``calibration/extrinsics/<cam>``
    rather than a flat ``calibration/<cam>``), and (2) whether the 'sound' clap-sync
    in the starter Config.toml actually aligns two independently-started phone clips.
    Both must be confirmed against an installed Pose2Sim + a real capture before this
    is trusted clinically. Comments below flag the exact spots to revisit.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
import string
from pathlib import Path

# Mirror app.py's DATA_DIR convention so staging lands in the SAME store the rest of
# the web app uses (and is overridable for read-only/serverless filesystems).
# We default to <this package dir>/_data just like app.py does, so a deployment that
# sets GAIT_STORE_DIR gets one consistent store across modules.
DATA_DIR = Path(os.environ.get("GAIT_STORE_DIR", Path(__file__).resolve().parent / "_data"))

# A two-phone trial is exactly these four clips. role x kind is the 2x2 matrix.
ROLES = ("cam1", "cam2")
KINDS = ("calibration", "trial")
# All four (role, kind) pairs that must be present for a session to be runnable.
REQUIRED_CLIPS = tuple((r, k) for r in ROLES for k in KINDS)

# Session codes: 6 chars, uppercase letters + digits, but WITHOUT the ambiguous
# 0/O/1/I/L so two people reading them aloud / typing on phones don't collide.
_CODE_ALPHABET = "".join(c for c in (string.ascii_uppercase + string.digits)
                         if c not in "O0I1L")
_CODE_LEN = 6

# Allowed upload extensions (defensive: these become on-disk filenames). Phones tend
# to produce .mov/.mp4; the in-browser recorder (see app.py _RECORD_BODY) makes
# webm/mp4. Keep this permissive but bounded.
_ALLOWED_EXTS = {"mov", "mp4", "m4v", "webm", "avi", "mkv"}


# --- code + paths ------------------------------------------------------------

def new_session_code() -> str:
    """Return a fresh 6-char session code (unambiguous alphabet).

    Best-effort uniqueness: re-rolls if a staging dir for the code already exists.
    Not cryptographically load-bearing -- it's a human-typed pairing token, not a
    secret -- but ``secrets`` keeps it unguessable enough that someone can't easily
    drop a clip into a stranger's in-progress session.
    """
    base = _twophone_root()
    for _ in range(50):
        code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LEN))
        if not (base / code).exists():
            return code
    # Astronomically unlikely; surface rather than silently collide.
    raise RuntimeError("could not allocate a unique session code")


def _twophone_root() -> Path:
    d = DATA_DIR / "twophone"
    d.mkdir(parents=True, exist_ok=True)
    return d


def staging_dir(code: str) -> Path:
    """Return (creating) the staging directory for a session code.

    Layout: <DATA_DIR>/twophone/<code>/  with a staged/ subfolder for the raw clips.
    """
    code = _safe_code(code)
    d = _twophone_root() / code
    (d / "staged").mkdir(parents=True, exist_ok=True)
    return d


def _safe_code(code: str) -> str:
    """Validate/normalize a code so it can never escape the staging root."""
    code = (code or "").strip().upper()
    if len(code) != _CODE_LEN or any(c not in _CODE_ALPHABET for c in code):
        raise ValueError(
            f"invalid session code {code!r}: expected {_CODE_LEN} chars from the "
            f"unambiguous alphabet"
        )
    return code


def _meta_path(code: str) -> Path:
    return staging_dir(code) / "meta.json"


def _load_meta(code: str) -> dict:
    p = _meta_path(code)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"code": _safe_code(code), "created": _dt.datetime.now().isoformat(),
            "clips": {}}


def _save_meta(code: str, meta: dict) -> None:
    _meta_path(code).write_text(json.dumps(meta, indent=2))


def _clip_key(role: str, kind: str) -> str:
    return f"{role}_{kind}"


# --- staging clips -----------------------------------------------------------

def save_clip(code: str, role: str, kind: str, data: bytes, ext: str) -> dict:
    """Stage one of the four clips for a two-phone session.

    role  in {cam1, cam2}; kind in {calibration, trial}. ``data`` is the raw upload
    bytes, ``ext`` the file extension (with or without a leading dot). Overwrites any
    prior clip for the same (role, kind) so a re-record replaces cleanly.

    Returns the per-clip record (path/size/ext/received). Records into meta.json so
    ``session_status`` can report progress and ``assemble_project`` can find the file.
    """
    code = _safe_code(code)
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    ext = (ext or "").lstrip(".").lower()
    if ext not in _ALLOWED_EXTS:
        raise ValueError(f"unsupported clip extension {ext!r}; allowed: {sorted(_ALLOWED_EXTS)}")
    if not data:
        raise ValueError("empty clip data")

    sdir = staging_dir(code)
    fname = f"{_clip_key(role, kind)}.{ext}"
    dest = sdir / "staged" / fname
    dest.write_bytes(data)

    meta = _load_meta(code)
    record = {
        "role": role,
        "kind": kind,
        "ext": ext,
        "filename": fname,
        "path": str(dest),
        "size": len(data),
        "received": _dt.datetime.now().isoformat(),
    }
    meta.setdefault("clips", {})[_clip_key(role, kind)] = record
    _save_meta(code, meta)
    return record


def session_status(code: str) -> dict:
    """Report which of the four clips are present and whether the session is ready.

    Returns:
        {
          "code": <code>,
          "created": <iso>,
          "present": {"cam1_calibration": bool, ... all four ...},
          "clips": {<key>: <record>, ...},   # only for staged clips
          "missing": [<key>, ...],           # human-readable, e.g. "cam2_trial"
          "ready": bool,                     # True iff all four are staged
        }
    """
    code = _safe_code(code)
    meta = _load_meta(code)
    clips = meta.get("clips", {})
    present = {}
    missing = []
    for role, kind in REQUIRED_CLIPS:
        key = _clip_key(role, kind)
        # A clip counts as present only if its file is actually on disk (guards
        # against a meta.json that references a clip that was later deleted).
        rec = clips.get(key)
        ok = bool(rec) and Path(rec.get("path", "")).exists()
        present[key] = ok
        if not ok:
            missing.append(key)
    return {
        "code": code,
        "created": meta.get("created"),
        "present": present,
        "clips": clips,
        "missing": missing,
        "ready": not missing,
    }


# --- assembly into the Pose2Sim project tree --------------------------------

def assemble_project(code: str) -> Path:
    """Build the Pose2Sim project tree from the four staged clips; return its dir.

    Creates <staging>/project with the folder layout pose2sim_runner expects
    (delegating folder + Config.toml creation to prepare_project), then drops the
    staged clips into per-camera subfolders:

        project/videos/cam1/<trial clip>        project/videos/cam2/<trial clip>
        project/calibration/cam1/<board clip>   project/calibration/cam2/<board clip>

    Raises if any of the four clips is missing (call session_status first).

    NOTE (needs real footage to confirm): the per-camera subfolder naming below is
    the documented "one folder per camera" Pose2Sim convention, but the EXACT names
    and depth this layer uses (flat ``calibration/cam1`` etc.) must be checked against
    the installed Pose2Sim version -- some versions split calibration into
    ``calibration/intrinsics/<cam>`` + ``calibration/extrinsics/<cam>``. If
    Pose2Sim.calibration() can't find the board clips, this is the first thing to fix.
    """
    code = _safe_code(code)
    status = session_status(code)
    if not status["ready"]:
        raise RuntimeError(
            f"session {code} is not complete; missing clips: {status['missing']}. "
            f"All four (cam1/cam2 x calibration/trial) must be staged first."
        )

    sdir = staging_dir(code)
    project = sdir / "project"

    # Lazy import: only pull in the biomech package when actually assembling, so
    # importing twophone stays cheap and dependency-free.
    from ..biomech import pose2sim_runner
    pose2sim_runner.prepare_project(project)  # makes calibration/ videos/ ... + Config.toml

    clips = status["clips"]
    for role in ROLES:
        # Trial clips -> videos/<cam>/ ; calibration clips -> calibration/<cam>/.
        trial = clips[_clip_key(role, "trial")]
        calib = clips[_clip_key(role, "calibration")]

        vids = project / "videos" / role
        cals = project / "calibration" / role
        vids.mkdir(parents=True, exist_ok=True)
        cals.mkdir(parents=True, exist_ok=True)

        _place(Path(trial["path"]), vids / f"{role}_trial.{trial['ext']}")
        _place(Path(calib["path"]), cals / f"{role}_calibration.{calib['ext']}")

    # Record where we assembled, for run_session / debugging.
    meta = _load_meta(code)
    meta["project_dir"] = str(project)
    meta["assembled"] = _dt.datetime.now().isoformat()
    _save_meta(code, meta)
    return project


def _place(src: Path, dest: Path) -> None:
    """Copy a staged clip into the project tree (copy, not move, so a failed run can
    be retried from the still-staged originals). Replaces any existing dest."""
    import shutil
    if not src.exists():
        raise FileNotFoundError(f"staged clip missing: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.copy2(src, dest)


# --- run the accurate pipeline + build the report ---------------------------

def run_session(code: str, gait_speed_m_s: float | None = None,
                subject: str | None = None, trial: str | None = None) -> dict:
    """Assemble + run the accurate (Pose2Sim) pipeline for a session, then report.

    Steps:
      1. assemble_project(code)  -> Pose2Sim project tree from the staged clips
      2. pipeline.run_accurate(project)  -> drives Pose2Sim (calibration ->
         triangulation -> OpenSim IK) and our kinematics report + signature flags,
         returning the .mot path.
      3. report.build_html_report(.mot)  -> the same self-contained HTML report the
         rest of the app serves, written into the staging dir so the web layer can
         surface it by session code.

    Returns: {"code", "project_dir", "mot", "report", "result"} where ``report`` is
    the path to report.html (or None if report generation was skipped).

    HEAVY + UNVALIDATED: this is the only function here that needs Pose2Sim + OpenSim
    installed AND real two-phone footage to do anything meaningful. With no such
    footage in the repo, this path has been reasoned against pipeline.run_accurate()
    but not executed end-to-end. Lazy-imports everything so the failure is at call
    time (with pose2sim_runner's loud "pip install pose2sim" guidance), never at
    import time.
    """
    code = _safe_code(code)
    project = assemble_project(code)

    # Lazy import the orchestrator (pulls biomech/pose/analysis only when run).
    from .. import pipeline
    result = pipeline.run_accurate(project, gait_speed_m_s=gait_speed_m_s)
    mot = result.get("mot")

    report_path = None
    if mot is not None:
        # Reuse the existing report builder, same call shape app.py uses, writing into
        # the staging dir so a web route can serve /twophone/<code>/report.
        from ..analysis import report
        sdir = staging_dir(code)
        title = f"{subject or 'Subject'} : {trial or 'two-phone trial'}"
        report_path = report.build_html_report(
            mot, sdir / "report.html", gait_speed_m_s=gait_speed_m_s,
            subject=subject or None, title=title,
        )
        # Persist outcome to meta.json for status/listing.
        meta = _load_meta(code)
        meta.update({
            "mot": str(mot),
            "report": str(report_path),
            "subject": subject,
            "trial": trial,
            "speed": gait_speed_m_s,
            "completed": _dt.datetime.now().isoformat(),
        })
        _save_meta(code, meta)

    return {
        "code": code,
        "project_dir": str(project),
        "mot": str(mot) if mot is not None else None,
        "report": str(report_path) if report_path is not None else None,
        "result": result,
    }
