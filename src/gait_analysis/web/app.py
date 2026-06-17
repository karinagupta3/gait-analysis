"""FastAPI web app: upload an OpenSim .mot, get the clinical report in the browser.

Runs locally (`gait-web`) and is containerizable for cloud. The heavy video->.mot
processing (Pose2Sim + OpenSim) is intentionally NOT here -- it runs where OpenSim is
installed; this app consumes the resulting .mot. Subject/session/trial storage is a
simple on-disk store so trials can be revisited and compared.

Install: pip install -e ".[web]"   Run: gait-web   (then open http://127.0.0.1:8000)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
import uuid
from pathlib import Path

from ..analysis import report
from .jobs import JobManager

# Lazy/guarded FastAPI import so the package imports without the web extra.
try:
    from fastapi import FastAPI, Form, Request, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
except ImportError:  # pragma: no cover - exercised only without the extra
    FastAPI = None  # type: ignore


# On-disk store: DATA_DIR/sessions/<id>/{meta.json, trial.mot, report.html}
# Override with GAIT_STORE_DIR (e.g. /tmp on read-only serverless filesystems).
DATA_DIR = Path(os.environ.get("GAIT_STORE_DIR", Path(__file__).resolve().parent / "_data"))


def _store_dir() -> Path:
    d = DATA_DIR / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _list_sessions() -> list[dict]:
    out = []
    for meta in sorted(_store_dir().glob("*/meta.json")):
        try:
            out.append(json.loads(meta.read_text()))
        except Exception:
            continue
    return sorted(out, key=lambda m: m.get("created", ""), reverse=True)


BASE_CSS = """
:root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--bg:#f8fafc;--card:#fff;
 --accent:#0d9488;--accent-d:#0f766e;--accent-soft:#f0fdfa;--ok:#15803d;--bad:#b91c1c;--radius:14px}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--accent-d);text-decoration:none}a:hover{text-decoration:underline}
header{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.85);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.bar{max-width:900px;margin:0 auto;padding:12px 20px;display:flex;align-items:center;justify-content:space-between}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;color:var(--ink);font-size:16px}
.brand:hover{text-decoration:none}
.logo{width:22px;height:22px;border-radius:7px;background:linear-gradient(135deg,var(--accent),#22d3ee)}
nav{display:flex;gap:2px}
nav a{color:var(--muted);font-size:14px;padding:6px 12px;border-radius:8px}
nav a:hover{color:var(--ink);background:var(--bg);text-decoration:none}
nav a.active{color:var(--accent-d);background:var(--accent-soft)}
main{max-width:900px;margin:0 auto;padding:28px 20px 48px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
h2.section{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:32px 0 12px;font-weight:600}
.lead{color:var(--muted);margin:0}.hero{margin-bottom:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:22px;box-shadow:0 1px 2px rgba(16,24,40,.04)}
label{display:block;font-size:13px;font-weight:600;color:#334155;margin:14px 0 6px}
.card form>label:first-of-type,.row2 label{margin-top:0}
input,select{width:100%;padding:10px 12px;font:inherit;color:var(--ink);background:#fff;border:1px solid var(--line);border-radius:9px;outline:none;transition:border-color .15s,box-shadow .15s}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.btn{display:inline-flex;align-items:center;gap:8px;margin-top:18px;background:var(--accent);color:#fff;border:0;border-radius:10px;padding:11px 20px;font:inherit;font-weight:600;cursor:pointer;transition:background .15s}
.btn:hover{background:var(--accent-d)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:560px){.row2{grid-template-columns:1fr}}
.slist{display:flex;flex-direction:column;gap:8px}
.s{display:flex;align-items:center;justify-content:space-between;gap:12px;background:var(--card);border:1px solid var(--line);border-radius:11px;padding:13px 16px;color:var(--ink)}
.s:hover{border-color:#cbd5e1;text-decoration:none;box-shadow:0 1px 3px rgba(16,24,40,.06)}
.s .who{font-weight:600}.s .meta{color:var(--muted);font-size:13px}
.empty{color:var(--muted);background:var(--card);border:1px dashed var(--line);border-radius:11px;padding:22px;text-align:center}
.banner{border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:0 0 18px}
.banner b{display:block;margin-bottom:6px}.ok{color:var(--ok)}.bad{color:var(--bad)}
.steps{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:4px 20px;margin:0 0 18px}
.steps h3{margin:14px 0 6px;font-size:15px}.steps ul{margin:6px 0 12px;padding-left:20px}.steps li{margin:4px 0}
.note{color:var(--muted);font-size:13px}
code{background:#f1f5f9;padding:1px 6px;border-radius:5px;font-size:13px}
pre{background:#0f172a;color:#e2e8f0;padding:14px 16px;border-radius:10px;overflow:auto;font-size:13px;line-height:1.5}
pre code{background:none;color:inherit;padding:0}
.badge{font-size:13px;font-weight:600;color:var(--accent-d);background:var(--accent-soft);padding:2px 12px;border-radius:999px;vertical-align:middle}
.err{color:var(--bad);margin:0 0 14px;font-weight:600}
footer{max-width:900px;margin:0 auto;padding:22px 20px 40px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}
"""


def _shell(title: str, body: str, active: str = "") -> str:
    def nav(href, label, key):
        return f'<a class="{"active" if key == active else ""}" href="{href}">{label}</a>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><style>{BASE_CSS}</style></head><body>'
            f'<header><div class="bar"><a class="brand" href="/"><span class="logo"></span>Gait Analysis</a>'
            f'<nav>{nav("/", "New report", "index")}{nav("/process", "Process video", "process")}'
            f'{nav("/setup", "Setup", "setup")}</nav></div></header>'
            f'<main>{body}</main>'
            f'<footer>Gait Analysis · clinical kinematics from OpenSim</footer></body></html>')


_INDEX_FORM = """<div class="card">
<form action="/report" method="post" enctype="multipart/form-data">
  <div class="row2">
    <div><label>Subject</label><input name="subject" placeholder="e.g. J. Smith"></div>
    <div><label>Trial label</label><input name="trial" placeholder="e.g. overground walk"></div>
  </div>
  <label>Gait speed (m/s, optional)</label><input name="speed" type="number" step="0.01" placeholder="1.2">
  <label>OpenSim .mot file</label><input name="mot" type="file" accept=".mot,.sto" required>
  <label>Marker .trc (optional &mdash; enables 3D playback + metric spatiotemporal)</label>
  <input name="trc" type="file" accept=".trc">
  <button class="btn" type="submit">Generate report</button>
</form></div>"""


def _render_index() -> str:
    sessions = _list_sessions()
    if not sessions:
        rows = '<div class="empty">No trials yet &mdash; upload a <code>.mot</code> to get started.</div>'
    else:
        rows = "".join(
            f'<a class="s" href="/session/{s["id"]}">'
            f'<span><span class="who">{s.get("subject") or "Subject"}</span>'
            f'<span class="meta"> &middot; {s.get("trial") or "trial"}</span></span>'
            f'<span class="meta">{s.get("created","")[:16].replace("T"," ")}</span></a>'
            for s in sessions)
    body = (f'<section class="hero"><h1>Gait Analysis</h1>'
            f'<p class="lead">Upload an OpenSim <code>.mot</code> to generate a clinical report, '
            f'or <a href="/process">process a video</a>.</p></section>'
            f'{_INDEX_FORM}'
            f'<h2 class="section">Recent trials</h2><div class="slist">{rows}</div>')
    return _shell("Gait Analysis", body, "index")


# Capture protocol (after OpenCap recording guidance + Pose2Sim multi-cam notes).
CAPTURE_HTML = """<div class="steps">
<h3>How to record (read before capturing)</h3>
<p class="note">Good kinematics start with good video &mdash; same rules as OpenCap-style capture.</p>
<b>Monocular (1 phone) &mdash; quick mode</b>
<ul>
<li><b>Placement:</b> phone on a tripod, ~3&ndash;4 m from the subject, lens at hip height, landscape.</li>
<li><b>View:</b> for <b>walking</b> film the <b>side</b> (sagittal) view; for <b>squats / sit-to-stand</b> film a 3/4 front view so both knees are visible.</li>
<li><b>Frame:</b> keep the <b>whole body in frame</b> for the entire movement; don't pan or zoom.</li>
<li><b>Quality:</b> 60 fps if possible, even lighting, plain background, fitted clothing, feet visible.</li>
<li><b>Content:</b> walking &rarr; 4&ndash;6 strides; squats &rarr; 3&ndash;5 reps; sit-to-stand &rarr; 5 rises (for 5xSTS).</li>
</ul>
<b>Two phones &mdash; accurate mode (Pose2Sim)</b>
<ul>
<li>Two phones ~60&deg; apart, both seeing the whole body; <b>genlock not needed</b>, but start both before the subject moves.</li>
<li>Record a <b>calibration object</b> (checkerboard) visible to both cameras first &mdash; required for metric scale.</li>
<li>2-phone runs on the CLI today (<code>gait-pipeline</code>); in-app upload is on the roadmap (A3).</li>
</ul></div>"""


def _process_body() -> str:
    return (f'<section class="hero"><h1>Process a video</h1>'
            f'<p class="lead">Single-phone quick mode: video &rarr; pose estimation &rarr; OpenSim &rarr; clinical report.</p></section>'
            f'{_status_banner()}{CAPTURE_HTML}'
            f'<div class="card"><form action="/process" method="post" enctype="multipart/form-data">'
            f'<div class="row2"><div><label>Subject</label><input name="subject" placeholder="e.g. J. Smith"></div>'
            f'<div><label>Trial label</label><input name="trial" placeholder="e.g. squats"></div></div>'
            f'<label>Task hint (optional)</label><input name="trial_hint" placeholder="walking | squat | sit-to-stand">'
            f'<label>Gait speed (m/s, optional)</label><input name="speed" type="number" step="0.01">'
            f'<label>Mode</label><select name="mode">'
            f'<option value="screening">2D screening (1 phone, side view)</option>'
            f'<option value="quick">3D quick (needs OpenSim model)</option></select>'
            f'<label>Video</label><input name="video" type="file" accept="video/*" required>'
            f'<button class="btn" type="submit">Upload &amp; process</button></form></div>')


def _setup_body() -> str:
    return (f'<section class="hero"><h1>Environment setup</h1>'
            f'<p class="lead">Reports from an existing <code>.mot</code> work out of the box. '
            f'Turning a <b>video</b> into joint angles needs OpenSim + a marked model.</p></section>'
            f'{_status_banner()}'
            '<div class="card"><h3 style="margin-top:6px">Adding OpenSim (for video &rarr; joint angles)</h3>'
            '<p class="note">1. Install OpenSim + the app extras:</p>'
            '<pre><code>conda create -n gait python=3.11\nconda activate gait\n'
            'conda install -c opensim-org opensim\npip install -e ".[web]" mediapipe</code></pre>'
            '<p class="note">2. Build a marked model and point the app at it via <code>GAIT_OSIM_MODEL</code>:</p>'
            '<pre><code>gait-build-model --base LaiUhlrich2022.osim --out LaiUhlrich2022_ga.osim\n'
            'export GAIT_OSIM_MODEL=$PWD/LaiUhlrich2022_ga.osim\ngait-web</code></pre>'
            '<p class="note">3. Validate it against your Pose2Sim output before trusting it '
            '(<code>gait-validate</code>; sagittal RMSE &le; ~5&deg;).</p></div>')


_JOB_BODY = """<section class="hero"><h1>Processing&hellip; <span id="st" class="badge">queued</span></h1>
<p class="lead">Pose estimation + OpenSim can take a few minutes. This page updates live and opens the report when done.</p></section>
<div class="err" id="err"></div>
<div class="card"><pre id="log">starting&hellip;</pre></div>
<script>
async function poll(){const r=await fetch('/api/job/__JID__');const j=await r.json();
document.getElementById('st').textContent=j.state;
document.getElementById('log').textContent=(j.log||[]).join('\\n');
if(j.state==='done'){location.href='/session/'+j.session_id;}
else if(j.state==='error'){document.getElementById('err').textContent=j.error;document.getElementById('st').textContent='error';}
else{setTimeout(poll,1500);}}
window.onload=poll;</script>"""


def _capabilities() -> dict:
    """What's installed/configured for video processing (so the UI can tell the user)."""
    import importlib.util
    model = os.environ.get("GAIT_OSIM_MODEL")
    return {
        "opensim": importlib.util.find_spec("opensim") is not None,
        "mediapipe": importlib.util.find_spec("mediapipe") is not None,
        "model": model if (model and Path(model).exists()) else None,
        "model_set": bool(model),
    }


def _status_banner() -> str:
    c = _capabilities()

    def row(ok, label, hint):
        mark = "&#10003;" if ok else "&#10007;"
        return f"<div class='{'ok' if ok else 'bad'}'>{mark} {label}{'' if ok else ' &mdash; ' + hint}</div>"

    ready = c["opensim"] and c["model"] and c["mediapipe"]
    rows = (row(c["opensim"], "OpenSim installed", "see setup below")
            + row(c["mediapipe"], "MediaPipe installed", "pip install mediapipe")
            + row(bool(c["model"]), "Marked model (GAIT_OSIM_MODEL)",
                  "file missing" if c["model_set"] else "not set"))
    head = ("Ready to process video." if ready
            else "Video processing needs setup (you can still upload a .mot directly).")
    bg = "#eafaf1" if ready else "#fdf2e9"
    return f"<div class='banner' style='background:{bg}'><b>{head}</b>{rows}</div>"


def _default_process(job, video_path: Path, sdir: Path, meta: dict) -> str:
    """Real quick-mode processing: video -> MediaPipe 3D -> OpenSim IK -> .mot -> report."""
    mode = meta.get("mode") or "screening"
    job.log.append(f"mode={mode}: starting")

    # Single-phone 2D sagittal screening: no OpenSim/model needed. Runs in a SUBPROCESS
    # so the heavy MediaPipe pass can't starve the uvicorn event loop (keeps the web
    # responsive — /health and status polls stay up while a job runs).
    if mode == "screening":
        import subprocess
        import sys as _sys
        job.log.append("MediaPipe pose -> 2D sagittal screening (subprocess) ...")
        proc = subprocess.run(
            [_sys.executable, "-m", "gait_analysis.web.screening_job",
             str(video_path), str(sdir), meta.get("subject") or ""],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise RuntimeError("screening failed: " + (tail[-1] if tail else "unknown error"))
        if not (sdir / "report.html").exists():
            raise RuntimeError("screening produced no report")
        (sdir / "meta.json").write_text(json.dumps({**meta, "created": _dt.datetime.now().isoformat()}))
        return sdir.name

    if mode != "quick":
        raise RuntimeError("Only screening (2D) and quick (3D) modes are wired into the app; "
                           "use the CLI for accurate 2-phone (Pose2Sim) capture.")
    model = os.environ.get("GAIT_OSIM_MODEL")
    if not model:
        raise RuntimeError("Set GAIT_OSIM_MODEL to your marked .osim (see docs/03).")

    from ..pipeline import run_quick
    job.log.append("running MediaPipe 3D -> .trc -> OpenSim IK ...")
    result = run_quick(video_path, model, sdir, gait_speed_m_s=meta.get("speed"))
    mot = result["mot"]
    job.log.append("building report ...")
    report.build_html_report(mot, sdir / "report.html", gait_speed_m_s=meta.get("speed"),
                             subject=meta.get("subject") or None,
                             title=f"{meta.get('subject') or 'Subject'} : {meta.get('trial') or 'trial'}")
    (sdir / "meta.json").write_text(json.dumps({**meta, "created": _dt.datetime.now().isoformat()}))
    return sdir.name


def create_app(process_fn=None):
    if FastAPI is None:
        raise SystemExit("FastAPI not installed. Run:  pip install -e \".[web]\"")
    app = FastAPI(title="Gait Analysis")
    jm = JobManager()
    process_fn = process_fn or _default_process

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _render_index()

    @app.post("/report", response_class=HTMLResponse)
    async def make_report(mot: UploadFile, subject: str = Form(""),
                          trial: str = Form(""), speed: str = Form(""),
                          trc: UploadFile | None = None):
        sid = uuid.uuid4().hex[:8]
        sdir = _store_dir() / sid
        sdir.mkdir(parents=True, exist_ok=True)
        mot_path = sdir / "trial.mot"
        mot_path.write_bytes(await mot.read())
        trc_path = None
        if trc is not None and (trc.filename or "").strip():
            trc_path = sdir / "markers.trc"
            trc_path.write_bytes(await trc.read())
        spd = float(speed) if speed.strip() else None
        html_path = sdir / "report.html"
        title = f"{subject or 'Subject'} : {trial or 'trial'}"
        report.build_html_report(mot_path, html_path, gait_speed_m_s=spd,
                                 subject=subject or None, title=title, trc_path=trc_path)
        meta = {"id": sid, "subject": subject, "trial": trial, "speed": spd,
                "created": _dt.datetime.now().isoformat()}
        (sdir / "meta.json").write_text(json.dumps(meta))
        return RedirectResponse(f"/session/{sid}", status_code=303)

    @app.get("/session/{sid}", response_class=HTMLResponse)
    def session(sid: str):
        html_path = _store_dir() / sid / "report.html"
        if not html_path.exists():
            return HTMLResponse("<p>Not found. <a href='/'>Back</a></p>", status_code=404)
        return html_path.read_text()

    @app.get("/api/sessions")
    def api_sessions():
        return JSONResponse(_list_sessions())

    # --- video -> processing -> report flow ---

    @app.get("/process", response_class=HTMLResponse)
    def process_form():
        return _shell("Process video", _process_body(), "process")

    @app.get("/setup", response_class=HTMLResponse)
    def setup_page():
        return _shell("Setup", _setup_body(), "setup")

    @app.post("/process")
    async def process(video: UploadFile, subject: str = Form(""), trial: str = Form(""),
                      trial_hint: str = Form(""), speed: str = Form(""), mode: str = Form("quick")):
        sid = uuid.uuid4().hex[:8]
        sdir = _store_dir() / sid
        sdir.mkdir(parents=True, exist_ok=True)
        suffix = Path(video.filename or "input.mov").suffix or ".mov"
        video_path = sdir / f"input{suffix}"
        video_path.write_bytes(await video.read())
        trial_full = f"{trial} ({trial_hint})".strip() if trial_hint.strip() else trial
        meta = {"id": sid, "subject": subject, "trial": trial_full,
                "speed": float(speed) if speed.strip() else None, "mode": mode}
        (sdir / "meta.json").write_text(json.dumps({**meta, "state": "processing"}))
        jid = jm.submit(lambda job: process_fn(job, video_path, sdir, meta))
        return RedirectResponse(f"/job/{jid}", status_code=303)

    @app.get("/job/{jid}", response_class=HTMLResponse)
    def job_page(jid: str):
        if jm.get(jid) is None:
            return HTMLResponse("<p>Unknown job. <a href='/'>Back</a></p>", status_code=404)
        return _shell("Processing", _JOB_BODY.replace("__JID__", jid), "process")

    @app.get("/api/job/{jid}")
    def api_job(jid: str):
        job = jm.get(jid)
        if job is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        return JSONResponse({"state": job.state, "error": job.error,
                             "session_id": job.session_id, "log": job.log})

    return app


def run():  # console entry point: gait-web
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
