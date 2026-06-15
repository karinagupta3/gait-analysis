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


INDEX_TMPL = """<!doctype html><html><head><meta charset="utf-8"><title>Gait Analysis</title>
<style>body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:30px auto;padding:0 16px;color:#222}}
h1{{margin-bottom:4px}} .sub{{color:#666;margin-bottom:24px}} form{{background:#f7f9fb;border:1px solid #e6eaee;border-radius:10px;padding:18px}}
label{{display:block;margin:10px 0 4px;font-weight:600}} input{{padding:7px;border:1px solid #cdd5dc;border-radius:6px;width:100%}}
button{{margin-top:16px;background:#2a9d8f;color:#fff;border:0;border-radius:7px;padding:10px 18px;font-size:15px;cursor:pointer}}
.s{{display:block;padding:10px 12px;border:1px solid #eee;border-radius:8px;margin:8px 0;text-decoration:none;color:#222}}
.s:hover{{background:#f4f6f8}} .s small{{color:#777}}</style></head><body>
<h1>Gait Analysis</h1><div class="sub">Upload an OpenSim <code>.mot</code> to generate a clinical report,
or <a href="/process">process a video &rarr;</a></div>
<form action="/report" method="post" enctype="multipart/form-data">
  <label>Subject</label><input name="subject" placeholder="e.g. J. Smith">
  <label>Trial label</label><input name="trial" placeholder="e.g. squats / overground walk">
  <label>Gait speed (m/s, optional)</label><input name="speed" type="number" step="0.01" placeholder="1.2">
  <label>OpenSim .mot file</label><input name="mot" type="file" accept=".mot,.sto" required>
  <button type="submit">Generate report</button>
</form>
<h2>Recent trials</h2>{sessions}
</body></html>"""


def _render_index() -> str:
    sessions = _list_sessions()
    if not sessions:
        rows = "<p style='color:#777'>No trials yet.</p>"
    else:
        rows = "".join(
            f'<a class="s" href="/session/{s["id"]}"><b>{s.get("subject","?")}</b> '
            f'&middot; {s.get("trial","trial")} <small>&middot; {s.get("created","")[:16]}</small></a>'
            for s in sessions)
    return INDEX_TMPL.format(sessions=rows)


_PAGE_CSS = """body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:30px auto;padding:0 16px;color:#222}
form{background:#f7f9fb;border:1px solid #e6eaee;border-radius:10px;padding:18px}label{display:block;margin:10px 0 4px;font-weight:600}
input,select{padding:7px;border:1px solid #cdd5dc;border-radius:6px;width:100%}button{margin-top:16px;background:#2a9d8f;color:#fff;border:0;border-radius:7px;padding:10px 18px;cursor:pointer}
.note{color:#777;font-size:13px}.ok{color:#1e8449}.bad{color:#c0392b}
.banner{padding:10px 14px;border-radius:8px;margin:14px 0;border:1px solid #e6eaee}
.steps{background:#fbfcfd;border:1px solid #eef1f4;border-radius:10px;padding:8px 18px;margin:14px 0}
.steps h3{margin:12px 0 4px}.steps li{margin:3px 0}"""

# Capture protocol (after the OpenCap recording guidance + Pose2Sim multi-cam notes).
CAPTURE_HTML = """<div class="steps">
<h3>How to record (read before capturing)</h3>
<p class="note">Good kinematics start with good video. Same rules as OpenCap-style capture.</p>
<b>Monocular (1 phone) &mdash; quick mode</b>
<ul>
<li><b>Placement:</b> phone on a tripod, ~3&ndash;4 m from the subject, lens at hip height, landscape.</li>
<li><b>View:</b> for <b>walking</b> film the <b>side</b> (sagittal) view; for <b>squats / sit-to-stand</b> film a 3/4 front view so both knees are visible.</li>
<li><b>Frame:</b> keep the <b>whole body in frame</b> for the entire movement; don't pan/zoom.</li>
<li><b>Quality:</b> 60 fps if possible, good even lighting, plain background, fitted clothing, shoes/feet visible.</li>
<li><b>Content:</b> walking &rarr; 4&ndash;6 strides; squats &rarr; 3&ndash;5 reps; sit-to-stand &rarr; 5 rises (for 5xSTS).</li>
</ul>
<b>Two phones &mdash; accurate mode (Pose2Sim)</b>
<ul>
<li>Two phones ~60&deg; apart, both seeing the whole body; <b>genlock not needed</b> but start both before the subject moves.</li>
<li>Record a <b>calibration object</b> (checkerboard) visible to both cameras first &mdash; required for metric (real-world) scale.</li>
<li>2-phone runs on the CLI today (<code>gait-pipeline</code>); in-app upload is on the roadmap (A3).</li>
</ul></div>"""

PROCESS_TMPL = """<!doctype html><html><head><meta charset="utf-8"><title>Process video</title>
<style>{css}</style></head><body>
<h1>Process a video</h1>
{banner}
{capture}
<form action="/process" method="post" enctype="multipart/form-data">
  <label>Subject</label><input name="subject" placeholder="e.g. J. Smith">
  <label>Trial label</label><input name="trial" placeholder="e.g. squats / overground walk">
  <label>Task hint (optional)</label><input name="trial_hint" placeholder="walking | squat | sit-to-stand">
  <label>Gait speed (m/s, optional)</label><input name="speed" type="number" step="0.01">
  <label>Mode</label><select name="mode"><option value="quick">quick (1 phone)</option></select>
  <label>Video</label><input name="video" type="file" accept="video/*" required>
  <button type="submit">Upload &amp; process</button>
</form>
<p><a href="/">&larr; back</a> &middot; <a href="/setup">environment setup</a></p></body></html>"""

SETUP_TMPL = """<!doctype html><html><head><meta charset="utf-8"><title>Setup</title>
<style>{css} pre{{background:#f4f6f8;padding:10px;border-radius:8px;white-space:pre-wrap}}</style></head><body>
<h1>Environment setup</h1>
{banner}
<h2>Adding OpenSim (needed for joint angles)</h2>
<p class="note">Reports from an existing <code>.mot</code> work without OpenSim. Processing a <b>video</b>
into joint angles needs OpenSim + a marked model.</p>
<ol>
<li>Install <a href="https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/">OpenSim</a> via conda:
<pre>conda create -n gait python=3.11
conda activate gait
conda install -c opensim-org opensim
pip install -e ".[web]" mediapipe</pre></li>
<li>Build a marked model (Track B) and point the app at it:
<pre>gait-inspect-model --model LaiUhlrich2022.osim   # check names
gait-build-model --base LaiUhlrich2022.osim --out LaiUhlrich2022_ga.osim
export GAIT_OSIM_MODEL=$PWD/LaiUhlrich2022_ga.osim
gait-web</pre></li>
<li>Validate it against your Pose2Sim output before trusting it:
<pre>gait-validate --ref pose2sim.mot --test quick.mot   # sagittal RMSE &le; ~5&deg;</pre></li>
</ol>
<p class="note">See <code>docs/05</code> (Track B) and <code>docs/10</code> (setup &amp; capture).</p>
<p><a href="/">&larr; back</a></p></body></html>"""

JOB_TMPL = """<!doctype html><html><head><meta charset="utf-8"><title>Processing</title>
<script>
async function poll(){{const r=await fetch('/api/job/{jid}');const j=await r.json();
document.getElementById('st').textContent=j.state;
document.getElementById('log').textContent=(j.log||[]).join('\\n');
if(j.state==='done'){{location.href='/session/'+j.session_id;}}
else if(j.state==='error'){{document.getElementById('err').textContent=j.error;}}
else{{setTimeout(poll,1500);}}}}
window.onload=poll;</script>
<style>body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:40px auto;padding:0 16px}}
pre{{background:#f4f6f8;padding:10px;border-radius:8px;white-space:pre-wrap}} .err{{color:#c0392b}}</style></head><body>
<h1>Processing&hellip; <span id="st">queued</span></h1>
<p>This can take a few minutes (pose estimation + OpenSim). The page will redirect when done.</p>
<div class="err" id="err"></div><pre id="log"></pre>
<p><a href="/">&larr; back</a></p></body></html>"""


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
    job.log.append(f"mode={meta.get('mode')}: starting")
    if meta.get("mode") != "quick":
        raise RuntimeError("Only quick (1-phone) mode is wired into the app; "
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
                          trial: str = Form(""), speed: str = Form("")):
        sid = uuid.uuid4().hex[:8]
        sdir = _store_dir() / sid
        sdir.mkdir(parents=True, exist_ok=True)
        mot_path = sdir / "trial.mot"
        mot_path.write_bytes(await mot.read())
        spd = float(speed) if speed.strip() else None
        html_path = sdir / "report.html"
        title = f"{subject or 'Subject'} : {trial or 'trial'}"
        report.build_html_report(mot_path, html_path, gait_speed_m_s=spd,
                                 subject=subject or None, title=title)
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
        return PROCESS_TMPL.format(css=_PAGE_CSS, banner=_status_banner(), capture=CAPTURE_HTML)

    @app.get("/setup", response_class=HTMLResponse)
    def setup_page():
        return SETUP_TMPL.format(css=_PAGE_CSS, banner=_status_banner())

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
        return JOB_TMPL.format(jid=jid)

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
