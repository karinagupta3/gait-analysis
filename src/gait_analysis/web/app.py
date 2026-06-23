"""FastAPI web app: upload an OpenSim .mot, get the clinical report in the browser.

Runs locally (`gait-web`) and is containerizable for cloud. The heavy video->.mot
processing (Pose2Sim + OpenSim) is intentionally NOT here -- it runs where OpenSim is
installed; this app consumes the resulting .mot. Subject/session/trial storage is a
simple on-disk store so trials can be revisited and compared.

Install: pip install -e ".[web]"   Run: gait-web   (then open http://127.0.0.1:8000)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from ..analysis import report
from . import dispatch, twophone
from .capture_page import CAPTURE_BODY
from .jobs import JobManager

# Lazy/guarded FastAPI import so the package imports without the web extra.
try:
    from fastapi import FastAPI, Form, Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
except ImportError:  # pragma: no cover - exercised only without the extra
    FastAPI = None  # type: ignore


# On-disk store: DATA_DIR/sessions/<id>/{meta.json, trial.mot, report.html}
# Override with GAIT_STORE_DIR (e.g. /tmp on read-only serverless filesystems).
DATA_DIR = Path(os.environ.get("GAIT_STORE_DIR", Path(__file__).resolve().parent / "_data"))

# Curated, verified full-body SIDE-VIEW walking clips for one-click testing. Hosted in
# OUR private blob container ("blob") so the server can always fetch them — Pexels'/etc.
# CDNs 403 datacenter IPs. "url" is the original source, used only for the optional
# browser "download" link.
SAMPLE_BLOB_CONTAINER = "gait-samples"
SAMPLE_VIDEOS = [
    {"id": "mixkit_man", "label": "Man walking (plaza)", "blob": "mixkit_man.mp4",
     "desc": "Full-body side view, ~12 s — cleanest reference clip.",
     "url": "https://assets.mixkit.co/videos/4855/4855-720.mp4"},
    {"id": "woman_park", "label": "Woman walking (park path)", "blob": "woman_park.mp4",
     "desc": "Full-body side view, ~9 s — tracks cleanly (88% usable).",
     "url": "https://videos.pexels.com/video-files/5535731/5535731-hd_1920_1080_25fps.mp4"},
    {"id": "woman_wall", "label": "Woman walking (by wall)", "blob": "woman_wall.mp4",
     "desc": "Full-body side view, ~8 s.",
     "url": "https://videos.pexels.com/video-files/6414085/6414085-hd_1920_1080_24fps.mp4"},
    {"id": "woman_poppy", "label": "Woman walking (long, ~40 s)", "blob": "woman_poppy.mp4",
     "desc": "Full-body side view, many gait cycles — best for a longer trial.",
     "url": "https://videos.pexels.com/video-files/4812188/4812188-hd_1920_1080_30fps.mp4"},
]
_ALLOWED_SAMPLE_HOSTS = {"assets.mixkit.co", "videos.pexels.com", "upload.wikimedia.org"}


def _sample_by_id(sid: str) -> dict | None:
    return next((s for s in SAMPLE_VIDEOS if s["id"] == sid), None)


# --- Authentication (app-level shared password) ------------------------------
# When GAIT_AUTH_PASSWORD is set, every page requires a signed session cookie
# obtained by entering the password at /login. When it is UNSET, auth is disabled
# (local dev + tests run open). /login and /health stay reachable so the sign-in
# page works and Azure's health probe isn't blocked.
AUTH_PASSWORD = os.environ.get("GAIT_AUTH_PASSWORD") or ""
AUTH_SECRET = (os.environ.get("GAIT_SECRET_KEY") or "dev-insecure-key").encode()
_COOKIE = "gait_session"
_SESSION_TTL = 12 * 3600  # seconds (re-login after a shift)
_OPEN_PATHS = {"/login", "/health"}


def _auth_enabled() -> bool:
    return bool(AUTH_PASSWORD)


def _make_session_token() -> str:
    exp = str(int(time.time()) + _SESSION_TTL)
    sig = hmac.new(AUTH_SECRET, exp.encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def _valid_session(token: str) -> bool:
    try:
        exp, sig = token.split(".", 1)
        good = hmac.new(AUTH_SECRET, exp.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, good) and int(exp) > time.time()
    except Exception:
        return False


_LOGIN_BODY = """<section class="hero"><h1>Sign in</h1>
<p class="lead">This tool is restricted to clinic staff.</p></section>
<!--err-->
<div class="card" style="max-width:420px">
<form action="/login" method="post">
  <label>Password</label><input name="password" type="password" autofocus required>
  <button class="btn" type="submit">Sign in</button>
</form></div>"""


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
.acts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:0 0 8px}@media(max-width:560px){.acts{grid-template-columns:1fr}}
.act{display:block;background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:18px 20px;box-shadow:0 1px 2px rgba(16,24,40,.04);transition:border-color .15s,box-shadow .15s}
.act:hover{border-color:var(--accent);box-shadow:0 2px 12px rgba(13,148,136,.12);text-decoration:none}
.act .t{font-weight:700;color:var(--ink);font-size:16px}
.act .d{color:var(--muted);font-size:13px;margin-top:4px;line-height:1.45}
.act .ic{font-size:20px;margin-right:8px}
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
            f'<header><div class="bar"><a class="brand" href="/"><span class="logo"></span>Gait &amp; Movement</a>'
            f'<nav>{nav("/process", "Process video", "process")}{nav("/record", "Record (phone)", "record")}'
            f'{nav("/capture", "2-phone", "capture")}{nav("/samples", "Samples", "samples")}'
            f'{nav("/setup", "Setup", "setup")}'
            f'{nav("/logout", "Sign out", "") if _auth_enabled() else ""}</nav></div></header>'
            f'<main>{body}</main>'
            f'<footer>Gait &amp; Movement Analysis · screening tool, not a diagnosis</footer></body></html>')


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


def _act(href, icon, title, desc):
    return (f'<a class="act" href="{href}"><div class="t"><span class="ic">{icon}</span>{title}</div>'
            f'<div class="d">{desc}</div></a>')


def _render_index() -> str:
    sessions = _list_sessions()
    acts = (
        '<div class="acts">'
        + _act("/process", "&#127909;", "Process a video",
               "Upload a clip &rarr; pose tracking + a clinical report.")
        + _act("/record", "&#128241;", "Record with a phone",
               "Open on a phone, record, and it processes automatically.")
        + _act("/capture", "&#127919;", "Two-phone 3D",
               "Accurate 3D from two synced phones (advanced).")
        + _act("/samples", "&#9654;", "Try a sample",
               "Run a built-in clip to see what a report looks like.")
        + '</div>')
    if sessions:
        rows = "".join(
            f'<a class="s" href="/session/{s["id"]}">'
            f'<span><span class="who">{s.get("subject") or "Subject"}</span>'
            f'<span class="meta"> &middot; {s.get("trial") or "trial"}</span></span>'
            f'<span class="meta">{s.get("created","")[:16].replace("T"," ")}</span></a>'
            for s in sessions)
        recent = f'<h2 class="section">Recent reports</h2><div class="slist">{rows}</div>'
    else:
        recent = ('<h2 class="section">Recent reports</h2>'
                  '<div class="empty">No reports yet &mdash; start with '
                  '&ldquo;Process a video&rdquo; or &ldquo;Try a sample&rdquo;.</div>')
    body = (f'<section class="hero"><h1>Gait &amp; Movement Analysis</h1>'
            f'<p class="lead">Record or upload a movement and get an objective report. Start here:</p></section>'
            f'{acts}{recent}')
    return _shell("Gait & Movement Analysis", body, "index")


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
    # Only offer the 3D mode where OpenSim is actually available (i.e. not the cloud app),
    # so users aren't given a dead option that always errors.
    mode_opts = '<option value="screening">2D screening (1 phone, side view)</option>'
    if _capabilities()["opensim"] or dispatch.storage_configured():
        mode_opts += ('<option value="quick">3D (OpenSim) — experimental; '
                      'single-camera depth is approximate</option>')
    return (
        '<section class="hero"><h1>Process a video</h1>'
        '<p class="lead">Pick the movement, record it from the side, and get a report. '
        'No setup needed.</p></section>'
        f'{_status_banner()}'
        '<div class="card"><form action="/process" method="post" enctype="multipart/form-data">'
        '<div class="row2"><div><label>Subject</label>'
        '<input name="subject" placeholder="e.g. J. Smith"></div>'
        '<div><label>Movement</label>'
        '<select name="task" id="task" onchange="updGuide()">'
        '<option value="gait">Walking (gait)</option>'
        '<option value="squat">Squat</option>'
        '<option value="sit_to_stand">Sit-to-stand</option>'
        '<option value="tug">Timed Up &amp; Go (TUG)</option>'
        '</select></div></div>'
        '<div id="guide" style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;'
        'padding:11px 14px;margin-top:14px;font-size:13px;color:#0f172a"></div>'
        '<div id="stsfields" style="display:none">'
        '<div class="row2"><div><label>Height (cm) &mdash; optional, for leg power</label>'
        '<input name="height_cm" type="number" step="0.1" placeholder="e.g. 170"></div>'
        '<div><label>Weight (kg) &mdash; optional, for leg power</label>'
        '<input name="weight_kg" type="number" step="0.1" placeholder="e.g. 70"></div></div></div>'
        '<label>Trial label (optional)</label>'
        '<input name="trial" placeholder="auto-named from the movement">'
        f'<label>Mode</label><select name="mode">{mode_opts}</select>'
        '<label>Video</label><input name="video" type="file" accept="video/*" required>'
        '<button class="btn" type="submit">Upload &amp; process</button></form></div>'
        '<script>'
        'var GUIDE={'
        'gait:"<b>Walking</b> &mdash; film the <b>side</b> (profile). Whole body in frame, ~3&ndash;4 m '
        'back, phone at hip height, landscape. Walk 4&ndash;6 strides across the view.",'
        'squat:"<b>Squat</b> &mdash; film the <b>side</b> (profile) for depth + trunk lean. Whole body '
        'in frame, ~3 m back. Do 3&ndash;5 squats at a steady pace.",'
        'sit_to_stand:"<b>Sit-to-stand</b> &mdash; film the <b>side</b> (profile); keep the whole body '
        '<i>and the chair</i> in frame, arms across chest (no hands). Do <b>5 stand&rarr;sit reps</b> '
        '(timed 5&times; test) <i>or</i> as many as you can in <b>30 s</b>. Enter height &amp; weight to '
        'also get a leg-power estimate.",'
        'tug:"<b>Timed Up &amp; Go</b> &mdash; film from the <b>side</b>, far enough back to keep the '
        'subject in frame the whole time. Subject sits in a chair, on \\"go\\" stands, walks ~<b>3 m</b>, '
        'turns, walks back, and sits. Total time is the result (&ge;13.5 s = elevated fall risk)."'
        '};'
        'function updGuide(){var t=document.getElementById("task").value;'
        'document.getElementById("guide").innerHTML=GUIDE[t]||"";'
        'document.getElementById("stsfields").style.display=(t=="sit_to_stand")?"block":"none";}'
        'updGuide();'
        '</script>')


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
            '(<code>gait-validate</code>; sagittal RMSE &le; ~5&deg;).</p></div>'
            '<h2 class="section">Advanced: report from an OpenSim .mot</h2>'
            '<p class="note">If you already have an OpenSim <code>.mot</code> motion file, generate a '
            'report directly (no video processing).</p>'
            + _INDEX_FORM)


def _samples_body() -> str:
    if not SAMPLE_VIDEOS:
        return ('<section class="hero"><h1>Sample videos</h1>'
                '<p class="lead">No samples configured.</p></section>')
    cards = ""
    for s in SAMPLE_VIDEOS:
        cards += (
            f'<div class="s"><span><span class="who">{s["label"]}</span>'
            f'<span class="meta"> &middot; {s["desc"]}</span></span>'
            f'<span style="display:flex;gap:8px">'
            f'<a class="meta" href="{s["url"]}" download>download</a>'
            f'<form action="/process-sample" method="post" style="margin:0">'
            f'<input type="hidden" name="sample_id" value="{s["id"]}">'
            f'<button class="btn" style="margin:0;padding:7px 14px" type="submit">Run &rarr;</button>'
            f'</form></span></div>')
    return (f'<section class="hero"><h1>Sample videos</h1>'
            f'<p class="lead">Verified full-body, side-view walking clips for testing. '
            f'Click <b>Run</b> to fetch one and generate a report, or <b>download</b> to try it via '
            f'<a href="/process">Process video</a>.</p></section>'
            f'<div class="slist">{cards}</div>')


_RECORD_TMPL = """<section class="hero"><h1>Record</h1>
<p class="lead">Pick the movement, choose orientation, frame the whole body, then Record &rarr; Stop.
The clip uploads and processes automatically. Works on a phone or a desktop webcam.</p></section>
<div class="card">
  <div class="row2"><div><label>Subject</label><input id="subject" placeholder="e.g. J. Smith"></div>
  <div><label>Movement</label><select id="task" onchange="upd()">
    <option value="gait">Walking (gait)</option><option value="squat">Squat</option>
    <option value="sit_to_stand">Sit-to-stand</option><option value="tug">Timed Up &amp; Go (TUG)</option>
  </select></div></div>
  <div id="guide" style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:11px 14px;margin-top:14px;font-size:13px;color:#0f172a"></div>
  <div id="stsfields" style="display:none"><div class="row2">
    <div><label>Height (cm) &mdash; optional, for leg power</label><input id="height_cm" type="number" step="0.1" placeholder="170"></div>
    <div><label>Weight (kg) &mdash; optional, for leg power</label><input id="weight_kg" type="number" step="0.1" placeholder="70"></div></div></div>
  <div class="row2">
    <div id="modewrap"><label>Mode</label><select id="mode">__MODE_OPTS__</select></div>
    <div><label>Orientation</label><select id="orient" onchange="initCam()">
      <option value="landscape">Landscape (wide)</option>
      <option value="portrait">Vertical (tall)</option></select></div></div>
  <label>Camera</label><select id="cam" onchange="initCam()"></select>
  <label>Preview</label>
  <video id="preview" autoplay muted playsinline style="width:100%;border-radius:10px;background:#000;max-height:60vh"></video>
  <div style="display:flex;gap:10px;margin-top:12px">
    <button class="btn" id="start" type="button" style="margin:0">&#9679; Record</button>
    <button class="btn" id="stop" type="button" disabled style="margin:0;background:#b91c1c">&#9632; Stop &amp; upload</button>
  </div>
  <p class="note" id="st" style="margin-top:12px">Requesting camera&hellip;</p>
</div>
<script>
var GUIDE={
gait:'<b>Walking</b> &mdash; film from the <b>side</b> (profile), whole body in frame, ~3&ndash;4 m back, lens at hip height. Walk 4&ndash;6 strides across the view.',
squat:'<b>Squat</b> &mdash; film from the <b>side</b>, whole body in frame, ~3 m back. Do 3&ndash;5 squats at a steady pace.',
sit_to_stand:'<b>Sit-to-stand</b> &mdash; film from the <b>side</b>; whole body + the chair in frame, arms across chest (no hands). Do 5 stand&rarr;sit reps, or as many as you can in 30 s. Enter height &amp; weight for leg power.',
tug:'<b>Timed Up &amp; Go</b> &mdash; film from the <b>side</b>, far enough to keep the subject in frame. Sit, stand on \\'go\\', walk ~3 m, turn, walk back, sit.'
};
var rec, chunks=[], stream;
function $(id){return document.getElementById(id);}
function upd(){var t=$('task').value;
  $('guide').innerHTML=GUIDE[t]||'';
  $('stsfields').style.display=(t=='sit_to_stand')?'block':'none';
  $('modewrap').style.display=(t=='gait')?'block':'none';}
function constraints(){var land=$('orient').value=='landscape';
  var v={width:{ideal:land?1920:1080},height:{ideal:land?1080:1920}};
  var dev=$('cam').value; if(dev) v.deviceId={exact:dev};
  return {video:v,audio:false};}
async function listCams(){try{
  var ds=await navigator.mediaDevices.enumerateDevices();
  var cams=ds.filter(function(d){return d.kind=='videoinput';});
  var sel=$('cam'); if(sel.options.length===cams.length && sel.options.length>0) return;
  var cur=sel.value; sel.innerHTML='';
  cams.forEach(function(c,i){var o=document.createElement('option');o.value=c.deviceId;o.text=c.label||('Camera '+(i+1));sel.add(o);});
  if(cur) sel.value=cur;}catch(e){}}
async function initCam(){try{
  if(stream){stream.getTracks().forEach(function(t){t.stop();});}
  stream=await navigator.mediaDevices.getUserMedia(constraints());
  $('preview').srcObject=stream; $('st').textContent='Ready. Press Record when framed.';
  await listCams();
  }catch(e){$('st').innerHTML='Camera unavailable ('+e.message+'). Use <a href="/process">Process video</a> to upload a clip instead.';}}
$('start').onclick=function(){if(!stream)return; chunks=[];
  var mt='video/webm';
  if(window.MediaRecorder&&MediaRecorder.isTypeSupported&&MediaRecorder.isTypeSupported('video/mp4'))mt='video/mp4';
  try{rec=new MediaRecorder(stream,{mimeType:mt});}catch(e){rec=new MediaRecorder(stream);}
  rec.ondataavailable=function(e){if(e.data&&e.data.size)chunks.push(e.data);};
  rec.start(); $('start').disabled=true; $('stop').disabled=false; $('st').textContent='Recording\\u2026';};
$('stop').onclick=function(){if(!rec)return;
  rec.onstop=async function(){
    var type=(chunks[0]&&chunks[0].type)||'video/webm';
    var ext=type.indexOf('mp4')>=0?'mp4':'webm';
    var blob=new Blob(chunks,{type:type}); var t=$('task').value;
    var fd=new FormData();
    fd.append('video',blob,'recording.'+ext);
    fd.append('subject',$('subject').value); fd.append('trial','');
    fd.append('task',t); fd.append('mode',(t=='gait')?$('mode').value:'screening');
    fd.append('height_cm',$('height_cm')?$('height_cm').value:'');
    fd.append('weight_kg',$('weight_kg')?$('weight_kg').value:'');
    $('st').textContent='Uploading \\u0026 processing\\u2026';
    try{var r=await fetch('/process',{method:'POST',body:fd}); window.location=r.url;}
    catch(e){$('st').textContent='Upload failed: '+e.message;}};
  rec.stop(); $('start').disabled=false; $('stop').disabled=true;};
upd(); initCam();
</script>"""


def _record_body() -> str:
    mode_opts = '<option value="screening">2D screening</option>'
    if _capabilities()["opensim"] or dispatch.storage_configured():
        mode_opts += '<option value="quick">3D OpenSim (walking, experimental)</option>'
    return _RECORD_TMPL.replace("__MODE_OPTS__", mode_opts)


_JOB_BODY = """<section class="hero"><h1>Processing&hellip; <span id="st" class="badge">queued</span></h1>
<p class="lead">Pose estimation can take a minute or two. This page updates live and opens the report when done.</p></section>
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


_GRAPH_BODY = """<section class="hero"><h1>Graph over time</h1>
<p class="lead">Tick the signals you want and plot them over the trial timeline. Hover for values.</p></section>
<div class="card">
  <div id="meta" class="note" style="margin-bottom:8px"></div>
  <div id="sigs" style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:14px"></div>
  <div style="position:relative;height:55vh"><canvas id="chart"></canvas></div>
  <p class="note" id="st" style="margin-top:10px">Loading&hellip;</p>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
var SID=__SID__, payload=null, chart=null;
var COLORS=['#2563eb','#16a34a','#b45309','#b91c1c','#7c3aed','#0891b2','#db2777','#65a30d','#0f766e','#9333ea'];
function $(id){return document.getElementById(id);}
function xaxis(){
  if(payload.t&&payload.t.length) return payload.t;
  var n=payload.n||0, fps=payload.fps||30, a=[]; for(var i=0;i<n;i++)a.push(+(i/fps).toFixed(3)); return a;}
function render(){
  if(!payload) return; var xs=xaxis(), ds=[], ci=0, unit='';
  Object.keys(payload.signals).forEach(function(k){
    if(!$('c_'+k)||!$('c_'+k).checked) return; var s=payload.signals[k]; unit=s.unit||unit;
    var pts=s.data.map(function(v,i){return {x:xs[i], y:v};});
    ds.push({label:s.label+(s.unit?' ('+s.unit+')':''), data:pts, borderColor:COLORS[ci%COLORS.length],
             backgroundColor:COLORS[ci%COLORS.length], borderWidth:1.6, pointRadius:0, spanGaps:false, tension:0.15}); ci++;});
  if(chart) chart.destroy();
  chart=new Chart($('chart'),{type:'line', data:{datasets:ds}, options:{
    animation:false, parsing:false, normalized:true, maintainAspectRatio:false,
    scales:{x:{type:'linear', title:{display:true,text:'time (s)'}},
            y:{title:{display:true,text:unit||'value'}}},
    plugins:{legend:{position:'top'}, tooltip:{mode:'index',intersect:false}},
    interaction:{mode:'nearest',axis:'x',intersect:false}}});
  $('st').textContent=ds.length?'':'Tick at least one signal above.';}
fetch('/session/'+SID+'/series.json').then(function(r){return r.json();}).then(function(p){
  payload=p; var names=Object.keys(p.signals);
  $('meta').textContent=names.length+' signals · '+(p.task||'')+' · '+(p.n||0)+' frames @ '+(p.fps||'?')+' fps';
  var box=$('sigs');
  names.forEach(function(k,i){var lab=document.createElement('label');
    lab.style='font-size:13px;font-weight:500;cursor:pointer';
    var on=i<3?'checked':'';
    lab.innerHTML='<input type="checkbox" id="c_'+k+'" '+on+'> '+p.signals[k].label;
    lab.querySelector('input').addEventListener('change',render); box.appendChild(lab);});
  if(!names.length){$('st').textContent='No signals available for this trial.'; return;}
  render();
}).catch(function(e){$('st').textContent='Could not load signals: '+e.message;});
</script>"""


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
    check, cross = "&#10003;", "&#10007;"

    # The DEFAULT (and the only cloud-available) mode is single-phone 2D screening, which
    # needs ONLY MediaPipe. OpenSim + a marked model are used solely by the advanced 3D
    # mode, so we present them as optional extras — not as a "needs setup" alarm.
    if c["mediapipe"]:
        head = "Ready for 2D screening &mdash; single-phone, side-view (the default mode)."
        bg = "#eafaf1"
        required = f"<div class='ok'>{check} MediaPipe installed (required for 2D screening)</div>"
    else:
        head = "MediaPipe missing &mdash; 2D screening unavailable."
        bg = "#fdf2e9"
        required = f"<div class='bad'>{cross} MediaPipe installed &mdash; pip install mediapipe</div>"

    def opt_row(ok, label):
        mark = check if ok else "&ndash;"
        return f"<div class='note' style='margin:2px 0'>{mark} {label}: {'yes' if ok else 'no'}</div>"

    optional = (
        "<div class='note' style='margin-top:8px;font-weight:600'>Optional &mdash; advanced 3D "
        "mode only (needs a 2-camera/Pose2Sim setup; not available on the cloud app):</div>"
        + opt_row(c["opensim"], "OpenSim installed")
        + opt_row(bool(c["model"]), "Marked model (GAIT_OSIM_MODEL)")
    )
    return f"<div class='banner' style='background:{bg}'><b>{head}</b>{required}{optional}</div>"


def _run_screening_subprocess(job, video_path: Path, sdir: Path, meta: dict) -> str:
    """Single-phone 2D sagittal screening in a SUBPROCESS so the heavy MediaPipe pass
    can't starve the uvicorn event loop (keeps /health + status polls responsive)."""
    import subprocess
    import sys as _sys
    job.log.append("MediaPipe pose -> 2D sagittal screening (subprocess) ...")
    proc = subprocess.run(
        [_sys.executable, "-m", "gait_analysis.web.screening_job",
         str(video_path), str(sdir), meta.get("subject") or "", meta.get("task") or "gait",
         str(meta.get("height_cm") or ""), str(meta.get("weight_kg") or "")],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError("screening failed: " + (tail[-1] if tail else "unknown error"))
    if not (sdir / "report.html").exists():
        raise RuntimeError("screening produced no report")
    (sdir / "meta.json").write_text(json.dumps({**meta, "created": _dt.datetime.now().isoformat()}))
    return sdir.name


def _sample_process(job, sample: dict, sdir: Path, meta: dict) -> str:
    """Fetch a curated sample clip then run 2D screening on it. Prefer our own private
    blob (always reachable from the server); fall back to the source URL otherwise."""
    video_path = sdir / "input.mp4"
    blob = sample.get("blob")
    if blob and os.environ.get("GAIT_STORAGE_CONNECTION"):
        job.log.append("loading sample clip ...")
        from azure.storage.blob import BlobServiceClient
        bc = (BlobServiceClient.from_connection_string(os.environ["GAIT_STORAGE_CONNECTION"])
              .get_blob_client(SAMPLE_BLOB_CONTAINER, blob))
        with open(video_path, "wb") as f:
            f.write(bc.download_blob().readall())
    else:
        import shutil as _sh
        import urllib.request
        from urllib.parse import urlparse
        url = sample.get("url", "")
        if urlparse(url).hostname not in _ALLOWED_SAMPLE_HOSTS:
            raise RuntimeError("sample source not available")
        job.log.append("downloading sample clip ...")
        req = urllib.request.Request(url, headers={"User-Agent":
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(video_path, "wb") as f:
            _sh.copyfileobj(resp, f)
    return _run_screening_subprocess(job, video_path, sdir, meta)


def _default_process(job, video_path: Path, sdir: Path, meta: dict) -> str:
    """Real quick-mode processing: video -> MediaPipe 3D -> OpenSim IK -> .mot -> report."""
    mode = meta.get("mode") or "screening"
    job.log.append(f"mode={mode}: starting")

    if mode == "screening":
        return _run_screening_subprocess(job, video_path, sdir, meta)

    if mode != "quick":
        raise RuntimeError("Only screening (2D) and quick (3D) modes are wired into the app; "
                           "use the 2-phone capture page for accurate (Pose2Sim) 3D.")

    # 3D quick mode: prefer the tier-2 OpenSim worker (the slim cloud image has no
    # OpenSim). Upload the video, enqueue a job, and poll the worker's status.json.
    if dispatch.storage_configured():
        import time as _t
        ext = video_path.suffix.lstrip(".") or "mp4"
        job.log.append("dispatching 3D job to the OpenSim worker (tier-2) ...")
        dispatch.dispatch_job(sdir.name, video_path, "quick", ext, speed=meta.get("speed"))
        deadline = _t.time() + 1800           # 30 min cap
        last = None
        while True:
            stt = dispatch.poll_status(sdir.name)
            state = stt.get("state")
            if state != last:
                job.log.append(f"worker: {state}")
                last = state
            if state == "done":
                dispatch.fetch_outputs(sdir.name, sdir)
                break
            if state == "error":
                raise RuntimeError("worker: " + stt.get("error", "3D processing failed"))
            if _t.time() > deadline:
                raise RuntimeError("3D worker timed out (30 min). Try a shorter clip.")
            _t.sleep(3)
        if not (sdir / "report.html").exists():
            raise RuntimeError("worker finished but returned no report")
        # Turn the OpenSim .mot into graphable signals (all 3D joint angles over time).
        mot = sdir / "coordinates.mot"
        if mot.exists():
            try:
                from ..analysis import series_export
                series_export.write_series(series_export.from_mot(mot), sdir / "series.json")
            except Exception as exc:
                job.log.append(f"series.json skipped: {exc}")
        (sdir / "meta.json").write_text(json.dumps({**meta, "created": _dt.datetime.now().isoformat()}))
        return sdir.name

    # Local fallback: only where OpenSim is actually installed.
    model = os.environ.get("GAIT_OSIM_MODEL")
    if not model:
        raise RuntimeError("3D needs either the cloud worker (set GAIT_STORAGE_CONNECTION) "
                           "or local OpenSim + GAIT_OSIM_MODEL.")

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

    @app.middleware("http")
    async def _require_auth(request: Request, call_next):
        # No-op when auth is disabled (no password configured).
        if _auth_enabled() and request.url.path not in _OPEN_PATHS:
            if not _valid_session(request.cookies.get(_COOKIE, "")):
                if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
                    return RedirectResponse("/login", status_code=303)
                return JSONResponse({"error": "authentication required"}, status_code=401)
        return await call_next(request)

    @app.get("/login", response_class=HTMLResponse)
    def login_form():
        return _shell("Sign in", _LOGIN_BODY.replace("<!--err-->", ""), "")

    @app.post("/login")
    def login_submit(password: str = Form("")):
        if _auth_enabled() and hmac.compare_digest(password, AUTH_PASSWORD):
            r = RedirectResponse("/", status_code=303)
            r.set_cookie(_COOKIE, _make_session_token(), max_age=_SESSION_TTL,
                         httponly=True, secure=True, samesite="lax")
            return r
        err = "<p class='err'>Incorrect password.</p>"
        return HTMLResponse(_shell("Sign in", _LOGIN_BODY.replace("<!--err-->", err), ""),
                            status_code=401)

    @app.get("/logout")
    def logout():
        r = RedirectResponse("/login", status_code=303)
        r.delete_cookie(_COOKIE)
        return r

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
        try:
            from ..analysis import series_export
            series_export.write_series(series_export.from_mot(mot_path), sdir / "series.json")
        except Exception:
            pass
        meta = {"id": sid, "subject": subject, "trial": trial, "speed": spd,
                "created": _dt.datetime.now().isoformat()}
        (sdir / "meta.json").write_text(json.dumps(meta))
        return RedirectResponse(f"/session/{sid}", status_code=303)

    @app.get("/session/{sid}", response_class=HTMLResponse)
    def session(sid: str):
        html_path = _store_dir() / sid / "report.html"
        if not html_path.exists():
            return HTMLResponse("<p>Not found. <a href='/'>Back</a></p>", status_code=404)
        report_html = html_path.read_text()
        inject = ""
        # Embed the video + 3D/OpenSim viewer INLINE (no separate tab) at the top.
        if (_store_dir() / sid / "synced" / "viewer.html").exists():
            inject += (
                f'<div style="margin:0 0 16px;font-family:system-ui,Arial,sans-serif">'
                f'<div style="font-size:13px;font-weight:600;color:#475569;margin:0 0 6px">'
                f'Video + 3D view</div>'
                f'<iframe src="/session/{sid}/synced/viewer.html" '
                f'style="width:100%;height:72vh;min-height:480px;border:1px solid #e2e8f0;'
                f'border-radius:10px" loading="lazy" title="Video and 3D skeleton"></iframe>'
                f'<div style="font-size:12px;color:#64748b;margin-top:4px">'
                f'Drag the 3D view to rotate. <a href="/session/{sid}/synced/viewer.html" '
                f'target="_blank">Open full screen</a></div></div>')
        if (_store_dir() / sid / "series.json").exists():
            inject += (
                f'<div style="margin:0 0 16px;font-family:system-ui,Arial,sans-serif;font-size:14px">'
                f'<a href="/session/{sid}/graph" target="_blank" '
                f'style="font-weight:600;color:#0369a1;text-decoration:none">'
                f'&#128202; Graph signals over time</a></div>')
        if inject:
            report_html = report_html.replace("<body>", "<body>" + inject, 1)
        return report_html

    @app.get("/session/{sid}/graph", response_class=HTMLResponse)
    def session_graph(sid: str):
        if not (_store_dir() / sid / "series.json").exists():
            return HTMLResponse("<p>No signals for this trial. <a href='/'>Back</a></p>", status_code=404)
        return _shell("Graph", _GRAPH_BODY.replace("__SID__", json.dumps(sid)), "")

    @app.get("/session/{sid}/series.json")
    def session_series(sid: str):
        fpath = _store_dir() / sid / "series.json"
        if not fpath.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(fpath, media_type="application/json")

    @app.get("/session/{sid}/synced/{filename:path}")
    def session_synced_file(sid: str, filename: str):
        fpath = _store_dir() / sid / "synced" / filename
        if not fpath.exists() or not fpath.is_file():
            return HTMLResponse("<p>Not found</p>", status_code=404)
        return FileResponse(fpath)

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

    @app.get("/record", response_class=HTMLResponse)
    def record_page():
        return _shell("Record", _record_body(), "record")

    @app.get("/samples", response_class=HTMLResponse)
    def samples_page():
        return _shell("Samples", _samples_body(), "samples")

    @app.post("/process-sample")
    def process_sample(sample_id: str = Form(...)):
        s = _sample_by_id(sample_id)
        if s is None:
            return HTMLResponse("<p>Unknown sample. <a href='/samples'>Back</a></p>", status_code=404)
        sid = uuid.uuid4().hex[:8]
        sdir = _store_dir() / sid
        sdir.mkdir(parents=True, exist_ok=True)
        meta = {"id": sid, "subject": s["label"], "trial": "sample", "speed": None, "mode": "screening"}
        (sdir / "meta.json").write_text(json.dumps({**meta, "state": "processing"}))
        jid = jm.submit(lambda job: _sample_process(job, s, sdir, meta))
        return RedirectResponse(f"/job/{jid}", status_code=303)

    # --- two-phone (accurate 3D) capture: collect 4 clips, then process ---

    @app.get("/capture", response_class=HTMLResponse)
    def capture_page():
        return _shell("Capture (2 phones)", CAPTURE_BODY, "capture")

    @app.post("/capture-upload")
    async def capture_upload(video: UploadFile, code: str = Form(...),
                             role: str = Form(...), kind: str = Form(...)):
        ext = (Path(video.filename or "clip.webm").suffix or ".webm").lstrip(".")
        try:
            twophone.save_clip(code, role, kind, await video.read(), ext)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        s = twophone.session_status(code)
        return JSONResponse({"received": s["present"], "ready": s["ready"], "missing": s["missing"]})

    @app.get("/capture-status")
    def capture_status(code: str):
        try:
            s = twophone.session_status(code)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"received": s["present"], "ready": s["ready"], "missing": s["missing"]})

    @app.post("/capture-run")
    def capture_run(code: str = Form(...)):
        s = twophone.session_status(code)
        if not s.get("ready"):
            body = (f"<section class='hero'><h1>Not ready</h1><p class='lead'>Session "
                    f"<b>{code}</b> still needs: {', '.join(s.get('missing', [])) or 'unknown'}. "
                    f"<a href='/capture'>Back to capture</a></p></section>")
            return HTMLResponse(_shell("Capture", body, "capture"), status_code=400)
        # The 4 clips are saved. Pose2Sim+OpenSim 3D runs where OpenSim exists (the
        # tier-2 worker / local dev). On the slim cloud app we stage + explain.
        if _capabilities()["opensim"]:
            sid = uuid.uuid4().hex[:8]
            sdir = _store_dir() / sid
            sdir.mkdir(parents=True, exist_ok=True)

            def _run(job):
                job.log.append(f"two-phone {code}: Pose2Sim triangulation + OpenSim IK ...")
                res = twophone.run_session(code, trial=f"two-phone {code}")
                rep = res.get("report")
                if rep and Path(rep).exists():
                    (sdir / "report.html").write_bytes(Path(rep).read_bytes())
                (sdir / "meta.json").write_text(json.dumps(
                    {"id": sid, "subject": "two-phone", "trial": code,
                     "created": _dt.datetime.now().isoformat()}))
                return sid
            jid = jm.submit(_run)
            return RedirectResponse(f"/job/{jid}", status_code=303)
        body = (f"<section class='hero'><h1>Trial captured &#10003;</h1>"
                f"<p class='lead'>All 4 clips for session <b>{code}</b> are saved.</p></section>"
                f"<div class='banner' style='background:#eff6ff;border-color:#bae6fd'>"
                f"<b>Next: 3D processing</b>Two-phone 3D (Pose2Sim triangulation &rarr; OpenSim) "
                f"runs on the OpenSim worker — that cloud step is the next piece being wired. Your "
                f"capture is saved. Single-phone 2D screening works now via "
                f"<a href='/process'>Process video</a>.</div>")
        return HTMLResponse(_shell("Capture", body, "capture"))

    @app.post("/process")
    async def process(video: UploadFile, subject: str = Form(""), trial: str = Form(""),
                      task: str = Form("gait"), speed: str = Form(""), mode: str = Form("screening"),
                      height_cm: str = Form(""), weight_kg: str = Form("")):
        sid = uuid.uuid4().hex[:8]
        sdir = _store_dir() / sid
        sdir.mkdir(parents=True, exist_ok=True)
        suffix = Path(video.filename or "input.mov").suffix or ".mov"
        video_path = sdir / f"input{suffix}"
        video_path.write_bytes(await video.read())
        task = task if task in ("gait", "squat", "sit_to_stand", "tug") else "gait"
        trial_full = trial or {"gait": "walk", "squat": "squat",
                               "sit_to_stand": "sit-to-stand", "tug": "TUG"}.get(task, task)

        def _num(s):
            try:
                return float(s) if s.strip() else None
            except ValueError:
                return None
        meta = {"id": sid, "subject": subject, "trial": trial_full, "task": task,
                "speed": _num(speed), "mode": mode,
                "height_cm": _num(height_cm), "weight_kg": _num(weight_kg)}
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
