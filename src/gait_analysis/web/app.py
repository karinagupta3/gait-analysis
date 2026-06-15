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
import tempfile
import uuid
from pathlib import Path

from ..analysis import report

# Lazy/guarded FastAPI import so the package imports without the web extra.
try:
    from fastapi import FastAPI, Form, Request, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
except ImportError:  # pragma: no cover - exercised only without the extra
    FastAPI = None  # type: ignore


# On-disk store: DATA_DIR/sessions/<id>/{meta.json, trial.mot, report.html}
DATA_DIR = Path(__file__).resolve().parent / "_data"


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
<h1>Gait Analysis</h1><div class="sub">Upload an OpenSim <code>.mot</code> to generate a clinical report.</div>
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


def create_app():
    if FastAPI is None:
        raise SystemExit("FastAPI not installed. Run:  pip install -e \".[web]\"")
    app = FastAPI(title="Gait Analysis")

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

    return app


def run():  # console entry point: gait-web
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
