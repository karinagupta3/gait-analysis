# UI + 3D skeletal rendering plan

Goal: turn the working pipeline into a thorough, accurate, *trustworthy* product —
a report a clinician/researcher reads in 30 seconds and a 3D playback like OpenCap's.
Built in layers so each ships value on its own.

## How the 3D skeletal rendering works (the OpenCap approach, and ours)
The skeleton is the **OpenSim model animated by the IK results** — no separate skeleton:
1. Inputs we already produce: the **scaled `.osim`** (carries bone meshes — `.vtp`/`.obj`
   geometry per body) and the **`.mot`** (every coordinate, every frame).
2. For each frame, set the model coordinates to that frame's `.mot` values and run
   **forward kinematics** → OpenSim's API gives each body's position + orientation
   (`body.getTransformInGround(state)`).
3. A **WebGL viewer (three.js)** loads the bone meshes once and re-poses them each frame
   from those transforms. OpenCap does exactly this in its web app.

**Our build:** a small exporter `biomech/export_scene.py` that, given `.osim` + `.mot`,
walks frames via the OpenSim API, collects per-frame body transforms + mesh references,
and writes a compact **glTF/JSON animation** (or per-frame transforms + a one-time mesh
bundle). The web viewer renders that. Runs wherever OpenSim runs (the conda env). No SMPL,
all Apache-licensed assets — commercial-safe.

## Layered roadmap

### Layer 0 — DONE: self-contained HTML report (`analysis/report.py`)
One `.html` with: confidence banner (cycles/duration/speed), phase-gated signature flags
(interpretations + confirming test), embedded R-vs-L joint-angle curves, ROM vs normative
table, limitations footer. Ship now: `gait-pipeline --from-mot <mot> --html report.html`.

### Layer 1 — Richer report content (next)
- **Normative band overlays** on the curves (shade Fukuchi 2018 mean±SD per coordinate vs
  % gait cycle), so deviations are visual, not just numeric.
- **Spatiotemporal panel** from gait events: cadence, step/stride time, stance/swing %,
  double-support, step width (derive from the augmented `.trc` foot markers).
- **Asymmetry indices**: Symmetry Index, GDI, GPS/MAP (docs/04 §2.4) computed and plotted.
- **Per-flag "what to do next"**: the confirming test + a plain-language explanation.

### Layer 2 — 3D viewer (`web/`)
- `biomech/export_scene.py`: `.osim` + `.mot` → glTF/JSON (meshes + per-frame transforms).
- A static three.js page: load scene, play/scrub timeline, rotate, toggle planes, and
  **scrub synced to the joint-angle plots** (hover a frame → pose + curve cursor move).
- Overlay the original video next to the 3D for visual validation.

### Layer 3 — App shell (`web/` + small API)
- **Subject → Session → Trial** data model (deferred earlier; add here) so trials are
  organized and **comparable across visits** (overlay today's curves on the baseline).
- Upload/capture → process → report, with quick-mode (1 phone) and accurate-mode (2 phone).
- Export PDF; share link.

## Accuracy & trust (non-negotiable, threads through every layer)
- Every report keeps the **confidence banner** (cycles, mode) and the **limitations footer**.
- Flags stay **decision-support, not diagnosis**, with multiple interpretations + a confirming
  test (docs/04). No muscle strength/structure claims from kinematics.
- Normative comparisons are **age/sex/speed-matched** where the dataset allows (Fukuchi).
- Markerless **transverse/frontal** coordinates are shown but labelled low-confidence.

## Tech choices (commercial-safe)
- Report: server-free **HTML + embedded PNG** (done) → later a small **FastAPI** (MIT) backend
  + **React/three.js** (MIT/Apache) frontend for the app shell.
- 3D: **three.js** (MIT). Meshes from the OpenSim model (Apache). No SMPL anywhere in the
  shipping product.
