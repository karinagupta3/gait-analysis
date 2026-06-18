"""Export per-frame signals to a uniform JSON the in-browser grapher consumes.

One shape for everything the user might want to plot over time:
  {task, fps, n, t?[...], signals: { key: {label, unit, data:[...]} }}
data may contain nulls (NaN gaps). Sources:
  * 2D screening metrics (gait both legs; squat/STS/TUG)  -> from_screening_metrics
  * an OpenSim .mot (every joint coordinate over time)    -> from_mot  (this is how
    "hooking up OpenSim" feeds the grapher — all 3D joint angles become signals)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _clean(arr, nd=2):
    """List with NaN/inf -> None, rounded; accepts numpy or list."""
    out = []
    for v in (arr.tolist() if hasattr(arr, "tolist") else list(arr)):
        out.append(None if v is None or (isinstance(v, float) and not math.isfinite(v))
                   else round(float(v), nd))
    return out


def from_screening_metrics(task: str, metrics: dict) -> dict:
    fps = float(metrics.get("fps") or 30.0)
    signals: dict[str, dict] = {}

    def add(key, label, unit, arr):
        if arr is None:
            return
        data = _clean(arr)
        if any(v is not None for v in data):
            signals[key] = {"label": label, "unit": unit, "data": data}

    if task == "gait":
        for side in ("right", "left"):
            s = (metrics.get("sides", {}).get(side, {}) or {}).get("_series", {}) or {}
            tag = side[0].upper()
            add(f"knee_{side}", f"Knee flexion ({tag})", "deg", s.get("knee"))
            add(f"hip_{side}", f"Hip flexion ({tag})", "deg", s.get("hip"))
    elif task in ("squat", "sit_to_stand"):
        s = metrics.get("_series", {}) or {}
        add("knee", "Knee flexion", "deg", s.get("knee"))
        add("hip", "Hip flexion", "deg", s.get("hip"))
        add("trunk", "Trunk lean", "deg", s.get("trunk"))
    elif task == "tug":
        s = metrics.get("_series", {}) or {}
        hy = s.get("hip_y")
        if hy is not None:
            add("hip_height", "Hip height (up=standing)", "px", [-float(v) for v in hy])
        add("hip_x", "Hip horizontal", "px", s.get("hip_x"))

    n = max((len(v["data"]) for v in signals.values()), default=0)
    return {"task": task, "fps": round(fps, 3), "n": n, "signals": signals}


def from_mot(mot_path) -> dict:
    """Every joint coordinate in an OpenSim .mot as a graphable signal."""
    from .kinematics import read_storage
    t, cols, meta = read_storage(mot_path)
    fps = (round(1.0 / float(np.median(np.diff(t))), 3) if len(t) > 1 else 30.0)
    unit = "deg" if str(meta.get("inDegrees", "yes")).lower() in ("yes", "true") else "rad"
    signals = {name: {"label": name.replace("_", " "), "unit": unit, "data": _clean(vals)}
               for name, vals in cols.items()}
    return {"task": "3d", "fps": fps, "n": int(len(t)), "t": _clean(t, 3), "signals": signals}


def write_series(payload: dict, out_path) -> Path:
    out_path = Path(out_path)
    out_path.write_text(json.dumps(payload))
    return out_path
