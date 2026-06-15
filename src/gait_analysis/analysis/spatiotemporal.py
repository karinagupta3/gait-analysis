"""Spatiotemporal gait parameters from 2D keypoints.

Gait-event detection uses the coordinate-based algorithm of Zeni et al. (2008),
*Gait & Posture* 27(4):710-714: heel strike occurs at the maximum anterior
position of the foot relative to the pelvis; toe-off at the maximum posterior
position. We use the ankle keypoint relative to the pelvis (hip midpoint) so
only COCO-17 keypoints are required.

IMPORTANT (honesty, see docs/01 s2.5):
  * Temporal parameters (cadence, step/stride time, temporal symmetry) are
    valid from a single 2D sagittal view and are scale-free.
  * Spatial parameters (step LENGTH in metres) require camera calibration or
    multi-view triangulation and are NOT computed here -- only pixel-space
    placeholders are reported, clearly labelled.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from ..config import COCO17, GaitConfig


def _interp_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate over NaN gaps (missing detections) in a 1D signal."""
    x = x.astype(float).copy()
    n = len(x)
    idx = np.arange(n)
    good = ~np.isnan(x)
    if good.sum() < 2:
        return x
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def _mask_low_conf(keypoints: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """Return keypoints with low-confidence points set to NaN. (T,17,2)."""
    kpts = keypoints.copy()
    low = scores < thr
    kpts[low] = np.nan
    return kpts


def detect_events(keypoints: np.ndarray, scores: np.ndarray, cfg: GaitConfig) -> dict:
    """Detect heel strikes / toe-offs per side using Zeni's coordinate method.

    Returns a dict with per-side event frame indices and the walking-direction
    sign that was inferred.
    """
    kpts = _mask_low_conf(keypoints, scores, cfg.min_keypoint_score)

    hipL = kpts[:, COCO17["left_hip"], 0]
    hipR = kpts[:, COCO17["right_hip"], 0]
    with warnings.catch_warnings():  # all-NaN frames are expected; filled below
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pelvis_x = np.nanmean(np.stack([hipL, hipR]), axis=0)
    pelvis_x = _interp_nans(pelvis_x)

    # Infer walking direction from net pelvis displacement (sign of x travel).
    direction = np.sign(pelvis_x[-1] - pelvis_x[0]) or 1.0
    min_sep = max(1, int(cfg.min_event_sep_s * cfg.fps))

    events: dict[str, dict] = {}
    for side, ankle_key in (("left", "left_ankle"), ("right", "right_ankle")):
        ankle_x = _interp_nans(kpts[:, COCO17[ankle_key], 0])
        # Foot position projected onto walking direction, relative to pelvis.
        rel = (ankle_x - pelvis_x) * direction
        # Heel strike: foot maximally forward (peak of rel).
        hs, _ = find_peaks(rel, distance=min_sep)
        # Toe-off: foot maximally backward (peak of -rel).
        to, _ = find_peaks(-rel, distance=min_sep)
        events[side] = {"heel_strike": hs, "toe_off": to}

    return {"direction": float(direction), "events": events}


def compute_parameters(keypoints: np.ndarray, scores: np.ndarray, cfg: GaitConfig) -> dict:
    """Compute spatiotemporal parameters and a temporal symmetry ratio."""
    det = detect_events(keypoints, scores, cfg)
    ev = det["events"]
    fps = cfg.fps

    def _intervals_s(frames: np.ndarray) -> np.ndarray:
        return np.diff(np.asarray(frames)) / fps if len(frames) >= 2 else np.array([])

    strideL = _intervals_s(ev["left"]["heel_strike"])   # ipsilateral HS-HS
    strideR = _intervals_s(ev["right"]["heel_strike"])

    # Cadence: total heel strikes (both feet) over the analysed duration.
    all_hs = np.sort(np.concatenate([ev["left"]["heel_strike"], ev["right"]["heel_strike"]]))
    cadence = np.nan
    if len(all_hs) >= 2:
        duration_s = (all_hs[-1] - all_hs[0]) / fps
        if duration_s > 0:
            cadence = (len(all_hs) - 1) / duration_s * 60.0  # steps per minute

    # Step time = interval between successive contralateral heel strikes.
    step_times = _intervals_s(all_hs)

    mean_strideL = float(np.mean(strideL)) if strideL.size else np.nan
    mean_strideR = float(np.mean(strideR)) if strideR.size else np.nan

    # Temporal symmetry ratio of stride times (1.0 = symmetric). See docs/01 s2.4.
    symmetry_ratio = np.nan
    if np.isfinite(mean_strideL) and np.isfinite(mean_strideR) and mean_strideR > 0:
        symmetry_ratio = mean_strideL / mean_strideR

    return {
        "n_frames": int(keypoints.shape[0]),
        "fps": float(fps),
        "direction": det["direction"],
        "n_heel_strikes": {"left": int(len(ev["left"]["heel_strike"])),
                           "right": int(len(ev["right"]["heel_strike"]))},
        "cadence_steps_per_min": float(cadence),
        "mean_step_time_s": float(np.mean(step_times)) if step_times.size else np.nan,
        "mean_stride_time_s": {"left": mean_strideL, "right": mean_strideR},
        "stride_time_symmetry_ratio_LR": float(symmetry_ratio),
        "_note": "Temporal params only. Step LENGTH (metric) needs calibration/triangulation.",
    }


def format_report(params: dict) -> str:
    lines = ["=== Spatiotemporal gait report (2D, temporal only) ==="]
    lines.append(f"Frames analysed : {params['n_frames']} @ {params['fps']:.1f} fps")
    lines.append(f"Walking dir sign: {params['direction']:+.0f} (x-axis)")
    lines.append(f"Heel strikes    : L={params['n_heel_strikes']['left']}  "
                 f"R={params['n_heel_strikes']['right']}")
    cad = params["cadence_steps_per_min"]
    lines.append(f"Cadence         : {cad:.1f} steps/min" if np.isfinite(cad) else
                 "Cadence         : n/a (need >=2 heel strikes)")
    st = params["mean_step_time_s"]
    lines.append(f"Mean step time  : {st:.3f} s" if np.isfinite(st) else "Mean step time  : n/a")
    sl, sr = params["mean_stride_time_s"]["left"], params["mean_stride_time_s"]["right"]
    lines.append(f"Mean stride time: L={sl:.3f}s  R={sr:.3f}s"
                 if np.isfinite(sl) and np.isfinite(sr) else "Mean stride time: n/a")
    sym = params["stride_time_symmetry_ratio_LR"]
    if np.isfinite(sym):
        flag = "OK" if 0.95 <= sym <= 1.05 else "ASYMMETRIC (>5%)"
        lines.append(f"Stride symmetry : L/R = {sym:.3f}  [{flag}]")
    else:
        lines.append("Stride symmetry : n/a")
    lines.append(f"Note            : {params['_note']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Spatiotemporal gait parameters from an RTMPose .npz")
    ap.add_argument("--keypoints", required=True, help="Input .npz from rtmpose_runner")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override fps (defaults to the value stored in the .npz)")
    args = ap.parse_args(argv)

    data = np.load(args.keypoints)
    fps = args.fps or float(data["fps"]) or GaitConfig.fps
    cfg = GaitConfig(fps=fps)
    params = compute_parameters(data["keypoints"], data["scores"], cfg)
    print(format_report(params))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
