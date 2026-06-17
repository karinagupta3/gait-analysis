"""Render pose-overlay frames the SAME way the synced viewer draws them (after the
frame-validity gate + smoothing), so we can visually inspect marker quality.

Usage: python scripts/debug_overlay.py <video> <out_prefix>
Writes <out_prefix>_*.png and prints how many frames survive the quality gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gait_analysis.pose import mediapipe3d  # noqa: E402
from gait_analysis.analysis.sagittal2d import smooth_along_time, valid_frame_mask  # noqa: E402

_BLAZE33 = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26),
            (26, 28), (28, 30), (30, 32), (28, 32), (0, 11), (0, 12)]
MIN_SCORE = 0.5


def draw_overlay(frame, img_lm, vis, valid):
    h, w = frame.shape[:2]
    if not valid:
        cv2.putText(frame, "no reliable pose (subject not fully in frame)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 220), 2)
        return frame
    pts = []
    for j in range(img_lm.shape[0]):
        x, y = img_lm[j]
        pts.append(None if (not np.isfinite([x, y]).all() or vis[j] < MIN_SCORE)
                   else (int(x * w), int(y * h)))
    for a, b in _BLAZE33:
        if pts[a] and pts[b]:
            cv2.line(frame, pts[a], pts[b], (255, 203, 158), 2)
    for p in pts:
        if p:
            cv2.circle(frame, p, 5, (191, 212, 45), -1)
    return frame


def main():
    video, prefix = sys.argv[1], sys.argv[2]
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
    d = mediapipe3d.extract_world_landmarks(video, max_frames=max_frames)
    img_raw = d["image_landmarks"]
    vis = d["visibility"]
    valid = valid_frame_mask(img_raw, vis)
    img = smooth_along_time(img_raw)
    T = img.shape[0]
    nvalid = int(valid.sum())
    print(f"frames_total={T}  frames_valid={nvalid} ({100*nvalid/T:.0f}%)")

    valid_idx = np.where(valid)[0]
    picks = {}
    if valid_idx.size:
        picks["valid_a"] = int(valid_idx[len(valid_idx) // 4])
        picks["valid_b"] = int(valid_idx[len(valid_idx) // 2])
        picks["valid_c"] = int(valid_idx[3 * len(valid_idx) // 4])
    # An invalid frame (entry/exit) to confirm it is now blanked.
    inv_idx = np.where(~valid)[0]
    if inv_idx.size:
        picks["invalid"] = int(inv_idx[len(inv_idx) // 2])

    cap = cv2.VideoCapture(video)
    for tag, fidx in picks.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = draw_overlay(frame, img[fidx], vis[fidx], bool(valid[fidx]))
        out = f"{prefix}_{tag}.png"
        cv2.imwrite(out, frame)
        print(f"{out}  frame={fidx}  valid={bool(valid[fidx])}")
    cap.release()


if __name__ == "__main__":
    main()
