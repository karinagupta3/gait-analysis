"""Run RTMPose on a single video and export 2D keypoints.

Quick-mode (1 phone) Phase-1 entry point. Outputs an .npz with:
    keypoints : float32 (T, 17, 2)   pixel x,y; NaN where the person is missing
    scores    : float32 (T, 17)      per-keypoint confidence
    fps       : float
    width,height : int

Usage:
    python -m gait_analysis.pose.rtmpose_runner --video walk.mov --out walk.npz
    python -m gait_analysis.pose.rtmpose_runner --video walk.mov --out walk.npz --overlay walk_overlay.mp4

RTMPose downloads its ONNX weights on first run (needs network once); they are
then cached locally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..config import PoseConfig


def _load_body(cfg: PoseConfig):
    """Import rtmlib lazily so the rest of the package imports without it."""
    try:
        from rtmlib import Body
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "rtmlib is not installed. Run:  pip install rtmlib onnxruntime\n"
            "(see setup/setup_macos.sh)"
        ) from exc
    return Body(mode=cfg.mode, backend=cfg.backend, device=cfg.device)


def _pick_person(keypoints: np.ndarray, scores: np.ndarray, min_score: float):
    """Select the single most confident person in a frame.

    keypoints: (N,17,2), scores: (N,17). Returns (17,2),(17,) or None.
    For Phase 1 we assume a single subject in view; if several are detected we
    keep the highest mean-confidence one.
    """
    if keypoints is None or len(keypoints) == 0:
        return None
    mean_scores = scores.mean(axis=1)
    best = int(np.argmax(mean_scores))
    if mean_scores[best] < min_score:
        return None
    return keypoints[best].astype(np.float32), scores[best].astype(np.float32)


def extract_keypoints(
    video_path: str | Path,
    cfg: PoseConfig | None = None,
    overlay_path: str | Path | None = None,
    max_frames: int | None = None,
):
    """Extract per-frame 2D keypoints from a video. Returns a dict of arrays."""
    import cv2  # local import: heavy, and keeps package import cheap

    cfg = cfg or PoseConfig()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    body = _load_body(cfg)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if overlay_path is not None:
        from rtmlib import draw_skeleton  # noqa: F401  (used below)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, fps or 30.0, (width, height))

    kpts_seq: list[np.ndarray] = []
    score_seq: list[np.ndarray] = []
    n_detected = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        keypoints, scores = body(frame)
        picked = _pick_person(keypoints, scores, cfg.min_person_score)
        if picked is None:
            kpts_seq.append(np.full((17, 2), np.nan, dtype=np.float32))
            score_seq.append(np.zeros((17,), dtype=np.float32))
        else:
            k, s = picked
            kpts_seq.append(k)
            score_seq.append(s)
            n_detected += 1

        if writer is not None:
            from rtmlib import draw_skeleton
            vis = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.3)
            writer.write(vis)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  ...{frame_idx} frames ({n_detected} with a subject)", file=sys.stderr)

    cap.release()
    if writer is not None:
        writer.release()

    if frame_idx == 0:
        raise RuntimeError("No frames read from video.")

    result = {
        "keypoints": np.stack(kpts_seq),       # (T,17,2)
        "scores": np.stack(score_seq),         # (T,17)
        "fps": np.float32(fps),
        "width": np.int32(width),
        "height": np.int32(height),
    }
    print(
        f"Extracted {frame_idx} frames @ {fps:.1f} fps; "
        f"subject detected in {n_detected} ({100 * n_detected / frame_idx:.0f}%).",
        file=sys.stderr,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RTMPose 2D keypoint extraction from a video.")
    ap.add_argument("--video", required=True, help="Input video (e.g. iPhone .mov)")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--overlay", default=None, help="Optional skeleton-overlay .mp4 path")
    ap.add_argument("--device", default=PoseConfig.device, help="cpu | cuda | mps")
    ap.add_argument("--mode", default=PoseConfig.mode, help="lightweight | balanced | performance")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = PoseConfig(mode=args.mode, device=args.device)
    result = extract_keypoints(args.video, cfg, overlay_path=args.overlay, max_frames=args.max_frames)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **result)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
