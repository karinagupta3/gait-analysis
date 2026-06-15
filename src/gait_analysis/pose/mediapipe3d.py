"""Single-camera 3D landmarks via MediaPipe BlazePose GHUM (quick mode).

Apache-2.0, commercial-use-OK, trained on Google's own data (no Human3.6M / SMPL
license contamination -- see docs/01 s2.5 and docs/03). Outputs metric 3D
"world landmarks" (33 points, in METRES, hip-centred), which we then convert to an
OpenSim marker file for inverse kinematics.

HONESTY: a single camera's DEPTH (z) is the weakest axis, so the out-of-sagittal
angles (adduction, rotation, subtalar) that depend on z are low-confidence. Quick
mode is for the "record one phone -> get a 3D model + broad data" experience;
trust sagittal angles, flag the rest. Accurate frontal/transverse needs 2 phones.

Output .npz:
    world_landmarks : (T, 33, 3) float32   metres, hip-centred  (NaN if no person)
    visibility      : (T, 33)    float32   0..1 confidence
    image_landmarks : (T, 33, 2) float32   normalised 0..1 image coords (for overlay)
    fps, width, height
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# BlazePose 33-landmark names (index order), per the MediaPipe Pose spec.
BLAZEPOSE_33 = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
N_LM = len(BLAZEPOSE_33)


def _load_pose(model_complexity: int = 2, min_conf: float = 0.5):
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "mediapipe is not installed. Run:  pip install mediapipe\n"
            "(quick-mode 3D; Apache-2.0, commercial-use-OK)"
        ) from exc
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,  # 2 = 'heavy', most accurate
        enable_segmentation=False,
        min_detection_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )


def extract_world_landmarks(
    video_path: str | Path,
    model_complexity: int = 2,
    overlay_path: str | Path | None = None,
    max_frames: int | None = None,
):
    """Run MediaPipe Pose over a video; return per-frame 3D world landmarks."""
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    pose = _load_pose(model_complexity=model_complexity)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if overlay_path is not None:
        import mediapipe as mp
        mp_draw = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, fps or 30.0, (width, height))

    world_seq, vis_seq, img_seq = [], [], []
    n_detected = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_world_landmarks is None:
            world_seq.append(np.full((N_LM, 3), np.nan, dtype=np.float32))
            vis_seq.append(np.zeros((N_LM,), dtype=np.float32))
            img_seq.append(np.full((N_LM, 2), np.nan, dtype=np.float32))
        else:
            wl = result.pose_world_landmarks.landmark
            world_seq.append(np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32))
            vis_seq.append(np.array([p.visibility for p in wl], dtype=np.float32))
            il = result.pose_landmarks.landmark
            img_seq.append(np.array([[p.x, p.y] for p in il], dtype=np.float32))
            n_detected += 1
            if writer is not None:
                mp_draw.draw_landmarks(frame, result.pose_landmarks,
                                       mp_pose.POSE_CONNECTIONS)

        if writer is not None:
            writer.write(frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  ...{frame_idx} frames ({n_detected} with a subject)", file=sys.stderr)

    cap.release()
    pose.close()
    if writer is not None:
        writer.release()
    if frame_idx == 0:
        raise RuntimeError("No frames read from video.")

    print(
        f"Extracted {frame_idx} frames @ {fps:.1f} fps; subject detected in "
        f"{n_detected} ({100 * n_detected / frame_idx:.0f}%).",
        file=sys.stderr,
    )
    return {
        "world_landmarks": np.stack(world_seq),   # (T,33,3) metres
        "visibility": np.stack(vis_seq),           # (T,33)
        "image_landmarks": np.stack(img_seq),      # (T,33,2) normalised
        "fps": np.float32(fps),
        "width": np.int32(width),
        "height": np.int32(height),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MediaPipe 3D world landmarks from a single video.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="Output .npz")
    ap.add_argument("--overlay", default=None, help="Optional overlay .mp4")
    ap.add_argument("--complexity", type=int, default=2, choices=[0, 1, 2],
                    help="0 lite, 1 full, 2 heavy (default, most accurate)")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    result = extract_world_landmarks(
        args.video, model_complexity=args.complexity,
        overlay_path=args.overlay, max_frames=args.max_frames,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **result)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
