"""RTMPose 2D keypoint extraction (Halpe-26, incl. feet) via rtmlib (CPU/ONNX).

RTMPose localizes joints more accurately than MediaPipe BlazePose (OpenCapBench),
and the Halpe-26 set includes real heel/big-toe/small-toe points. We use it only
for the in-plane (x, y) signal; depth comes from MediaPipe (see biomech/hybrid3d).
Pure-ONNX (rtmlib) so it runs on the CPU worker with no GPU/mmpose dependency, and
the model weights are permissively licensed (no SMPL).

Halpe-26 index order (returned as `keypoints`):
 0 nose 1 Leye 2 Reye 3 Lear 4 Rear 5 Lsho 6 Rsho 7 Lelb 8 Relb 9 Lwri 10 Rwri
 11 Lhip 12 Rhip 13 Lkne 14 Rkne 15 Lank 16 Rank 17 head 18 neck 19 midhip
 20 LbigToe 21 RbigToe 22 LsmallToe 23 RsmallToe 24 Lheel 25 Rheel
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# body keypoints used to pick/track the main subject + estimate pixel stature
_BODY = [0, 5, 6, 11, 12, 13, 14, 15, 16, 24, 25]


def extract_rtmpose(video_path: str | Path, mode: str = "lightweight") -> dict:
    """Run RTMPose BodyWithFeet over a video; track the main subject across frames.

    Returns {keypoints (T,26,2) pixels, scores (T,26), fps, width, height}. Frames
    with no detection get NaN keypoints / zero scores.
    """
    import cv2
    from rtmlib import BodyWithFeet

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)   # match mediapipe extraction
    except Exception:
        pass
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    model = BodyWithFeet(mode=mode, backend="onnxruntime", device="cpu")

    kps, scs = [], []
    prev_c = None
    W = H = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        kp, sc = model(frame)
        kp = np.asarray(kp, dtype=float)
        sc = np.asarray(sc, dtype=float)
        if kp.ndim != 3 or len(kp) == 0:
            kps.append(np.full((26, 2), np.nan)); scs.append(np.zeros(26)); continue
        # main subject = largest body, biased toward continuity with the last frame
        cents = kp[:, _BODY, :].mean(axis=1)                       # (N,2)
        areas = np.array([np.ptp(p[_BODY, 0]) * np.ptp(p[_BODY, 1]) for p in kp])
        if prev_c is None:
            m = int(np.argmax(areas))
        else:
            d = np.linalg.norm(cents - prev_c, axis=1)
            m = int(np.argmax(areas / (1.0 + d)))
        prev_c = cents[m]
        kps.append(kp[m]); scs.append(sc[m])
    cap.release()
    return {"keypoints": np.asarray(kps), "scores": np.asarray(scs),
            "fps": fps, "width": int(W), "height": int(H)}
