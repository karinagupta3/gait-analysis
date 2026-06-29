"""Hybrid monocular 3D: RTMPose's sharp 2D (x, y) + MediaPipe's depth (z).

A single side-view camera measures the SAGITTAL plane (image x, y) well but depth
poorly. RTMPose localizes the in-plane joints more accurately than MediaPipe, but is
2D-only; MediaPipe gives a (rough) metric depth. So we keep MediaPipe's z and replace
the in-plane x, y with RTMPose -- improving the reliable plane without needing a GPU,
a 2D->3D lifter, or the (non-commercial) SMPL body models. The output is a drop-in
replacement for the mediapipe3d extraction dict, so the rest of the pipeline
(remap -> facing -> marker augmentation -> OpenSim) is unchanged.

Depth (z) and the occluded far leg stay limited -- that needs a second camera.
"""

from __future__ import annotations

import numpy as np

# Halpe-26 (RTMPose) index -> BlazePose-33 slot (only the joints we drive from RTMPose).
_H2BP = {
    0: 0,                       # nose
    1: 2, 2: 5, 3: 7, 4: 8,     # L/R eye, L/R ear
    5: 11, 6: 12,               # shoulders
    7: 13, 8: 14,               # elbows
    9: 15, 10: 16,              # wrists
    11: 23, 12: 24,             # hips
    13: 25, 14: 26,             # knees
    15: 27, 16: 28,             # ankles
    24: 29, 25: 30,             # heels
    20: 31, 21: 32,             # big toes -> foot_index slots
}
_HALPE_BODY = [0, 5, 6, 11, 12, 13, 14, 15, 16, 24, 25]


def build_hybrid(mp: dict, rtm: dict, height_m: float, min_score: float = 0.3) -> dict:
    """Combine a mediapipe3d dict (`mp`) with an rtmpose2d dict (`rtm`).

    Returns a dict in the SAME shape as mediapipe3d.extract_world_landmarks
    (world_landmarks (T,33,3) in MediaPipe convention, visibility (T,33),
    image_landmarks, fps, width, height), with the in-plane x/y of the driven joints
    taken from RTMPose (metre-scaled via subject height) and z from MediaPipe.
    """
    mw = np.asarray(mp["world_landmarks"], dtype=float).copy()   # (T,33,3) mp frame
    mvis = np.asarray(mp["visibility"], dtype=float).copy()
    rk = np.asarray(rtm["keypoints"], dtype=float)               # (Tr,26,2) px
    rs = np.asarray(rtm["scores"], dtype=float)
    T = min(mw.shape[0], rk.shape[0])
    if T == 0:
        return mp

    hip_px = 0.5 * (rk[:, 11, :] + rk[:, 12, :])                 # (Tr,2) pixel midhip

    # metres-per-pixel from subject height: pixel stature ~ 0.85 * height (nose->heel).
    spans = []
    for f in range(rk.shape[0]):
        ok = rs[f, _HALPE_BODY] > min_score
        if ok.sum() >= 4:
            yy = rk[f, _HALPE_BODY, 1][ok]
            spans.append(float(yy.max() - yy.min()))
    pix_span = np.median(spans) if spans else (rtm.get("height", 1) * 0.6)
    mpp = float(height_m) / (pix_span / 0.85 + 1e-6)

    # rescale MediaPipe metric depth to the subject's real scale (MediaPipe under-
    # estimates stature), so z is consistent with the RTMPose-derived x/y.
    mp_y = mw[..., 1]
    mp_span = float(np.nanpercentile(mp_y, 97.5) - np.nanpercentile(mp_y, 2.5))
    z_scale = float(height_m) / (mp_span + 1e-6)

    out = mw.copy()
    ovis = mvis.copy()
    for hidx, bidx in _H2BP.items():
        for f in range(T):
            if rs[f, hidx] < min_score or not np.isfinite(rk[f, hidx]).all():
                continue                                          # keep MediaPipe slot
            dx = (rk[f, hidx, 0] - hip_px[f, 0]) * mpp            # right(+), mp convention
            dy = (rk[f, hidx, 1] - hip_px[f, 1]) * mpp            # down(+), mp convention
            z = mw[f, bidx, 2] * z_scale                          # depth from MediaPipe
            out[f, bidx] = (dx, dy, z)
            ovis[f, bidx] = rs[f, hidx]

    return {"world_landmarks": out, "visibility": ovis,
            "image_landmarks": mp["image_landmarks"],
            "fps": mp.get("fps", rtm.get("fps", 30.0)),
            "width": mp["width"], "height": mp["height"]}
