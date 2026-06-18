"""Synced side-by-side viewer: the video with tracked markers (left) + OpenSim model (right).

The OpenCap-style view. Left is the original video with the 2D keypoints drawn over it on a
canvas; right is the 3D rendering. The right pane renders, in priority order:
  - the actual OpenSim musculoskeletal model (bone .vtp meshes) when a model + .mot are given
    (via export_model_motion), or
  - the marker skeleton from a .trc as a fallback.
Both panes are driven by the SAME video timeline, so play/pause/scrub on the video moves the
3D in lock-step.

Builds from artifacts the pipeline already produces: the pose .npz (rtmpose COCO-17 pixel
keypoints, or mediapipe BlazePose-33 image landmarks) + the original video + (model + .mot)
or a .trc.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .viz3d import trc_to_scene

# Skeleton links by index for the two pose formats we export.
_COCO17 = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
           (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
_BLAZE33 = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26),
            (26, 28), (28, 30), (30, 32), (28, 32), (0, 11), (0, 12)]
# Gait skeleton: head + trunk + both legs, NO arms. The arms aren't used for sagittal
# gait, and the far (occluded) arm's wrist is the main source of markers that "float"
# off the body in a monocular side view — so we don't draw them at all.
_GAIT_LINKS = [(11, 12), (11, 23), (12, 24), (23, 24),
               (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),   # left leg + foot
               (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),   # right leg + foot
               (0, 11), (0, 12)]                                   # head


def kpts2d_from_npz(npz_path, min_score: float = 0.5) -> dict:
    """2D overlay scene: per-frame pixel keypoints + skeleton links + video dimensions.

    For mediapipe input, whole frames where the body is clipped/low-confidence (the
    subject entering or leaving the frame) are blanked, and the kept frames are
    de-jittered, so the overlay only ever shows clean tracking.
    """
    data = np.load(npz_path)
    w, h = int(data["width"]), int(data["height"])
    fps = float(data["fps"]) or 30.0
    frame_valid = None
    if "keypoints" in data:                       # rtmpose COCO-17, pixel coords
        kp = data["keypoints"].astype(float)
        sc = data["scores"].astype(float) if "scores" in data else np.ones(kp.shape[:2])
        links = _COCO17
    elif "image_landmarks" in data:               # mediapipe BlazePose-33, normalized
        from .sagittal2d import smooth_along_time, valid_frame_mask
        norm = data["image_landmarks"].astype(float)
        sc = data["visibility"].astype(float) if "visibility" in data else np.ones(norm.shape[:2])
        frame_valid = valid_frame_mask(norm, sc)          # drop entry/exit junk frames
        norm = smooth_along_time(norm)                    # de-jitter the kept frames
        kp = norm
        kp[..., 0] *= w
        kp[..., 1] *= h
        links = _GAIT_LINKS
    else:
        raise ValueError("npz lacks 'keypoints' (rtmpose) or 'image_landmarks' (mediapipe)")

    # Draw only joints that are part of the skeleton — drops MediaPipe's finger/face
    # detail points (17-22, 1-10) that otherwise render as a loose dot cluster near the
    # hands/head and read as "markers are off".
    drawn = {i for lk in links for i in lk}

    frames = []
    for f in range(kp.shape[0]):
        if frame_valid is not None and not frame_valid[f]:
            frames.append([None] * kp.shape[1])           # blank: don't draw on junk frames
            continue
        row = []
        for j in range(kp.shape[1]):
            x, y = kp[f, j]
            row.append(None if (j not in drawn or not np.isfinite([x, y]).all() or sc[f, j] < min_score)
                       else [round(float(x), 1), round(float(y), 1)])
        frames.append(row)
    return {"fps": round(fps, 3), "w": w, "h": h, "links": links, "frames": frames}


def render_overlay_video(video_path, npz_path, out_path, min_score: float = 0.5):
    """Burn the gait skeleton INTO the video pixels (same coordinate space as the
    landmarks), so on-screen alignment is GUARANTEED regardless of how the browser
    handles rotation/letterbox/scaling.

    Encoding: OpenCV reliably writes mp4v (MPEG-4 Part 2) on every build, but browsers
    can't PLAY mp4v, and the H.264 (avc1) encoder is missing from the Linux OpenCV
    wheel. So we write a temp mp4v with OpenCV, then transcode to browser-playable
    H.264 with the ffmpeg CLI (libx264). Returns out_path, or None (→ caller falls
    back to the browser canvas overlay) if ffmpeg or video writing is unavailable.
    """
    import os
    import shutil as _sh
    import subprocess

    import cv2
    from .sagittal2d import smooth_along_time, valid_frame_mask

    if _sh.which("ffmpeg") is None:        # need ffmpeg to make a playable H.264 file
        return None
    d = np.load(npz_path)
    if "image_landmarks" not in d:
        return None
    norm = smooth_along_time(d["image_landmarks"].astype(float))
    vis = d["visibility"].astype(float) if "visibility" in d else np.ones(norm.shape[:2])
    valid = valid_frame_mask(d["image_landmarks"].astype(float), vis)
    drawn = {i for lk in _GAIT_LINKS for i in lk}
    fps = float(d["fps"]) if "fps" in d else 30.0

    out_path = Path(out_path)
    tmp = out_path.with_suffix(".raw.mp4")        # OpenCV-written mp4v, then transcoded
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)   # decode upright (match extraction)
    except Exception:
        pass
    writer, idx, drew = None, 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps or 30.0, (w, h))
                if not writer.isOpened():
                    writer.release()
                    return None
            if idx < len(norm) and (idx >= len(valid) or valid[idx]):
                pts = []
                for j in range(norm.shape[1]):
                    x, y = norm[idx, j]
                    pts.append(None if (j not in drawn or not np.isfinite([x, y]).all()
                                        or vis[idx, j] < min_score)
                               else (int(round(x * w)), int(round(y * h))))
                for a, b in _GAIT_LINKS:
                    if pts[a] and pts[b]:
                        cv2.line(frame, pts[a], pts[b], (255, 200, 120), 2, cv2.LINE_AA)
                for p in pts:
                    if p:
                        cv2.circle(frame, p, 5, (90, 220, 90), -1, cv2.LINE_AA)
                drew += 1
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if idx == 0 or drew == 0 or not tmp.exists():
        return None
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(out_path)], check=True)
    except Exception as exc:
        print(f"[note] ffmpeg transcode failed: {exc}")
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else None


def synced_html(video_name: str, kpts2d: dict, scene3d: dict | None, mode: str,
                burned: bool = False) -> str:
    """mode: 'model' (OpenSim meshes), 'markers' (.trc skeleton), or 'none'."""
    return (_TMPL
            .replace("__VIDEO__", json.dumps(video_name))
            .replace("__KP__", json.dumps(kpts2d))
            .replace("__MODE__", json.dumps(mode))
            .replace("__BURNED__", json.dumps(bool(burned)))
            .replace("__SCENE__", json.dumps(scene3d)))


def _world_landmarks_to_scene3d(data, min_score: float = 0.5) -> dict:
    """Build a 3D marker scene from mediapipe world_landmarks stored in a loaded NPZ.

    Uses the SAME frame-validity gate as the 2D overlay so the 3D body and the video
    skeleton appear/disappear together, and de-jitters the kept frames.
    """
    from .sagittal2d import smooth_along_time, valid_frame_mask
    wl = smooth_along_time(data["world_landmarks"].astype(float))   # (T,33,3) hip-centred m
    vis = data["visibility"].astype(float) if "visibility" in data else np.ones(wl.shape[:2])
    fps = float(data["fps"]) if "fps" in data else 30.0
    frame_valid = (valid_frame_mask(data["image_landmarks"].astype(float), vis)
                   if "image_landmarks" in data else None)
    drawn = {i for lk in _GAIT_LINKS for i in lk}   # head+trunk+legs, no arms/fingers/face
    frames = []
    for f in range(wl.shape[0]):
        if frame_valid is not None and not frame_valid[f]:
            frames.append([None] * wl.shape[1])
            continue
        row = []
        for j in range(wl.shape[1]):
            x, y, z = wl[f, j]
            v = float(vis[f, j])
            # MediaPipe: x=right, y=down (image), z=depth toward camera
            # Three.js:  x=right, y=up,           z=toward viewer
            row.append([round(float(x), 4), round(float(-y), 4), round(float(-z), 4)]
                       if j in drawn and np.isfinite([x, y, z]).all() and v >= min_score else None)
        frames.append(row)
    return {"fps": round(fps, 3), "names": [str(i) for i in range(33)],
            "links": _GAIT_LINKS, "frames": frames}


def build(video_path, npz_path, out_dir, trc_path=None, model=None, mot=None,
          max_frames: int = 150, geometry=None) -> Path:
    """Build the synced viewer folder. If model+mot given, the right pane is the OpenSim model
    (writes geometry/ into out_dir); else if trc given, it's the marker skeleton; else falls
    back to the 3D world landmarks from the pose NPZ."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)

    # Prefer a BURNED-IN overlay (skeleton drawn into the pixels): alignment is then
    # guaranteed because it lives in the same coordinate space as the landmarks. Fall
    # back to the browser canvas overlay only if video re-encoding is unavailable.
    burned, left_video = False, video_path.name
    try:
        if render_overlay_video(video_path, npz_path, out_dir / "overlay.mp4") is not None:
            burned, left_video = True, "overlay.mp4"
    except Exception as exc:
        print(f"[note] burned overlay failed ({exc}); using browser overlay")
    if not burned:
        shutil.copy(video_path, out_dir / video_path.name)
    kpts2d = kpts2d_from_npz(npz_path)

    scene3d, mode = None, "none"
    if model and mot:
        # Render the actual OpenSim model: export geometry + per-frame body transforms here.
        from ..biomech.export_model_motion import export_scene
        export_scene(model, mot, out_dir, max_frames=max_frames, geometry=geometry)   # writes out_dir/{geometry,motion.json}
        scene3d = json.loads((out_dir / "motion.json").read_text())
        mode = "model"
    elif trc_path:
        scene3d, mode = trc_to_scene(trc_path), "markers"
    else:
        # Fallback: use 3D world landmarks embedded in the pose NPZ (always available from
        # mediapipe extraction) — shows the body skeleton without needing TRC or OpenSim.
        try:
            d = np.load(npz_path)
            if "world_landmarks" in d:
                scene3d = _world_landmarks_to_scene3d(d)
                mode = "markers"
        except Exception as exc:
            print(f"[note] world-landmarks 3D fallback failed: {exc}")

    (out_dir / "viewer.html").write_text(synced_html(left_video, kpts2d, scene3d, mode, burned))
    print(f"Wrote {out_dir}/viewer.html (left: {'burned-in overlay' if burned else 'canvas overlay'}, "
          f"right pane: {mode}).")
    return out_dir / "viewer.html"


_TMPL = """<!doctype html><html><head><meta charset="utf-8"><title>Video + OpenSim (synced)</title>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<style>body{margin:0;background:#0e1116;color:#cbd5e1;font:14px -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{display:flex;flex-wrap:wrap;gap:8px;padding:8px}.pane{flex:1 1 420px;position:relative;min-width:360px}
#vid{width:100%;display:block;border-radius:8px;background:#000}#ov{position:absolute;left:0;top:0;pointer-events:none}
#viz{width:100%;height:62vh;background:#0e1116;border-radius:8px}h3{margin:6px 2px;font-weight:600}
.tag{font-weight:400;color:#7c8aa0;font-size:12px}</style></head><body>
<div class="wrap">
  <div class="pane"><h3>Video + tracked markers</h3><video id="vid" src=__VIDEO__ controls playsinline></video><canvas id="ov"></canvas></div>
  <div class="pane"><h3>OpenSim rendering <span class="tag" id="rkind"></span></h3><div id="viz"></div></div>
</div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {VTKLoader} from 'three/addons/loaders/VTKLoader.js';
const KP=__KP__, S=__SCENE__, MODE=__MODE__, BURNED=__BURNED__, vid=document.getElementById('vid');
document.getElementById('rkind').textContent = MODE==='model'?'(musculoskeletal model)':MODE==='markers'?'(marker skeleton)':'(no 3D)';
// ---- left: 2D overlay on canvas, mapped to the ACTUAL displayed video rect ----
// The browser may letterbox the video inside the <video> box (object-fit) and uses
// the video's post-rotation intrinsic size. We map keypoints (in KP.w x KP.h space)
// into the real content rectangle so the skeleton lands ON the person, not above/
// beside them. We also warn if the extracted aspect ratio disagrees with what the
// browser is showing (a sign the source had a rotation flag handled differently).
const ov=document.getElementById('ov'), cx=ov.getContext('2d');
function fit(){ov.width=vid.clientWidth; ov.height=vid.clientHeight;}
vid.addEventListener('loadedmetadata',fit); addEventListener('resize',fit);
function frameOf(fps,n){return Math.min(Math.max(0,Math.round(vid.currentTime*fps)), n-1);}
function rect(){
  // object-fit:contain — the video content centered inside the element box.
  const vw=vid.videoWidth||KP.w, vh=vid.videoHeight||KP.h;
  const cw=ov.width, ch=ov.height, s=Math.min(cw/vw, ch/vh);
  const dw=vw*s, dh=vh*s;
  return {ox:(cw-dw)/2, oy:(ch-dh)/2, sx:dw/KP.w, sy:dh/KP.h};
}
function draw2d(){ if(BURNED) return;   // skeleton is baked into the video pixels
  const r=rect(); cx.clearRect(0,0,ov.width,ov.height);
  const fr=KP.frames[frameOf(KP.fps,KP.frames.length)]||[];
  cx.strokeStyle='#9ecbff'; cx.lineWidth=2;
  for(const [i,j] of KP.links){const a=fr[i],b=fr[j]; if(!a||!b)continue;
    cx.beginPath(); cx.moveTo(r.ox+a[0]*r.sx,r.oy+a[1]*r.sy); cx.lineTo(r.ox+b[0]*r.sx,r.oy+b[1]*r.sy); cx.stroke();}
  cx.fillStyle='#2dd4bf'; for(const p of fr){if(!p)continue; cx.beginPath(); cx.arc(r.ox+p[0]*r.sx,r.oy+p[1]*r.sy,3.5,0,7); cx.fill();}}
// ---- right: three.js scene driven by the same video time ----
let render3d=()=>{};
if(S){const el=document.getElementById('viz'),W=el.clientWidth,H=el.clientHeight||500;
  const scene=new THREE.Scene();
  const cam=new THREE.PerspectiveCamera(50,W/H,0.01,100); cam.position.set(2.4,1.1,2.4);
  const r=new THREE.WebGLRenderer({antialias:true}); r.setSize(W,H); el.appendChild(r.domElement);
  const ctrl=new OrbitControls(cam,r.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff,0x223,1.15));
  const dl=new THREE.DirectionalLight(0xffffff,0.7); dl.position.set(2,4,3); scene.add(dl);
  scene.add(new THREE.GridHelper(4,16,0x334155,0x1f2937));
  if(MODE==='model'){
    ctrl.target.set(0,0.9,0);
    const mat=new THREE.MeshLambertMaterial({color:0xe8e2d6}), groups={}, loader=new VTKLoader();
    for(const b of S.bodies){const g=new THREE.Group(); groups[b.name]=g; scene.add(g);
      for(const m of b.meshes){ if(!m.found) continue;
        loader.load(m.file, geom=>{geom.computeVertexNormals(); const mesh=new THREE.Mesh(geom,mat);
          mesh.scale.set(...m.scale); mesh.position.set(...m.offset_pos); mesh.quaternion.set(...m.offset_quat); g.add(mesh);}); } }
    render3d=()=>{const fr=S.frames[frameOf(S.fps,S.frames.length)];
      for(const name in groups){const t=fr[name]; if(!t)continue;
        groups[name].position.set(t[0],t[1],t[2]); groups[name].quaternion.set(t[3],t[4],t[5],t[6]);}
      ctrl.update(); r.render(scene,cam);};
  } else {                                            // marker skeleton from .trc
    const M=S.names.length, pGeo=new THREE.BufferGeometry(), pArr=new Float32Array(M*3);
    pGeo.setAttribute('position',new THREE.BufferAttribute(pArr,3));
    scene.add(new THREE.Points(pGeo,new THREE.PointsMaterial({color:0x2dd4bf,size:0.055})));
    const lGeo=new THREE.BufferGeometry(), lArr=new Float32Array(Math.max(1,S.links.length)*6);
    lGeo.setAttribute('position',new THREE.BufferAttribute(lArr,3));
    scene.add(new THREE.LineSegments(lGeo,new THREE.LineBasicMaterial({color:0x9ecbff})));
    render3d=()=>{const fr=S.frames[frameOf(S.fps,S.frames.length)];
      for(let m=0;m<M;m++){const p=fr[m]; pArr[m*3]=p?p[0]:NaN; pArr[m*3+1]=p?p[1]:NaN; pArr[m*3+2]=p?p[2]:NaN;}
      pGeo.attributes.position.needsUpdate=true;
      for(let k=0;k<S.links.length;k++){const a=fr[S.links[k][0]],b=fr[S.links[k][1]];
        lArr[k*6]=a?a[0]:NaN;lArr[k*6+1]=a?a[1]:NaN;lArr[k*6+2]=a?a[2]:NaN;
        lArr[k*6+3]=b?b[0]:NaN;lArr[k*6+4]=b?b[1]:NaN;lArr[k*6+5]=b?b[2]:NaN;}
      lGeo.attributes.position.needsUpdate=true; ctrl.update(); r.render(scene,cam);};
  }}
// Re-fit the canvas to the video EVERY frame: the two-pane flex layout and the
// three.js panel reflow after metadata loads, so a one-time fit() leaves the canvas
// at a stale size and the skeleton lands offset (e.g. above the person).
function loop(){requestAnimationFrame(loop); fit(); draw2d(); render3d();}
fit(); loop();
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synced video+markers / OpenSim 3D viewer folder")
    ap.add_argument("--video", required=True)
    ap.add_argument("--npz", required=True, help="pose .npz (rtmpose or mediapipe)")
    ap.add_argument("--model", default=None, help="OpenSim .osim (right pane = the model)")
    ap.add_argument("--mot", default=None, help="OpenSim .mot driving the model")
    ap.add_argument("--trc", default=None, help="marker .trc (right pane fallback if no model)")
    ap.add_argument("--geometry", default=None, help="Folder holding the model's bone .vtp meshes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    build(args.video, args.npz, args.out, trc_path=args.trc, model=args.model, mot=args.mot,
          geometry=args.geometry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
