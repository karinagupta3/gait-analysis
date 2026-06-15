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


def kpts2d_from_npz(npz_path, min_score: float = 0.3) -> dict:
    """2D overlay scene: per-frame pixel keypoints + skeleton links + video dimensions."""
    data = np.load(npz_path)
    w, h = int(data["width"]), int(data["height"])
    fps = float(data["fps"]) or 30.0
    if "keypoints" in data:                       # rtmpose COCO-17, pixel coords
        kp = data["keypoints"].astype(float)
        sc = data["scores"].astype(float) if "scores" in data else np.ones(kp.shape[:2])
        links = _COCO17
    elif "image_landmarks" in data:               # mediapipe BlazePose-33, normalized
        kp = data["image_landmarks"].astype(float)
        kp[..., 0] *= w
        kp[..., 1] *= h
        sc = data["visibility"].astype(float) if "visibility" in data else np.ones(kp.shape[:2])
        links = _BLAZE33
    else:
        raise ValueError("npz lacks 'keypoints' (rtmpose) or 'image_landmarks' (mediapipe)")

    frames = []
    for f in range(kp.shape[0]):
        row = []
        for j in range(kp.shape[1]):
            x, y = kp[f, j]
            row.append(None if (not np.isfinite([x, y]).all() or sc[f, j] < min_score)
                       else [round(float(x), 1), round(float(y), 1)])
        frames.append(row)
    return {"fps": round(fps, 3), "w": w, "h": h, "links": links, "frames": frames}


def synced_html(video_name: str, kpts2d: dict, scene3d: dict | None, mode: str) -> str:
    """mode: 'model' (OpenSim meshes), 'markers' (.trc skeleton), or 'none'."""
    return (_TMPL
            .replace("__VIDEO__", json.dumps(video_name))
            .replace("__KP__", json.dumps(kpts2d))
            .replace("__MODE__", json.dumps(mode))
            .replace("__SCENE__", json.dumps(scene3d)))


def build(video_path, npz_path, out_dir, trc_path=None, model=None, mot=None,
          max_frames: int = 150) -> Path:
    """Build the synced viewer folder. If model+mot given, the right pane is the OpenSim model
    (writes geometry/ into out_dir); else if trc given, it's the marker skeleton."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)
    shutil.copy(video_path, out_dir / video_path.name)
    kpts2d = kpts2d_from_npz(npz_path)

    scene3d, mode = None, "none"
    if model and mot:
        # Render the actual OpenSim model: export geometry + per-frame body transforms here.
        from ..biomech.export_model_motion import export_scene
        export_scene(model, mot, out_dir, max_frames=max_frames)   # writes out_dir/{geometry,motion.json}
        scene3d = json.loads((out_dir / "motion.json").read_text())
        mode = "model"
    elif trc_path:
        scene3d, mode = trc_to_scene(trc_path), "markers"

    (out_dir / "viewer.html").write_text(synced_html(video_path.name, kpts2d, scene3d, mode))
    print(f"Wrote {out_dir}/viewer.html (right pane: {mode}). "
          f"Serve the folder (python -m http.server) and open viewer.html.")
    return out_dir / "viewer.html"


_TMPL = """<!doctype html><html><head><meta charset="utf-8"><title>Video + OpenSim (synced)</title>
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
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import {OrbitControls} from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import {VTKLoader} from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/VTKLoader.js';
const KP=__KP__, S=__SCENE__, MODE=__MODE__, vid=document.getElementById('vid');
document.getElementById('rkind').textContent = MODE==='model'?'(musculoskeletal model)':MODE==='markers'?'(marker skeleton)':'(no 3D)';
// ---- left: 2D overlay on canvas, scaled to displayed video size ----
const ov=document.getElementById('ov'), cx=ov.getContext('2d');
function fit(){ov.width=vid.clientWidth; ov.height=vid.clientHeight;}
vid.addEventListener('loadedmetadata',fit); addEventListener('resize',fit);
function frameOf(fps,n){return Math.min(Math.max(0,Math.round(vid.currentTime*fps)), n-1);}
function draw2d(){const sx=ov.width/KP.w, sy=ov.height/KP.h; cx.clearRect(0,0,ov.width,ov.height);
  const fr=KP.frames[frameOf(KP.fps,KP.frames.length)]||[];
  cx.strokeStyle='#9ecbff'; cx.lineWidth=2;
  for(const [i,j] of KP.links){const a=fr[i],b=fr[j]; if(!a||!b)continue;
    cx.beginPath(); cx.moveTo(a[0]*sx,a[1]*sy); cx.lineTo(b[0]*sx,b[1]*sy); cx.stroke();}
  cx.fillStyle='#2dd4bf'; for(const p of fr){if(!p)continue; cx.beginPath(); cx.arc(p[0]*sx,p[1]*sy,3.5,0,7); cx.fill();}}
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
function loop(){requestAnimationFrame(loop); draw2d(); render3d();}
fit(); loop();
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synced video+markers / OpenSim 3D viewer folder")
    ap.add_argument("--video", required=True)
    ap.add_argument("--npz", required=True, help="pose .npz (rtmpose or mediapipe)")
    ap.add_argument("--model", default=None, help="OpenSim .osim (right pane = the model)")
    ap.add_argument("--mot", default=None, help="OpenSim .mot driving the model")
    ap.add_argument("--trc", default=None, help="marker .trc (right pane fallback if no model)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    build(args.video, args.npz, args.out, trc_path=args.trc, model=args.model, mot=args.mot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
