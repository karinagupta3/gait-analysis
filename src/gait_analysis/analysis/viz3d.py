"""Animated 3D motion playback for the report (OpenCap-style), from a marker .trc.

This is the first layer of "OpenSim visuals": the reconstructed body moving in 3D, built
from the marker trajectories the pipeline already produces. It renders in the browser with
three.js (markers + anatomical links, play/scrub). The next layer (loading the OpenSim
musculoskeletal *model geometry* and driving body transforms from the .mot) is scaffolded
in biomech/export_model_motion.py and slots into the same viewer.
"""

from __future__ import annotations

import json

import numpy as np

from .spatiotemporal_3d import read_trc

# Anatomical links by marker name (covers HALPE_26 / Pose2Sim and our blazepose markerset).
_LINKS = [
    ("RShoulder", "LShoulder"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("RShoulder", "RHip"), ("LShoulder", "LHip"), ("RHip", "LHip"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("RAnkle", "RHeel"), ("RAnkle", "RBigToe"),
    ("RHeel", "RBigToe"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"), ("LAnkle", "LHeel"), ("LAnkle", "LBigToe"),
    ("LHeel", "LBigToe"),
    ("Neck", "Head"), ("Neck", "Nose"), ("Neck", "RShoulder"), ("Neck", "LShoulder"),
    ("Hip", "Neck"),
]


def trc_to_scene(trc_path, max_frames: int = 200) -> dict:
    times, names, pos = read_trc(trc_path)        # pos (T, M, 3), metres
    T, M, _ = pos.shape
    step = max(1, -(-T // max_frames))   # ceil division so result <= max_frames
    sub = pos[::step]                              # (F, M, 3)
    center = np.nanmean(sub.reshape(-1, 3), axis=0)
    sub = sub - center                             # center the figure at the origin
    fps = (1.0 / np.median(np.diff(times))) / step if T > 1 else 30.0

    name_idx = {n: i for i, n in enumerate(names)}
    links = [[name_idx[a], name_idx[b]] for a, b in _LINKS if a in name_idx and b in name_idx]

    frames = []
    for fr in sub:
        frames.append([None if not np.isfinite(p).all()
                       else [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
                       for p in fr])
    return {"names": names, "fps": round(float(fps), 2), "links": links, "frames": frames}


_VIEWER = """<section><h2>3D motion playback</h2>
<p class="meta">Animated reconstruction from the marker .trc &mdash; drag to rotate, scroll to zoom.
The next layer renders the full OpenSim musculoskeletal model (see roadmap).</p>
<div id="viz" style="width:100%;height:460px;background:#0e1116;border-radius:10px"></div>
<div style="margin-top:8px"><button id="viz-play" type="button">&#10073;&#10073; Pause</button>
<input id="viz-scrub" type="range" min="0" max="0" value="0" style="width:68%;vertical-align:middle"></div>
<script type="module">
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import {OrbitControls} from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
const S = __SCENE__;
const el = document.getElementById('viz'), W = el.clientWidth, H = el.clientHeight;
const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(50, W/H, 0.01, 100); cam.position.set(2.4,0.9,2.4);
const r = new THREE.WebGLRenderer({antialias:true}); r.setSize(W,H); r.setPixelRatio(devicePixelRatio);
el.appendChild(r.domElement);
const ctrl = new OrbitControls(cam, r.domElement); ctrl.target.set(0,0,0);
scene.add(new THREE.AmbientLight(0xffffff,1.0));
scene.add(new THREE.GridHelper(4,16,0x334155,0x1f2937));
const M = S.names.length;
const pGeo = new THREE.BufferGeometry();
const pArr = new Float32Array(M*3); pGeo.setAttribute('position', new THREE.BufferAttribute(pArr,3));
scene.add(new THREE.Points(pGeo, new THREE.PointsMaterial({color:0x2dd4bf,size:0.055,sizeAttenuation:true})));
const lGeo = new THREE.BufferGeometry();
const lArr = new Float32Array(Math.max(1,S.links.length)*6);
lGeo.setAttribute('position', new THREE.BufferAttribute(lArr,3));
scene.add(new THREE.LineSegments(lGeo, new THREE.LineBasicMaterial({color:0x9ecbff})));
let f=0, playing=true, last=0;
const scrub=document.getElementById('viz-scrub'), btn=document.getElementById('viz-play');
scrub.max=S.frames.length-1;
btn.onclick=()=>{playing=!playing; btn.innerHTML=playing?'&#10073;&#10073; Pause':'&#9654; Play';};
scrub.oninput=()=>{f=+scrub.value; playing=false; btn.innerHTML='&#9654; Play'; draw();};
function draw(){
  const fr=S.frames[f];
  for(let m=0;m<M;m++){const p=fr[m]; pArr[m*3]=p?p[0]:NaN; pArr[m*3+1]=p?p[1]:NaN; pArr[m*3+2]=p?p[2]:NaN;}
  pGeo.attributes.position.needsUpdate=true;
  for(let k=0;k<S.links.length;k++){const a=fr[S.links[k][0]], b=fr[S.links[k][1]];
    lArr[k*6]=a?a[0]:NaN; lArr[k*6+1]=a?a[1]:NaN; lArr[k*6+2]=a?a[2]:NaN;
    lArr[k*6+3]=b?b[0]:NaN; lArr[k*6+4]=b?b[1]:NaN; lArr[k*6+5]=b?b[2]:NaN;}
  lGeo.attributes.position.needsUpdate=true;
}
function loop(t){requestAnimationFrame(loop);
  if(playing && t-last > 1000/S.fps){f=(f+1)%S.frames.length; scrub.value=f; last=t;}
  draw(); ctrl.update(); r.render(scene,cam);}
requestAnimationFrame(loop);
</script></section>"""


def report_section(trc_path) -> str:
    """HTML section with the 3D viewer, or '' if the .trc can't be turned into a scene."""
    try:
        scene = trc_to_scene(trc_path)
        if not scene["frames"] or not scene["links"]:
            return ""
        return _VIEWER.replace("__SCENE__", json.dumps(scene))
    except Exception:
        return ""
