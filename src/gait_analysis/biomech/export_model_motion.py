"""Export an OpenSim model + motion to a self-contained 3D scene (layer-2 visuals).

Drives the model's body transforms from a .mot via the OpenSim API and pairs each body
with its mesh geometry (the .vtp bone files that ship with the model), producing a folder
the browser can play back: motion.json (transforms per frame, inlined into the HTML) +
geometry/*.vtp (loaded with three.js VTKLoader) + model_viewer.html.

Runs where OpenSim is installed (the Mac / conda 'gait' env). Serve the output folder
(e.g. `python -m http.server` in it) and open model_viewer.html.

    gait-export-3d --model LaiUhlrich2022_ga.osim --mot coordinates.mot --out scene/
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from ..analysis import kinematics

_TRANSLATIONAL_SUFFIXES = ("_tx", "_ty", "_tz")


def _viewer_html(scene: dict) -> str:
    """Self-contained viewer: motion inlined, meshes fetched from ./geometry via VTKLoader."""
    return _VIEWER_TMPL.replace("__SCENE__", json.dumps(scene))


def export_scene(osim_path, mot_path, out_dir, max_frames: int = 150, geometry=None) -> Path:
    import opensim as osim

    out_dir = Path(out_dir)
    geo_out = out_dir / "geometry"
    geo_out.mkdir(parents=True, exist_ok=True)

    time, coords, meta = kinematics.read_storage(mot_path)
    in_deg = bool(meta.get("inDegrees", True))
    T = len(time)
    step = max(1, -(-T // max_frames))
    idxs = list(range(0, T, step))
    fps = (1.0 / (time[1] - time[0])) / step if T > 1 else 30.0

    # Where the bone .vtp meshes might live (explicit --geometry wins).
    osim_dir = Path(osim_path).resolve().parent
    search_dirs = [osim_dir, osim_dir / "Geometry", osim_dir.parent / "Geometry"]
    if geometry:
        search_dirs.insert(0, Path(geometry).expanduser())
    for d in search_dirs:                       # let OpenSim resolve meshes from these dirs too
        try:
            osim.ModelVisualizer.addDirToGeometrySearchPaths(str(d))
        except Exception:
            pass

    model = osim.Model(str(osim_path))
    state = model.initSystem()

    # --- collect mesh geometry per body, copy the .vtp files ---
    comp_list = (model.getComponentsList if hasattr(model, "getComponentsList")
                 else model.getComponentList)()
    bodies_geo: dict[str, list] = {}
    for comp in comp_list:
        mesh = osim.Mesh.safeDownCast(comp)
        if mesh is None:
            continue
        fname = mesh.getGeometryFilename()
        base = osim.PhysicalFrame.safeDownCast(mesh.getFrame().findBaseFrame())
        if base is None:
            continue
        xbf = mesh.getFrame().findTransformInBaseFrame()
        p, q = xbf.p(), xbf.R().convertRotationToQuaternion()
        sc = mesh.get_scale_factors()
        src = next((d / fname for d in search_dirs if (d / fname).exists()), None)
        if src is not None:
            shutil.copy(src, geo_out / Path(fname).name)
        bodies_geo.setdefault(base.getName(), []).append({
            "file": f"geometry/{Path(fname).name}",
            "found": src is not None,
            "scale": [sc.get(0), sc.get(1), sc.get(2)],
            "offset_pos": [p.get(0), p.get(1), p.get(2)],
            "offset_quat": [q.get(1), q.get(2), q.get(3), q.get(0)],   # x,y,z,w
        })

    # --- per-frame body transforms in ground ---
    cs = model.getCoordinateSet()
    frames = []
    for f in idxs:
        for i in range(cs.getSize()):
            c = cs.get(i)
            name = c.getName()
            if name not in coords:
                continue
            v = float(coords[name][f])
            if in_deg and not name.endswith(_TRANSLATIONAL_SUFFIXES):
                v = math.radians(v)
            try:
                c.setValue(state, v, False)
            except Exception:
                pass
        model.assemble(state)
        model.realizePosition(state)
        fr = {}
        for bn in bodies_geo:
            tg = model.getBodySet().get(bn).getTransformInGround(state)
            p, q = tg.p(), tg.R().convertRotationToQuaternion()
            fr[bn] = [p.get(0), p.get(1), p.get(2), q.get(1), q.get(2), q.get(3), q.get(0)]
        frames.append(fr)

    scene = {"fps": round(float(fps), 2),
             "bodies": [{"name": k, "meshes": v} for k, v in bodies_geo.items()],
             "frames": frames}
    (out_dir / "motion.json").write_text(json.dumps(scene))
    (out_dir / "model_viewer.html").write_text(_viewer_html(scene))
    missing = [m["file"] for v in bodies_geo.values() for m in v if not m["found"]]
    if missing:
        print(f"[warn] {len(missing)} mesh file(s) not found near {osim_dir} "
              f"(set the model's Geometry folder alongside it): {missing[:4]}...")
    print(f"Wrote {out_dir}/model_viewer.html ({len(frames)} frames, {len(bodies_geo)} bodies). "
          f"Serve the folder (python -m http.server) and open it.")
    return out_dir


_VIEWER_TMPL = """<!doctype html><html><head><meta charset="utf-8"><title>OpenSim model playback</title>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<style>body{margin:0;background:#0e1116;color:#cbd5e1;font:14px -apple-system,Segoe UI,Roboto,sans-serif}
#bar{position:fixed;bottom:0;left:0;right:0;padding:10px;background:#0e1116cc;display:flex;gap:10px;align-items:center}
button{background:#2a9d8f;color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}
input[type=range]{flex:1}</style></head><body>
<div id="bar"><button id="play">&#10073;&#10073;</button><input id="scrub" type="range" min="0" value="0"><span id="t"></span></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {VTKLoader} from 'three/addons/loaders/VTKLoader.js';
const S = __SCENE__;
const scene=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 100); cam.position.set(2.2,1.2,2.2);
const r=new THREE.WebGLRenderer({antialias:true}); r.setSize(innerWidth,innerHeight); document.body.appendChild(r.domElement);
const ctrl=new OrbitControls(cam,r.domElement); ctrl.target.set(0,0.9,0);
scene.add(new THREE.HemisphereLight(0xffffff,0x223,1.1));
const dl=new THREE.DirectionalLight(0xffffff,0.7); dl.position.set(2,4,3); scene.add(dl);
scene.add(new THREE.GridHelper(4,16,0x334155,0x1f2937));
const mat=new THREE.MeshLambertMaterial({color:0xe8e2d6});
const groups={}; const loader=new VTKLoader();
for(const b of S.bodies){const g=new THREE.Group(); groups[b.name]=g; scene.add(g);
  for(const m of b.meshes){ if(!m.found) continue;
    loader.load(m.file, geom=>{geom.computeVertexNormals(); const mesh=new THREE.Mesh(geom,mat);
      mesh.scale.set(...m.scale); mesh.position.set(...m.offset_pos);
      mesh.quaternion.set(...m.offset_quat); g.add(mesh);}); } }
let f=0, playing=true, last=0;
const scrub=document.getElementById('scrub'), play=document.getElementById('play'), tlab=document.getElementById('t');
scrub.max=S.frames.length-1;
play.onclick=()=>{playing=!playing; play.innerHTML=playing?'&#10073;&#10073;':'&#9654;';};
scrub.oninput=()=>{f=+scrub.value; playing=false; play.innerHTML='&#9654;'; apply();};
function apply(){const fr=S.frames[f];
  for(const name in groups){const t=fr[name]; if(!t) continue;
    groups[name].position.set(t[0],t[1],t[2]); groups[name].quaternion.set(t[3],t[4],t[5],t[6]);}
  tlab.textContent=(f/S.fps).toFixed(2)+'s';}
function loop(ts){requestAnimationFrame(loop);
  if(playing && ts-last>1000/S.fps){f=(f+1)%S.frames.length; scrub.value=f; last=ts;}
  apply(); ctrl.update(); r.render(scene,cam);}
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight; cam.updateProjectionMatrix(); r.setSize(innerWidth,innerHeight);});
requestAnimationFrame(loop);
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export an OpenSim model+motion to a 3D viewer folder")
    ap.add_argument("--model", required=True, help="Scaled .osim")
    ap.add_argument("--mot", required=True, help="Motion .mot (coordinates)")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--geometry", default=None, help="Folder holding the model's bone .vtp meshes")
    args = ap.parse_args(argv)
    export_scene(args.model, args.mot, args.out, args.max_frames, geometry=args.geometry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
