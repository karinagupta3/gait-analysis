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
import re
import shutil
from pathlib import Path

from ..analysis import kinematics

_TRANSLATIONAL_SUFFIXES = ("_tx", "_ty", "_tz")

# Skip the tiny hand/finger bone meshes — irrelevant to gait and ~50 extra meshes
# that bog down the browser. Keep pelvis/legs/feet/trunk/skull/arms.
_SKIP_MESH = ("capitate", "hamate", "lunate", "metacarpal", "pisiform", "scaphoid",
              "trapezium", "trapezoid", "triquetrum", "index_", "middle_", "ring_",
              "little_", "thumb_", "_distal", "_medial", "_proximal")


def _mesh_candidates(fname: str) -> list[str]:
    """Filenames to try for a referenced mesh. The LaiUhlrich model names leg bones
    side-FIRST (l_femur.vtp) but Pose2Sim's bundled geometry names them side-LAST
    (femur_l.vtp, talus_lv.vtp). Try both orderings (and the talus 'v' variant)."""
    name = Path(fname).name
    stem = Path(fname).stem
    out = [name]
    m = re.match(r"^([lr])_(.+)$", stem)              # l_femur -> femur_l / femur_lv / femur
    if m:
        side, bone = m.group(1), m.group(2)
        out += [f"{bone}_{side}.vtp", f"{bone}_{side}v.vtp", f"{bone}.vtp"]
    m2 = re.match(r"^(.+?)_([lr])v?$", stem)           # femur_l / talus_lv -> l_femur
    if m2:
        bone, side = m2.group(1), m2.group(2)
        out += [f"{side}_{bone}.vtp"]
    return out


def _resolve_mesh(fname: str, search_dirs):
    """First existing file among the name candidates across the search dirs, or None."""
    for cand in _mesh_candidates(fname):
        for d in search_dirs:
            if (d / cand).exists():
                return d / cand
    return None


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
    # Pose2Sim ships the LaiUhlrich2022 bone meshes (OpenSim_Setup/Geometry); use them
    # so the cloud worker (which has Pose2Sim) can render bones with no extra assets.
    try:
        import Pose2Sim
        search_dirs.append(Path(Pose2Sim.__file__).resolve().parent / "OpenSim_Setup" / "Geometry")
    except Exception:
        pass
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
        if any(t in fname.lower() for t in _SKIP_MESH):   # drop hand/finger detail
            continue
        base = osim.PhysicalFrame.safeDownCast(mesh.getFrame().findBaseFrame())
        if base is None:
            continue
        xbf = mesh.getFrame().findTransformInBaseFrame()
        p, q = xbf.p(), xbf.R().convertRotationToQuaternion()
        sc = mesh.get_scale_factors()
        src = _resolve_mesh(fname, search_dirs)        # handles l_femur <-> femur_l etc.
        if src is not None:
            dst = geo_out / Path(fname).name
            if src.resolve() != dst.resolve():            # tolerate --geometry pointing at the output dir
                shutil.copy(src, dst)
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

    # Drop the model ONTO the ground plane. The monocular pipeline outputs hip-centred
    # coordinates with no ground reference, so the pelvis sat at y=0 and the legs/feet
    # rendered BELOW the floor grid. Shift every body up by the lowest body height over
    # the whole clip (preserving vertical motion) so the lowest point rests on y=0.
    ys = [t[1] for fr in frames for t in fr.values()]
    if ys:
        y_off = min(ys)
        for fr in frames:
            for t in fr.values():
                t[1] -= y_off

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
<script>
// Classic script (runs first): a persistent on-page banner. If you do NOT see "viewer build B3"
// at the top, you are looking at a cached old page -> open in an Incognito window.
window.__say=function(m,bg){var d=document.getElementById('msg');if(!d){d=document.createElement('div');d.id='msg';
 d.style.cssText='position:fixed;top:0;left:0;right:0;z-index:9;font:12px/1.4 monospace;padding:6px 10px;white-space:pre-wrap;color:#fff';
 (document.body||document.documentElement).appendChild(d);} d.style.background=bg||'#1e3a5f'; d.textContent=m;};
window.__say('viewer build B3 -- initializing 3D (if this banner stays blue, the module never ran)...');
addEventListener('error',function(e){window.__say('JS ERROR: '+(e.message||e.filename||e)+(e.lineno?(' (line '+e.lineno+')'):''),'#7f1d1d');});
addEventListener('unhandledrejection',function(e){window.__say('PROMISE ERROR: '+((e.reason&&e.reason.message)||e.reason),'#7f1d1d');});
</script>
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
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x12161d);   // visibly not pure black if WebGL renders
const cam=new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 100);
const r=new THREE.WebGLRenderer({antialias:true}); r.setSize(innerWidth,innerHeight); document.body.appendChild(r.domElement);
const ctrl=new OrbitControls(cam,r.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x223,1.2));
const dl=new THREE.DirectionalLight(0xffffff,0.8); dl.position.set(2,4,3); scene.add(dl);
scene.add(new THREE.GridHelper(4,16,0x55617a,0x333c52));
const names=S.bodies.map(b=>b.name);
// Aim the camera at the model's actual location at frame 0 (so it can't be off-screen).
let cx=0,cy=0.9,cz=0,nb=0; const f0=S.frames[0]||{};
for(const k in f0){cx+=f0[k][0]; cy+=f0[k][1]; cz+=f0[k][2]; nb++;}
if(nb){cx/=nb; cy/=nb; cz/=nb;} ctrl.target.set(cx,cy,cz); cam.position.set(cx+2.2,cy+1.0,cz+2.2);
// GUARANTEED-visible body-origin point cloud (needs no meshes) -> you always see the skeleton move.
const oGeo=new THREE.BufferGeometry(), oArr=new Float32Array(names.length*3);
oGeo.setAttribute('position',new THREE.BufferAttribute(oArr,3));
scene.add(new THREE.Points(oGeo,new THREE.PointsMaterial({color:0x2dd4bf,size:0.05})));
// OpenSim bone meshes (enhancement on top of the points).
const mat=new THREE.MeshLambertMaterial({color:0xe8e2d6}); const groups={}, loader=new VTKLoader(); let want=0,got=0,fail=0;
function status(extra){window.__say('bodies '+names.length+' · frames '+S.frames.length+' · meshes '+got+'/'+want+(fail?(' · FAILED '+fail):'')+(extra?(' -- '+extra):''), fail?'#7f1d1d':'#14532d');}
for(const b of S.bodies){const g=new THREE.Group(); groups[b.name]=g; scene.add(g);
  for(const m of b.meshes){ if(!m.found) continue; want++;
    loader.load(m.file, geom=>{geom.computeVertexNormals(); const mesh=new THREE.Mesh(geom,mat);
      mesh.scale.set(...m.scale); mesh.position.set(...m.offset_pos); mesh.quaternion.set(...m.offset_quat); g.add(mesh); got++; status();},
      undefined,
      err=>{fail++; status('VTKLoader cannot read '+m.file);}); } }
status('rendering');
let f=0, playing=true, last=0;
const scrub=document.getElementById('scrub'), play=document.getElementById('play'), tlab=document.getElementById('t');
scrub.max=S.frames.length-1;
play.onclick=()=>{playing=!playing; play.innerHTML=playing?'&#10073;&#10073;':'&#9654;';};
scrub.oninput=()=>{f=+scrub.value; playing=false; play.innerHTML='&#9654;'; apply();};
function apply(){const fr=S.frames[f];
  for(let i=0;i<names.length;i++){const t=fr[names[i]];
    if(!t){oArr[i*3]=NaN; continue;}
    oArr[i*3]=t[0]; oArr[i*3+1]=t[1]; oArr[i*3+2]=t[2];
    const g=groups[names[i]]; if(g){g.position.set(t[0],t[1],t[2]); g.quaternion.set(t[3],t[4],t[5],t[6]);}}
  oGeo.attributes.position.needsUpdate=true;
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
