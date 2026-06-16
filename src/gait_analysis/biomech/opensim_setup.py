"""Generate OpenSim setup XML (IK Task Set + IK Tool) from our marker weights.

Hand-writing OpenSim XML is error-prone, so we generate it from markerset.py. The
output is plain OpenSim 4.x XML that the InverseKinematicsTool reads directly -- no
OpenSim install needed to *write* it (only to *run* it), which keeps generation
testable offline.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from .markerset import IK_MARKER_WEIGHTS

OPENSIM_DOC_VERSION = "40000"  # OpenSim 4.x document version


def _rel_to(setup_path: str | Path, target: str) -> str:
    """Path of `target` relative to the setup file's directory.

    OpenSim's ScaleTool (ModelScaler/MarkerPlacer) resolves marker/output paths
    relative to the setup-file directory and doubles absolute paths
    (e.g. ``/tmp//tmp/walk.trc``), so we emit setup-relative paths.
    """
    setup_dir = Path(setup_path).resolve().parent
    return os.path.relpath(Path(target).resolve(), setup_dir)

# Scaling measurements: each scales body segment(s) from the distance between a pair of
# OUR markers (joint-centre markers from blazepose_to_trc / markerset). Bilateral and
# uniform XYZ scaling -- a sensible default the user can refine. Marker names must match
# the .trc and the marked model.
SCALE_MEASUREMENTS: list[dict] = [
    {"name": "pelvis", "pairs": [("left_hip", "right_hip")],
     "bodies": [("pelvis", "X Y Z")]},
    {"name": "femur_r", "pairs": [("right_hip", "right_knee")],
     "bodies": [("femur_r", "X Y Z")]},
    {"name": "femur_l", "pairs": [("left_hip", "left_knee")],
     "bodies": [("femur_l", "X Y Z")]},
    {"name": "tibia_r", "pairs": [("right_knee", "right_ankle")],
     "bodies": [("tibia_r", "X Y Z")]},
    {"name": "tibia_l", "pairs": [("left_knee", "left_ankle")],
     "bodies": [("tibia_l", "X Y Z")]},
    {"name": "foot_r", "pairs": [("right_heel", "right_foot_index")],
     "bodies": [("calcn_r", "X Y Z"), ("talus_r", "X Y Z"), ("toes_r", "X Y Z")]},
    {"name": "foot_l", "pairs": [("left_heel", "left_foot_index")],
     "bodies": [("calcn_l", "X Y Z"), ("talus_l", "X Y Z"), ("toes_l", "X Y Z")]},
    {"name": "torso", "pairs": [("left_shoulder", "left_hip"), ("right_shoulder", "right_hip")],
     "bodies": [("torso", "X Y Z")]},
    {"name": "humerus_r", "pairs": [("right_shoulder", "right_elbow")],
     "bodies": [("humerus_r", "X Y Z")]},
    {"name": "humerus_l", "pairs": [("left_shoulder", "left_elbow")],
     "bodies": [("humerus_l", "X Y Z")]},
    {"name": "forearm_r", "pairs": [("right_elbow", "right_wrist")],
     "bodies": [("ulna_r", "X Y Z"), ("radius_r", "X Y Z"), ("hand_r", "X Y Z")]},
    {"name": "forearm_l", "pairs": [("left_elbow", "left_wrist")],
     "bodies": [("ulna_l", "X Y Z"), ("radius_l", "X Y Z"), ("hand_l", "X Y Z")]},
]


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print helper (ElementTree has no built-in indent pre-3.9 guarantee)."""
    pad = "\n" + "\t" * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "\t"
        for child in elem:
            _indent(child, level + 1)
            if not (child.tail or "").strip():
                child.tail = pad + "\t"
        if not (elem[-1].tail or "").strip():
            elem[-1].tail = pad
    else:
        if level and not (elem.tail or "").strip():
            elem.tail = pad


def build_ik_task_set(weights: dict[str, float] | None = None) -> ET.Element:
    """Build an <IKTaskSet> element from marker weights (weight>0 only)."""
    weights = weights or IK_MARKER_WEIGHTS
    task_set = ET.Element("IKTaskSet", {"name": "gait_analysis_markers"})
    objects = ET.SubElement(task_set, "objects")
    for marker, weight in weights.items():
        if weight <= 0:
            continue
        task = ET.SubElement(objects, "IKMarkerTask", {"name": marker})
        ET.SubElement(task, "apply").text = "true"
        ET.SubElement(task, "weight").text = f"{weight:g}"
    ET.SubElement(task_set, "groups")
    return task_set


def write_ik_task_set_xml(path: str | Path, weights: dict[str, float] | None = None) -> Path:
    """Write a standalone <IKTaskSet> document (reusable by ScaleTool too)."""
    doc = ET.Element("OpenSimDocument", {"Version": OPENSIM_DOC_VERSION})
    doc.append(build_ik_task_set(weights))
    _indent(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(doc).write(path, encoding="unicode", xml_declaration=True)
    return path


def build_measurement_set(measurements: list[dict] | None = None) -> ET.Element:
    """Build a <MeasurementSet> for the ScaleTool's ModelScaler."""
    measurements = measurements or SCALE_MEASUREMENTS
    mset = ET.Element("MeasurementSet")
    objects = ET.SubElement(mset, "objects")
    for m in measurements:
        meas = ET.SubElement(objects, "Measurement", {"name": m["name"]})
        pair_set = ET.SubElement(meas, "MarkerPairSet")
        pobjs = ET.SubElement(pair_set, "objects")
        for m1, m2 in m["pairs"]:
            mp = ET.SubElement(pobjs, "MarkerPair")
            ET.SubElement(mp, "markers").text = f"{m1} {m2}"
        body_set = ET.SubElement(meas, "BodyScaleSet")
        bobjs = ET.SubElement(body_set, "objects")
        for body, axes in m["bodies"]:
            bs = ET.SubElement(bobjs, "BodyScale", {"name": body})
            ET.SubElement(bs, "axes").text = axes
        ET.SubElement(meas, "apply").text = "true"
    ET.SubElement(mset, "groups")
    return mset


def write_scale_setup_xml(
    path: str | Path,
    model_file: str,
    static_trc: str,
    output_model_file: str,
    mass_kg: float = 70.0,
    height_m: float = 1.75,
    time_range: tuple[float, float] = (0.0, 1.0),
    weights: dict[str, float] | None = None,
    measurements: list[dict] | None = None,
) -> Path:
    """Write an OpenSim ScaleTool setup: measurement-based ModelScaler + MarkerPlacer.

    `static_trc` is a short standing/neutral-pose capture used to size the model and
    place markers. Produces a subject-scaled .osim for IK.
    """
    doc = ET.Element("OpenSimDocument", {"Version": OPENSIM_DOC_VERSION})
    tool = ET.SubElement(doc, "ScaleTool", {"name": "subject"})
    ET.SubElement(tool, "mass").text = f"{mass_kg:g}"
    ET.SubElement(tool, "height").text = f"{height_m * 1000:g}"  # mm
    ET.SubElement(tool, "age").text = "-1"

    gmm = ET.SubElement(tool, "GenericModelMaker")
    ET.SubElement(gmm, "model_file").text = str(model_file)
    ET.SubElement(gmm, "marker_set_file").text = "Unassigned"

    scaler = ET.SubElement(tool, "ModelScaler")
    ET.SubElement(scaler, "apply").text = "true"
    ET.SubElement(scaler, "scaling_order").text = "measurements"
    scaler.append(build_measurement_set(measurements))
    ET.SubElement(scaler, "marker_file").text = _rel_to(path, static_trc)
    ET.SubElement(scaler, "time_range").text = f"{time_range[0]:g} {time_range[1]:g}"
    ET.SubElement(scaler, "preserve_mass_distribution").text = "true"
    ET.SubElement(scaler, "output_model_file").text = _rel_to(path, output_model_file)

    placer = ET.SubElement(tool, "MarkerPlacer")
    ET.SubElement(placer, "apply").text = "true"
    placer.append(build_ik_task_set(weights))
    ET.SubElement(placer, "marker_file").text = _rel_to(path, static_trc)
    ET.SubElement(placer, "time_range").text = f"{time_range[0]:g} {time_range[1]:g}"
    ET.SubElement(placer, "output_model_file").text = _rel_to(path, output_model_file)
    ET.SubElement(placer, "output_motion_file").text = ""

    _indent(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(doc).write(path, encoding="unicode", xml_declaration=True)
    return path


def write_ik_tool_setup_xml(
    path: str | Path,
    model_file: str,
    marker_file: str,
    output_motion_file: str,
    time_range: tuple[float, float] | None = None,
    weights: dict[str, float] | None = None,
    report_errors: bool = True,
) -> Path:
    """Write a full InverseKinematicsTool setup the OpenSim IK tool can run."""
    doc = ET.Element("OpenSimDocument", {"Version": OPENSIM_DOC_VERSION})
    tool = ET.SubElement(doc, "InverseKinematicsTool", {"name": "gait_ik"})
    ET.SubElement(tool, "model_file").text = str(model_file)
    ET.SubElement(tool, "report_errors").text = "true" if report_errors else "false"
    tool.append(build_ik_task_set(weights))
    ET.SubElement(tool, "marker_file").text = str(marker_file)
    ET.SubElement(tool, "coordinate_file").text = ""
    if time_range is not None:
        ET.SubElement(tool, "time_range").text = f"{time_range[0]:g} {time_range[1]:g}"
    ET.SubElement(tool, "output_motion_file").text = str(output_motion_file)
    _indent(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(doc).write(path, encoding="unicode", xml_declaration=True)
    return path
