"""Generate OpenSim setup XML (IK Task Set + IK Tool) from our marker weights.

Hand-writing OpenSim XML is error-prone, so we generate it from markerset.py. The
output is plain OpenSim 4.x XML that the InverseKinematicsTool reads directly -- no
OpenSim install needed to *write* it (only to *run* it), which keeps generation
testable offline.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .markerset import IK_MARKER_WEIGHTS

OPENSIM_DOC_VERSION = "40000"  # OpenSim 4.x document version


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
