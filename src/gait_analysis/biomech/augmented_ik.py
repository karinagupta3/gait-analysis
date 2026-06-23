"""Scale + inverse kinematics on AUGMENTED anatomical markers (the OpenCap path).

Given a .trc of the 43 "_study" anatomical markers (from marker_augmentation),
size the LaiUhlrich2022 OpenSim model to the subject and solve IK -- reusing the
exact bundled OpenCap/Pose2Sim setups (Markers_LSTM.xml, Scaling_Setup_*_LSTM.xml,
IK_Setup_*_LSTM.xml) and their anatomically-weighted IK tasks. This replaces the
old hand-placed markerset, which is why the trunk no longer renders hunched.

perform_scaling tolerates non-static clips (it trims fast/crouched/outlier frames
and scales each segment from the trimmed mean), so we don't need a separate static
calibration trial -- the motion clip itself drives scaling.
"""

from __future__ import annotations

from pathlib import Path


# Frontal/transverse DOFs a single side-view camera CANNOT measure. MediaPipe's
# depth axis is too noisy, so leaving these free makes IK swing them wildly or pin
# them at their limits (we saw hip_rotation stuck at -70deg, pelvis listing +/-40deg) --
# which renders as a twisted skeleton that doesn't match the upright person on video.
# We lock them to neutral so IK solves a clean SAGITTAL-plane skeleton (the plane a
# side view actually measures). This is the honest single-camera output; frontal/
# transverse angles need a second camera (the OpenCap setup).
_LOCK_SUBSTR = ("_list", "rotation", "adduction", "bending", "_add", "_rot",
                "subtalar", "pro_sup", "Flex_Ext")   # Flex_Ext => lumbar segments only


def _lock_nonsagittal(opensim, model_path: Path) -> None:
    """Lock non-sagittal coordinates to neutral in the scaled model, in place."""
    model = opensim.Model(str(model_path))
    cs = model.getCoordinateSet()
    locked = []
    for i in range(cs.getSize()):
        co = cs.get(i)
        nm = co.getName()
        if any(s in nm for s in _LOCK_SUBSTR):
            co.setDefaultValue(0.0)
            co.setDefaultLocked(True)
            locked.append(nm)
    model.printToXML(str(model_path))
    print(f"[augmented_ik] locked {len(locked)} non-sagittal DOFs: {locked}")


def scale_and_ik(trc_path: str | Path, out_dir: str | Path,
                 height_m: float, mass_kg: float,
                 use_simple_model: bool = False,
                 lock_nonsagittal: bool = False) -> tuple[Path, Path]:
    """Scale the bundled model to the subject and run IK on the augmented .trc.

    Returns (mot_path, scaled_model_path). Both land in `out_dir`, named after the
    .trc stem (Pose2Sim's convention): <stem>.mot and <stem>.osim.
    """
    trc_path, out_dir = Path(trc_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import opensim
    from Pose2Sim import kinematics as K

    osim_setup = K.get_opensim_setup_dir()
    # The model's bone meshes live in OpenSim_Setup/Geometry; register them so
    # ScaleTool can load the model.
    opensim.ModelVisualizer.addDirToGeometrySearchPaths(str(osim_setup / "Geometry"))

    K.perform_scaling(
        trc_path, "LSTM", out_dir, osim_setup,
        use_simple_model=use_simple_model,
        subject_height=height_m, subject_mass=mass_kg,
        remove_scaling_setup=True,
    )
    scaled_model = out_dir / (trc_path.stem + ".osim")
    if not scaled_model.exists():
        raise RuntimeError(f"Scaling produced no model at {scaled_model}")

    if lock_nonsagittal:
        _lock_nonsagittal(opensim, scaled_model)

    K.perform_IK(trc_path, out_dir, osim_setup, "LSTM", remove_IK_setup=True)
    mot = out_dir / (trc_path.stem + ".mot")
    if not mot.exists():
        raise RuntimeError(f"IK produced no motion at {mot}")
    return mot, scaled_model
