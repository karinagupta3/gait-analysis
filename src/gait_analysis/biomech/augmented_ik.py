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


def scale_and_ik(trc_path: str | Path, out_dir: str | Path,
                 height_m: float, mass_kg: float,
                 use_simple_model: bool = False) -> tuple[Path, Path]:
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

    K.perform_IK(trc_path, out_dir, osim_setup, "LSTM", remove_IK_setup=True)
    mot = out_dir / (trc_path.stem + ".mot")
    if not mot.exists():
        raise RuntimeError(f"IK produced no motion at {mot}")
    return mot, scaled_model
