"""Markerless gait analysis toolkit.

Phase 1 scaffold. Pipeline (see docs/01-clinical-landscape-and-build-plan.md):

    iPhone video
      -> pose.rtmpose_runner      (RTMPose 2D keypoints, COCO-17)
      -> [quick mode]  monocular 3D (SMPL) -> OpenSim IK        (TODO Phase 1)
         [accurate mode] Pose2Sim triangulation -> OpenSim IK   (TODO Phase 1)
      -> analysis.spatiotemporal   (cadence, step/stride time, symmetry)
      -> report                    (TODO Phase 2)

Everything here uses commercial-license-safe components only.
"""

__version__ = "0.1.0"
