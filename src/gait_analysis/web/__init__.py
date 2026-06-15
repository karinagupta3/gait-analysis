"""Web app for the gait-analysis system (FastAPI).

Local-first and containerizable. Serves the report UI; the heavy video->.mot
processing (Pose2Sim + OpenSim) runs separately where OpenSim is installed.
"""
