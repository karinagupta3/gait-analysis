# Setup & capture guide

For clinic staff / users running a session. Two parts: **get set up once**, then **record well**.

## Part 1 — Setup (once)

### Just viewing a `.mot` report
No OpenSim needed. `pip install -e ".[web]"`, run `gait-web`, open the page, upload the `.mot`.

### Processing a video into joint angles (needs OpenSim)
```
conda create -n gait python=3.11
conda activate gait
conda install -c opensim-org opensim
pip install -e ".[web]" mediapipe
```
Then build + point the app at a marked model (Track B — see `docs/05`):
```
gait-inspect-model --model LaiUhlrich2022.osim          # confirm body/joint names
gait-build-model --base LaiUhlrich2022.osim --out LaiUhlrich2022_ga.osim
export GAIT_OSIM_MODEL=$PWD/LaiUhlrich2022_ga.osim
gait-web                                                 # the /process page banner turns green
```
Validate before trusting it: `gait-validate --ref pose2sim.mot --test quick.mot` (sagittal RMSE ≤ ~5°).

The web app's **/setup** page shows live status (OpenSim / MediaPipe / model) so you know what's missing.

## Part 2 — How to record

Kinematic quality is set at capture time. Same principles as OpenCap-style markerless capture.

### Monocular (1 phone) — quick mode
| What | Guidance |
|---|---|
| Phone position | Tripod, ~3–4 m away, lens at **hip height**, **landscape** |
| View | Walking → **side/sagittal**; squats / sit-to-stand → **3/4 front** (both knees visible) |
| Framing | **Whole body in frame** the entire movement; no pan/zoom |
| Settings | 60 fps if available; even lighting; plain background; fitted clothing; feet/shoes visible |
| Content | Walking: 4–6 strides · Squats: 3–5 reps · Sit-to-stand: 5 rises (for 5×STS) |

### Two phones — accurate mode (Pose2Sim)
- Two phones ~**60° apart**, both seeing the whole body; start both **before** the subject moves.
- Film a **checkerboard calibration** visible to both cameras first — required for metric (real-world) scale.
- 2-phone runs on the CLI today (`gait-pipeline`); in-app multi-video upload is roadmap item **A3**.

### Why these rules (the short version)
- Side view maximizes the **sagittal** plane, where markerless is most accurate (hip/knee/ankle flexion).
- Full-body framing lets the pose model and OpenSim IK see every segment every frame (fewer dropouts).
- Calibration is what converts pixels into **metres** — without it, step length and other spatial measures aren't valid.

> Clinical interpretation of what we measure (per condition, with references) lives in `docs/11`.
