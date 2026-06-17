# Tier-2 — the OpenSim cloud worker

## What tier-2 is
Tier-1 is the slim FastAPI web app (uploads, session storage, report viewing). It
runs on a small pip-only image and **cannot run OpenSim** — OpenSim ships only as a
conda package and is large. Tier-2 is a **separate, heavier conda image** whose only
job is the number crunching: turn a video into an OpenSim `.mot` for the 3D modes.

The two tiers never share a process. They communicate only through Azure Blob
Storage and a Storage Queue.

## The `.mot` contract
The OpenSim `coordinates.mot` is the canonical hand-off between the tiers. Tier-2
produces it; tier-1 consumes it. Tier-1 can render the full kinematics report and
clinical signature flags from a `.mot` alone (`pipeline.report_from_mot()`), with no
OpenSim installed. The `.trc` (markers) and `report.html` are auxiliary; the `.mot`
is the real product, and a job with no `coordinates.mot` is treated as failed.

## Blob + queue layout
```
Storage Queue : gait-jobs
Input  blob   : gait-in/<session_id>/video.<ext>
Output blobs  : gait-out/<session_id>/coordinates.mot   <- the contract
                gait-out/<session_id>/markers.trc
                gait-out/<session_id>/report.html
                gait-out/<session_id>/status.json       <- processing|done|error
```
Queue message (JSON):
```json
{"session_id": "abc123", "mode": "quick", "ext": "mp4", "speed": null}
```
- `mode: "quick"`    → 1-phone 3D, `pipeline.run_quick(video, GAIT_OSIM_MODEL, outdir)`
- `mode: "accurate"` → 2-phone, `pipeline.run_accurate(project_dir)`
- `speed` is an optional float (m/s) used as gait-speed context for the signature flags.

## How to deploy
The agent sandbox has no Azure CLI/credentials, so deployment is a script the
**operator runs** after `az login`, from the repo root:
```bash
az login                            # once
bash setup/deploy_worker_azure.sh   # creates storage + containers + queue, builds, deploys
```
It creates the Storage account (`gait-in`/`gait-out` containers + `gait-jobs` queue),
builds `Dockerfile.worker` in ACR (`linux/amd64`), and creates an event-driven
**Container Apps Job** `gait-worker` (2 vCPU / 4Gi) that scales `0 → N` on the queue
length and gets `GAIT_STORAGE_CONNECTION` wired from the storage account key.

The marked OpenSim model `LaiUhlrich2022_ga.osim` is baked into the image at
`/app/models/` (built from the base model via the `gait-build-model` console script,
or used directly if already committed) and exposed as `GAIT_OSIM_MODEL`.

## How tier-1 dispatches a job
1. Upload the video to `gait-in/<session_id>/video.<ext>`.
2. Enqueue a JSON message on `gait-jobs` (see above).
3. The scale rule starts a worker replica; it writes `status.json` (`processing`),
   runs the pipeline, uploads `coordinates.mot` (+ `.trc`, `report.html`), writes
   `status.json` (`done`), and deletes the queue message.
4. Tier-1 polls `gait-out/<session_id>/status.json` and, on `done`, renders the `.mot`.

Tier-1 needs the **same** `GAIT_STORAGE_CONNECTION` to upload and enqueue:
```bash
az storage account show-connection-string -n <STORAGE> -g gait-rg -o tsv
```

## KNOWN CAVEAT — validate 3D angles before clinical trust
The marker placement in `src/gait_analysis/biomech/build_marked_model.py` is still
**provisional**: the named markers injected into `LaiUhlrich2022_ga.osim` are placed
per `marker_placement.PLACEMENTS`, which has not been validated against a gold-standard
(Track A / Pose2Sim). Until that placement is validated, the resulting 3D joint angles
from tier-2 are **provisional and must not be clinically trusted**. Treat tier-2 output
as engineering validation only, not patient-facing measurement.
