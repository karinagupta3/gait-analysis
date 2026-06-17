#!/usr/bin/env bash
# Deploy the TIER-2 OpenSim worker to Azure as an event-driven Container Apps Job.
#
# WHAT THIS CREATES
#   - a Storage account with:
#       * blob containers  gait-in  (uploaded videos) and  gait-out  (.mot/.trc/report)
#       * a Storage Queue  gait-jobs  (one JSON message per job)
#   - the worker image in ACR (built from Dockerfile.worker)
#   - a Container Apps JOB "gait-worker" that scales 0 -> N off the gait-jobs queue
#     length, runs 2 vCPU / 4Gi per replica, and wires GAIT_STORAGE_CONNECTION from
#     the storage account key.
#
# WHY A SCRIPT: the agent sandbox has no Azure CLI/credentials, so it cannot run `az`.
# >>> THE OPERATOR RUNS THIS, logged into Azure, from the repo root: <<<
#
#     az login                            # (once)
#     bash setup/deploy_worker_azure.sh   # creates resources + builds + deploys
#
# It is idempotent — re-run it to ship new worker code (rebuilds the image, updates job).
# Override any name via env, e.g.:  STORAGE=mygaitsa bash setup/deploy_worker_azure.sh
set -euo pipefail

# --- configurable names (existing project defaults) --------------------------
RG="${RG:-gait-rg}"                 # resource group (shared with tier-1)
LOC="${LOC:-eastus2}"               # region
ACR="${ACR:-gaitacr494}"            # container registry (shared with tier-1)
ENVN="${ENVN:-gait-env}"            # Container Apps environment (shared with tier-1)
STORAGE="${STORAGE:-gaitstore$RANDOM}"  # storage acct name (3-24 lowercase, globally unique)

JOB="${JOB:-gait-worker}"
IMAGE="gait-worker:latest"
QUEUE="gait-jobs"
IN_CONTAINER="gait-in"
OUT_CONTAINER="gait-out"

command -v az >/dev/null || { echo "Azure CLI not found. Install: https://aka.ms/azure-cli"; exit 1; }
az account show >/dev/null 2>&1 || { echo "Not logged in. Run: az login"; exit 1; }

echo ">> RG=$RG  LOC=$LOC  ACR=$ACR  ENV=$ENVN  STORAGE=$STORAGE  JOB=$JOB"
az extension add --name containerapp --upgrade --yes >/dev/null 2>&1 || true
az provider register -n Microsoft.App --wait >/dev/null 2>&1 || true
az provider register -n Microsoft.ContainerRegistry --wait >/dev/null 2>&1 || true

echo ">> [1/5] resource group (idempotent)"
az group create -n "$RG" -l "$LOC" -o none

echo ">> [2/5] storage account + containers + queue"
az storage account create -n "$STORAGE" -g "$RG" -l "$LOC" \
  --sku Standard_LRS --kind StorageV2 -o none
# Grab a connection string we can both (a) use for the data-plane creates below
# and (b) inject into the worker as GAIT_STORAGE_CONNECTION.
CONN="$(az storage account show-connection-string -n "$STORAGE" -g "$RG" -o tsv)"
az storage container create -n "$IN_CONTAINER"  --connection-string "$CONN" -o none
az storage container create -n "$OUT_CONTAINER" --connection-string "$CONN" -o none
az storage queue create     -n "$QUEUE"         --connection-string "$CONN" -o none

echo ">> [3/5] build the worker image in ACR (uses Dockerfile.worker; ACR defaults to linux/amd64)"
# Skip with SKIP_BUILD=1 when the image is already current in ACR.
if [ "${SKIP_BUILD:-}" = "1" ]; then
  echo "   SKIP_BUILD=1 -> reusing existing $IMAGE in $ACR"
else
  az acr build -r "$ACR" -f Dockerfile.worker -t "$IMAGE" .
fi

echo ">> [4/5] ensure the Container Apps environment exists"
az containerapp env show -n "$ENVN" -g "$RG" >/dev/null 2>&1 \
  || az containerapp env create -n "$ENVN" -g "$RG" -l "$LOC" -o none

echo ">> [5/5] create/update the event-driven worker JOB"
# Storage Queue scaling needs the account key as a secret the scale rule references.
KEY="$(az storage account keys list -n "$STORAGE" -g "$RG" --query '[0].value' -o tsv)"
ACR_SERVER="$ACR.azurecr.io"
ACR_USER="$(az acr credential show -n "$ACR" --query username -o tsv)"
ACR_PASS="$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)"

# Common args for create/update. The scale rule:
#   - type azure-queue: scale on gait-jobs length
#   - queueLength=1: start a replica as soon as one job is waiting
#   - min 0 / max 5: scale to zero when idle (no cost), burst up to 5 in parallel
COMMON=(
  --image "$ACR_SERVER/$IMAGE"
  --trigger-type Event
  --replica-timeout 3600          # allow up to 1h per OpenSim run
  --replica-retry-limit 1         # retry a failed replica once
  --parallelism 5                 # up to 5 concurrent replicas
  --replica-completion-count 1    # one successful replica completes an execution
  --min-executions 0              # scale to zero when the queue is empty
  --max-executions 5
  --scale-rule-name queue-len
  --scale-rule-type azure-queue
  --scale-rule-metadata "queueName=$QUEUE" "queueLength=1" "accountName=$STORAGE"
  --scale-rule-auth "connection=queue-conn"
  --cpu 2.0 --memory 4Gi
  --secrets "queue-conn=$CONN" "storage-conn=$CONN" "acr-pass=$ACR_PASS"
  --registry-server "$ACR_SERVER" --registry-username "$ACR_USER" --registry-password "$ACR_PASS"
  --env-vars "GAIT_STORAGE_CONNECTION=secretref:storage-conn"
)

if az containerapp job show -n "$JOB" -g "$RG" >/dev/null 2>&1; then
  echo ">> updating existing job $JOB"
  az containerapp job update -n "$JOB" -g "$RG" "${COMMON[@]}" -o none
else
  echo ">> creating job $JOB"
  az containerapp job create -n "$JOB" -g "$RG" --environment "$ENVN" "${COMMON[@]}" -o none
fi

echo ""
echo ">> Tier-2 worker deployed."
echo ">> Storage account : $STORAGE   (containers: $IN_CONTAINER, $OUT_CONTAINER; queue: $QUEUE)"
echo ">> Worker job      : $JOB        (scales 0->5 on $QUEUE length, 2 vCPU / 4Gi)"
echo ""
echo ">> NEXT STEPS (tier-1 dispatch):"
echo "   1. Upload a video to:  $IN_CONTAINER/<session_id>/video.<ext>"
echo "   2. Enqueue a message on '$QUEUE':"
echo '        {"session_id":"<sid>","mode":"quick","ext":"mp4","speed":null}'
echo "   3. The job spins up, writes $OUT_CONTAINER/<session_id>/coordinates.mot (+ .trc, report.html)"
echo "      and status.json. Tier-1 polls status.json and renders the .mot."
echo ""
echo ">> Give tier-1 the SAME GAIT_STORAGE_CONNECTION so it can upload + enqueue:"
echo "   az storage account show-connection-string -n $STORAGE -g $RG -o tsv"
