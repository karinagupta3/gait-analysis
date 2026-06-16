#!/usr/bin/env bash
# One-command Azure deploy for the gait-analysis web tier (Container Apps).
#
# Why a script: the agent sandbox has no Azure CLI/credentials, so it can't run `az`.
# You run this once from the repo root, logged into Azure. It is idempotent -- re-run it
# to ship new code (it rebuilds the image and rolls the app).
#
#   az login                     # (once)
#   bash setup/deploy_azure.sh   # creates resources + builds + deploys, prints the URL
#
# Override any name via env, e.g.:  RG=my-rg LOC=westus APP=gait bash setup/deploy_azure.sh
set -euo pipefail

RG="${RG:-gait-rg}"
LOC="${LOC:-eastus}"
ACR="${ACR:-gaitacr$RANDOM}"        # must be globally unique; default randomizes
ENVN="${ENVN:-gait-env}"
APP="${APP:-gait-web}"

command -v az >/dev/null || { echo "Azure CLI not found. Install: https://aka.ms/azure-cli"; exit 1; }
az account show >/dev/null 2>&1 || { echo "Not logged in. Run: az login"; exit 1; }

echo ">> RG=$RG  LOC=$LOC  ACR=$ACR  ENV=$ENVN  APP=$APP"
az extension add --name containerapp --upgrade --yes >/dev/null 2>&1 || true
az provider register -n Microsoft.ContainerRegistry --wait >/dev/null 2>&1 || true
az provider register -n Microsoft.App --wait >/dev/null 2>&1 || true
az provider register -n Microsoft.OperationalInsights --wait >/dev/null 2>&1 || true

echo ">> [1/4] resource group + container registry"
az group create -n "$RG" -l "$LOC" -o none
az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled true -o none

echo ">> [2/4] build image in ACR (uses the repo Dockerfile)"
az acr build -r "$ACR" -t gait-web:latest .

echo ">> [3/4] Container Apps environment"
# A prior failed run can leave the env half-provisioned; recreate it if so.
state="$(az containerapp env show -n "$ENVN" -g "$RG" --query properties.provisioningState -o tsv 2>/dev/null || echo '')"
if [ "$state" = "Failed" ] || [ "$state" = "Canceled" ]; then
  echo ">> existing environment is '$state'; deleting and recreating ..."
  az containerapp env delete -n "$ENVN" -g "$RG" --yes -o none 2>/dev/null || true
fi
az containerapp env create -n "$ENVN" -g "$RG" -l "$LOC" -o none 2>/dev/null || true

# Provisioning is async -- wait until the environment is actually ready before creating the app.
echo ">> waiting for the environment to finish provisioning (can take a few minutes) ..."
for i in $(seq 1 60); do
  state="$(az containerapp env show -n "$ENVN" -g "$RG" --query properties.provisioningState -o tsv 2>/dev/null || echo '')"
  echo "   env state: ${state:-<creating>}"
  case "$state" in
    Succeeded) break ;;
    Failed|Canceled) echo "Environment provisioning $state. Re-run the script."; exit 1 ;;
  esac
  sleep 10
done
[ "$state" = "Succeeded" ] || { echo "Timed out waiting for the environment. Re-run the script."; exit 1; }

echo ">> [4/4] deploy / update the app"
if az containerapp show -n "$APP" -g "$RG" >/dev/null 2>&1; then
  az containerapp update -n "$APP" -g "$RG" --image "$ACR.azurecr.io/gait-web:latest" -o none
else
  az containerapp create -n "$APP" -g "$RG" --environment "$ENVN" \
    --image "$ACR.azurecr.io/gait-web:latest" \
    --registry-server "$ACR.azurecr.io" \
    --target-port 8000 --ingress external \
    --min-replicas 1 --max-replicas 2 \
    --env-vars GAIT_STORE_DIR=/tmp/gait-store MPLCONFIGDIR=/tmp/mpl -o none
fi

FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo ""
echo ">> Deployed:  https://$FQDN"
echo ">> Note: upload .mot(+.trc) -> report. Video->.mot processing stays where OpenSim is installed."
