# Deploying on Azure

**Why Azure (vs Vercel):** Vercel/serverless can host only the lightweight web tier; it
**cannot** run OpenSim/Pose2Sim (large native libs, optional GPU). Azure can host **both**
the web app *and* the heavy processing, so the whole pipeline lives in one cloud. Azure
also offers a **HIPAA BAA** if you ever move beyond research — a plus for a clinical pivot.

> Tooling note: this session has Vercel/Supabase tooling but not Azure, so the steps below
> are the `az` CLI commands you run. The app is already containerized (`Dockerfile`).

## Architecture on Azure
```
                ┌── Azure Container Apps ──────────────┐
  Clinician ───▶│  web app (our Dockerfile)            │  <- deployable now
                │  upload .mot, view report, sessions  │
                └──────────────┬───────────────────────┘
                               │ reads/writes
                ┌──────────────▼───────────────────────┐
   Blob Storage │  videos / .trc / .mot / report.html  │
                └──────────────┬───────────────────────┘
                               │ triggers
                ┌──────────────▼───────────────────────┐
   processing   │  Container Apps Job OR GPU VM         │  <- Pose2Sim + OpenSim
   (heavy)      │  video -> RTMPose -> OpenSim -> .mot  │     (conda image, optional GPU)
                └──────────────────────────────────────┘
   Sessions DB: Azure Database for PostgreSQL (replaces the on-disk file store)
```

## Phase 1 — web tier (deployable today)
The web app (report viewer + session store) runs on **Azure Container Apps**.

```bash
# 0. Prereqs: az login; pick names
RG=gait-rg; LOC=eastus; ACR=gaitacr$RANDOM; APP=gait-web; ENVNAME=gait-env

az group create -n $RG -l $LOC
az acr create -n $ACR -g $RG --sku Basic --admin-enabled true

# 1. Build the image in the cloud (uses our Dockerfile) and push to ACR
az acr build -r $ACR -t gait-web:latest .

# 2. Container Apps environment + the web app
az containerapp env create -n $ENVNAME -g $RG -l $LOC
az containerapp create -n $APP -g $RG --environment $ENVNAME \
  --image $ACR.azurecr.io/gait-web:latest \
  --registry-server $ACR.azurecr.io \
  --target-port 8000 --ingress external \
  --min-replicas 1 --max-replicas 2
az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv
```
Open the printed FQDN → the app. Redeploy on changes with `az acr build ...` then
`az containerapp update -n $APP -g $RG --image $ACR.azurecr.io/gait-web:latest`.

## Phase 2 — persistence (Blob + Postgres)
Move the on-disk session store to managed services so it survives restarts and scales:
- **Azure Blob Storage** for `.mot`/reports/videos (`az storage account create ...`).
- **Azure Database for PostgreSQL Flexible Server** for subjects/sessions/trials
  (`az postgres flexible-server create ...`), wired via `DATABASE_URL`.
The app's `web/app.py` store is intentionally small/swappable for this.

## Phase 3 — processing engine (the heavy part)
Run Pose2Sim + OpenSim as a **separate** unit so the web tier stays light:
- **Container Apps Job** (CPU) or an **Azure GPU VM** (NC-series) for RTMPose, with a
  conda image that has OpenSim (mirror the `gait` env). Triggered when a video lands in
  Blob; writes the `.mot` back; the web app renders it.
- Keep this image separate from the web image (OpenSim is large; the web tier shouldn't carry it).

## CI/CD
`.github/workflows/azure-deploy.yml` (template) builds the image to ACR and updates the
Container App on push — add `AZURE_CREDENTIALS`, `ACR_NAME`, `RG`, `APP` as repo secrets.

## Cost / credits
- Web tier on Container Apps is cheap (scale-to-low). Penn (`@seas.upenn.edu`) likely
  qualifies for **Azure for Students** credits.
- The **GPU processing box is the cost driver** — run it **on-demand** (start per batch,
  stop after) rather than 24/7.

## Data note
Research only, no real patients → no PHI/HIPAA obligations now. If you later handle patient
data, sign an Azure **BAA**, enable private networking, and don't put identifiers in Blob/DB
without encryption + access controls.
