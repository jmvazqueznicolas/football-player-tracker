# GCP setup guide (from zero)

End-to-end setup for the cloud pieces of this project: a GCS bucket for DVC + MLflow artifacts, a service account for CI, and a Compute Engine VM with a GPU for training. Estimated total time: 30–45 minutes the first time.

> Cost expectation for the whole project: ~10–25 USD. Google gives 300 USD in free credits when you first sign up, more than enough.

## 1. Create a Google Cloud account

1. Go to [console.cloud.google.com](https://console.cloud.google.com).
2. Sign in with your Google account (or create one).
3. Accept the Terms of Service. You'll be prompted to start the **free trial**: 300 USD in credits valid for 90 days.
4. Add a credit/debit card (required for verification — you won't be charged unless you exceed the free credits and explicitly upgrade to a paid account).

## 2. Create a project

In the top-left project picker, click **"New Project"**:

- **Project name:** `football-player-tracker`
- **Project ID:** `football-tracker-<your-suffix>` (must be globally unique; the console will suggest one)
- Organization: leave as is.

Click **Create**. Wait ~30s and select the project from the top-left dropdown.

> Save the **Project ID** somewhere; you'll use it constantly. Add it to your `.env` as `GCP_PROJECT_ID`.

## 3. Enable required APIs

Open Cloud Shell (the terminal icon top-right of the console) — it has `gcloud` pre-installed and authenticated.

```bash
gcloud config set project YOUR_PROJECT_ID

gcloud services enable \
  compute.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com
```

Wait for each to finish; the first time it may take 1–2 minutes.

## 4. Install `gcloud` CLI locally (macOS)

```bash
# Homebrew
brew install --cask google-cloud-sdk

# Verify
gcloud --version

# Authenticate
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

## 5. Request GPU quota

By default new GCP accounts have **zero GPU quota** in most regions. Request it before you need to train.

1. Console → **IAM & Admin → Quotas & System Limits**.
2. Filter by `GPUS_ALL_REGIONS` (global) — request increase to **1**.
3. Also filter by `NVIDIA_T4_GPUS` in region `us-central1` — request **1**.
4. Submit. Approval usually takes a few hours to 1 business day.

While you wait, you can do everything else (dataset setup, code, MLflow local, etc.).

## 6. Create a GCS bucket for artifacts

Bucket names are globally unique.

```bash
export BUCKET=football-tracker-artifacts-$(whoami)
export REGION=us-central1

gcloud storage buckets create gs://$BUCKET \
  --location=$REGION \
  --uniform-bucket-level-access
```

We'll keep everything in this single bucket under prefixes:

- `gs://$BUCKET/dvc/`     — DVC-tracked datasets and models
- `gs://$BUCKET/mlflow/`  — MLflow artifacts (model files, plots)
- `gs://$BUCKET/data/`    — raw data backups

Add the bucket name to your `.env` as `GCS_BUCKET=football-tracker-artifacts-<yourname>`.

## 7. Create a service account for the project

This is the identity GitHub Actions and your local code will use to push to GCS.

```bash
export SA_NAME=football-tracker-sa
export SA_EMAIL=$SA_NAME@$GCP_PROJECT_ID.iam.gserviceaccount.com

gcloud iam service-accounts create $SA_NAME \
  --display-name="Football Tracker service account"

# Grant access to the bucket only (least privilege)
gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.objectAdmin"

# For Compute Engine training jobs
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/compute.instanceAdmin.v1"

# Download a JSON key (used locally and as a GitHub Action secret)
gcloud iam service-accounts keys create gcp-credentials.json \
  --iam-account=$SA_EMAIL
```

> **Important:** `gcp-credentials.json` is already in `.gitignore`. Never commit it. Add the file content as a GitHub Action secret named `GCP_SA_KEY` later, for the training workflow.

## 8. Configure DVC remote

From the repo root, with Poetry env active:

```bash
poetry run dvc init
poetry run dvc remote add -d gcs gs://$BUCKET/dvc
poetry run dvc remote modify gcs credentialpath gcp-credentials.json
git add .dvc .dvcignore
git commit -m "chore: initialize DVC with GCS remote"
```

## 9. (Optional) Configure MLflow with GCS artifact backend

For Session 2 you can keep MLflow purely local with `MLFLOW_TRACKING_URI=file:./mlruns`. To persist artifacts to GCS so they survive VM restarts, set in `.env`:

```
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_ARTIFACT_LOCATION=gs://your-bucket/mlflow
```

## 10. Launch a training VM (Session 1 or 2 — when ready to train)

```bash
export ZONE=us-central1-a
export VM_NAME=football-trainer

gcloud compute instances create $VM_NAME \
  --zone=$ZONE \
  --machine-type=n1-standard-8 \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --preemptible \
  --service-account=$SA_EMAIL \
  --scopes=cloud-platform

# SSH in
gcloud compute ssh $VM_NAME --zone=$ZONE
```

The Deep Learning VM image comes with CUDA, cuDNN, conda and PyTorch pre-installed. Once inside:

```bash
git clone https://github.com/USERNAME/football-player-tracker.git
cd football-player-tracker
pip install poetry==1.8.3
poetry install --with dev
poetry run dvc pull
poetry run python -m football_tracker.training.train
```

**When done, STOP THE VM** (you only pay for storage when stopped):

```bash
gcloud compute instances stop $VM_NAME --zone=$ZONE
```

Or delete it entirely if you have everything saved to GCS:

```bash
gcloud compute instances delete $VM_NAME --zone=$ZONE
```

## 11. Cost guardrails

- **Set a budget alert.** Console → Billing → Budgets & alerts → Create budget. Set 50 USD with alerts at 50 / 90 / 100 percent.
- **Use preemptible / spot instances** (`--preemptible`) — up to 80% cheaper, but can be killed with 30s notice. Fine for our trainings since checkpoints are short.
- **Always stop the VM when idle.** T4 idle costs ~7 USD/day.
- **Free GCS egress:** keep your training data inside the same region as your bucket.

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ResourceExhausted: insufficient regional quota` | GPU quota not granted | Wait for quota approval; try `us-central1-c` if `-a` is full |
| `Permission denied on bucket` | Service account missing role | Re-run the `add-iam-policy-binding` step |
| `dvc pull` fails with auth error | `GOOGLE_APPLICATION_CREDENTIALS` not set | `export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/gcp-credentials.json` |
| Training freezes on first batch | DataLoader workers > CPU cores | Lower `workers` in `configs/training/default.yaml` |
