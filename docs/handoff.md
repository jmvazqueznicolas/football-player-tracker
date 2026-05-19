# Project Handoff — Football Player Tracker

**Status as of:** Session 1 nearly complete. Sanity training validated end-to-end.
**Next immediate step:** Configure ngrok to expose local MLflow to Colab (Task #19).

---

## 1. User profile and underlying goal

**User:** Jesús Manuel Vázquez Nicolás (jmvazqueznicolas@gmail.com), based in Mexico City.

**Background:**
- 5 years of professional experience in Computer Vision / Deep Learning, mostly with TensorFlow Object Detection API and YOLO/Ultralytics.
- Did a Master's thesis in CV ~10 years ago.
- Has worked at companies in R&D doing object detection, including AutoML on Vertex AI.
- Self-described as having done "artisanal" deployments: trained models exported to .pb/.h5 and run from local scripts, never via proper endpoints or pipelines.

**Skills gaps identified vs. target role:**
- MLflow, Weights & Biases — knew of them, hadn't used
- DVC for dataset/model versioning — not used
- Automated training pipelines (Kubeflow, Airflow, GitHub Actions for ML) — theoretical only
- Optuna / bayesian HPO — not used
- FiftyOne for error analysis — not used
- Docker for ML serving (vs general software) — not used
- Drift monitoring (Evidently, Arize) — not used

**Strong areas:**
- Python, Git, virtual environments, Docker basics
- PyTorch and TensorFlow
- YOLO / Ultralytics
- COCO / YOLO annotation formats
- Label Studio for annotation
- GPU training (local + Colab)
- Basic metrics (mAP, IoU, precision, recall, confusion matrix)

**Target role:**
- ML / MLOps Engineer position in CDMX (Polanco) or Buenos Aires (Palermo).
- Hybrid initially, 3 days presential.
- Listed requirements: 5+ yr CV/DL (EXCLUYENTE), pipelines de MLOps (EXCLUYENTE), Python, PyTorch/TensorFlow, YOLO/Ultralytics, MLflow/W&B, COCO/YOLO formats, GPU training, Git, CVAT/Label Studio/Roboflow, dataset/model versioning, reproducibility, automation.
- Diagnosis: he can defend 5+ years of CV experience well; the MLOps side is where he needs visible evidence. This project IS that evidence.

**Time budget for project:** ~40 hours before technical interviews.

**Working language:** Spanish. He prefers casual technical Spanish, "tú" form. He is intellectually curious and wants to learn the concepts, not just execute.

**Active-learning preference:** From this point forward, he prefers a Socratic mode where HE makes observations and asks clarifying questions, and the assistant validates/corrects/expands. He explicitly requested this. Honor it.

---

## 2. The project

**Name:** `football-player-tracker`
**Location on his machine:** `/Users/jmvazqueznicolas/Documents/Repositorios_Github/football-player-tracker`
**Cowork mount:** The folder IS mounted (Option C of the handoff workflow). All Read/Write/Edit go directly to his repo. Don't write to the temp `outputs/` directory.

**Bash workspace path mapping:** `/sessions/awesome-beautiful-feynman/mnt/football-player-tracker/`

**Domain:** detection and tracking of football players in match footage.

**Dataset:** Roboflow Universe `roboflow-jvuqo/football-players-detection-2frwp` v1 (CC BY 4.0).
- 4 classes (alphabetical order from YOLO annotation files): `['ball', 'goalkeeper', 'player', 'referee']`
- 1072 train / 38 valid / 13 test images
- All images are 640×640 (Roboflow already resized them)
- Severe class imbalance: 28.3× ratio between `player` (21,423 instances) and `goalkeeper` (756) in train
- **100%** of bounding boxes occupy less than 0.5% of image area (all tiny objects — significant for training strategy)
- Average ~24 annotations per image (denser than typical COCO)

---

## 3. Stack and architectural decisions

| Layer | Tool |
|-------|------|
| Language | Python 3.12 (3.13 was rejected due to ecosystem instability) |
| Dependency mgmt | Poetry 1.8.4+ (with `virtualenvs.in-project = true`) |
| Configuration | Hydra (configs in `configs/`) |
| Model framework | Ultralytics YOLOv11 (nano for baseline) |
| Tracking | MLflow **3.12.0+** (2.22.x had SIGSEGV on macOS arm64) |
| Tracking (parallel) | W&B planned for Session 2 |
| Dataset versioning | DVC + GCS bucket (planned — pending GCP) |
| Annotation source | Roboflow Universe (downloaded via `scripts/download_dataset.py`) |
| HPO | Optuna (planned, Session 3) |
| Error analysis | FiftyOne (planned, Session 3) |
| Serving | FastAPI + Streamlit (planned, Sessions 4-5) |
| Containerization | Docker multi-stage (CPU image scaffolded) |
| Demo deploy | Hugging Face Spaces (planned, Session 5) |
| CI/CD | GitHub Actions (CI scaffolded; train/deploy workflows are placeholders) |
| Cloud compute | GCP Compute Engine + T4 GPU (account created, GPU quota pending approval) |
| Remote tracking bridge | **ngrok** tunneling local MLflow to Colab (in setup) |
| Drift monitoring | Evidently AI (planned, Session 7) |

**Why these choices:**
- MLflow + W&B in parallel because the job description mentions both and demonstrating both is a portfolio differentiator.
- ngrok over DagsHub because the user wanted "more control" and to learn the tunnel pattern (it's a transferable skill for any remote training).
- GCP over AWS because the user has previously used Vertex AI; faster onboarding.
- Hugging Face Spaces for demo because Docker support is free with public URL.

---

## 4. Repository structure (what's currently in the repo)

```
football-player-tracker/
├── .github/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       ├── ci.yml                  ← functional CI (lint + test + Docker build)
│       ├── train.yml               ← placeholder for GCP training job
│       └── deploy.yml              ← placeholder for HF Spaces deploy
├── configs/                        ← Hydra configs
│   ├── config.yaml                 ← root (composes defaults)
│   ├── data/football.yaml          ← roboflow-jvuqo/football-players-detection-2frwp v1
│   ├── model/yolov11n.yaml         ← class names in correct order ['ball','goalkeeper','player','referee']
│   ├── model/yolov11s.yaml         ← same, larger variant
│   ├── training/default.yaml       ← all training hyperparams (epochs=50 default, device=null=auto)
│   ├── tracking/botsort.yaml       ← tracking config (used in Session 4)
│   └── tuning/optuna.yaml          ← HPO search space (used in Session 3)
├── data/processed/football-players/    ← downloaded dataset (DVC-tracked when GCP ready)
│   ├── data.yaml                   ← MODIFIED: path:. instead of ../, paths fixed
│   ├── train/{images,labels}/      ← 1072 each
│   ├── valid/{images,labels}/      ← 38 each
│   ├── test/{images,labels}/       ← 13 each
│   └── README.roboflow.txt
├── docker/
│   ├── Dockerfile                  ← multi-stage CPU
│   ├── Dockerfile.gpu              ← NVIDIA PyTorch image for training
├── docs/
│   ├── architecture.md             ← stack table, data flow diagram, reproducibility contract
│   └── gcp_setup.md                ← step-by-step GCP onboarding (account, bucket, SA, GPU quota)
├── notebooks/
│   └── 01_eda.ipynb                ← 9-section EDA, executed successfully, user has reviewed
├── scripts/
│   └── download_dataset.py         ← REWRITTEN to use argparse (Typer had conflicts with Click)
├── src/football_tracker/
│   ├── __init__.py
│   ├── cli.py                      ← typer CLI (`football-tracker info|version`)
│   ├── api/
│   │   ├── main.py                 ← FastAPI skeleton (/health, /)
│   │   └── streamlit_app.py        ← Streamlit skeleton
│   ├── data/
│   │   └── dataset.py              ← placeholder for Session 1
│   ├── inference/predictor.py      ← placeholder for Session 4
│   ├── models/loader.py            ← placeholder for Session 2
│   ├── tracking/tracker.py         ← placeholder for Session 4
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                ← IMPLEMENTED. Hydra + MLflow + Ultralytics callbacks
│   │   ├── tune.py                 ← placeholder for Session 3
│   │   ├── callbacks.py            ← IMPLEMENTED. Bridges Ultralytics→MLflow on_fit_epoch_end + on_train_end
│   │   └── utils.py                ← IMPLEMENTED. flatten_dict, cfg_to_flat_params, sanitize_metric_key
│   └── utils/
│       └── logging.py              ← loguru-based logging setup
├── tests/
│   ├── conftest.py                 ← fixtures: project_root, configs_dir, fixtures_dir
│   ├── test_smoke.py               ← imports
│   └── test_api.py                 ← FastAPI health endpoint
├── .env.example                    ← variables expected (ROBOFLOW_API_KEY, MLFLOW_TRACKING_URI, WANDB_*, etc.)
├── .gitignore                      ← ML-aware (excludes mlruns/, wandb/, data/raw, *.pt, *.onnx, etc.)
├── .pre-commit-config.yaml         ← black uses language_version: python3 (NOT python3.11)
├── .dockerignore
├── pyproject.toml                  ← Poetry config with all deps + ruff/black/mypy/pytest
├── Makefile                        ← setup, install-dev, lint, format, test, download-data, sanity-train, train, mlflow-ui, etc.
├── README.md                       ← roadmap with checkboxes per session
├── LICENSE                         ← MIT
└── docs/handoff.md                 ← THIS FILE
```

---

## 5. Session 1 progress (what's done vs pending)

**Completed tasks (#1–#18):**
1. Project scaffolding (folders, configs, src structure)
2. Poetry setup with python@3.12 (Python 3.13 was rejected due to compatibility)
3. pre-commit hooks (ruff, black, mypy, file checks)
4. Hydra configs (model, data, training, tracking, tuning)
5. README with badges, quickstart, roadmap
6. .gitignore aware of ML artifacts
7. `.env.example` with all secrets/URIs
8. Docker (CPU multi-stage; GPU separate)
9. CI workflow (lint+test+docker build); train/deploy workflows are placeholders
10. Makefile with `setup`, `sanity-train`, etc.
11. GCP setup guide in `docs/gcp_setup.md`
12. Roboflow download script (uses argparse, not Typer)
13. Dataset downloaded: roboflow-jvuqo/football-players-detection-2frwp v1
14. EDA notebook with 9 sections, executed and reviewed
15. Training script with full MLflow + Hydra instrumentation
16. Sanity training (2 epochs on MPS) executed successfully

**In progress:**
- Task #19: ngrok setup (just guided user through it)

**Pending:**
- Task #20: Colab notebook for full training
- Task #21: 30-epoch baseline training on Colab + close Session 1
- Sessions 2-8: HPO, error analysis, ONNX/serving, Docker for serving, deploy demo, CI/CD, drift monitoring, polish

---

## 6. Concepts already taught and internalized

The user is in active-learning mode. He has confirmed understanding (with corrections from us) on:

- **Why baseline-first** ("scientific method" + "fast iteration loop" framing)
- **Class imbalance in object detection** (focal loss, class-weighted loss, copy-paste augmentation, why simple oversampling doesn't translate from tabular)
- **mAP averaging is per-class then mean** (so slice analysis matters more than global mAP)
- **K-fold CV vs bootstrap** (when to use which, what set to apply each to, train/valid/test distinction)
- **EDA is part of MLOps but not via MLflow** (DVC, Great Expectations, FiftyOne, Evidently fill that role; MLflow is for training)
- **Object detection tiny-object considerations** (his dataset is 100% tiny objects, may need imgsz=960)
- **Aspect ratio anomaly on ball** (centered at ~0.5 instead of expected ~1.0, suggests motion blur or annotation noise)
- **MLflow as a library vs server** (logging URI is decoupled from training location)
- **Ultralytics callback system** (how `on_fit_epoch_end` bridges to MLflow)
- **Hydra config flattening to MLflow params**
- **Why MLflow 3.x uses aliases instead of stages**
- **Why training script disables Ultralytics' built-in MLflow** (we manage the run lifecycle)

He has NOT yet been explicitly taught:
- Anchor-free vs anchor-based detectors
- NMS (non-maximum suppression) internals
- Exact mAP calculation (IoU thresholds, AP integral)
- Mosaic / mixup augmentation mechanics
- LR schedulers (cosine, warmup) in depth
- ONNX/TensorRT export differences

He has expressed interest in revisiting some of these "more questions, but later" — note for future sessions.

---

## 7. Gotchas and decisions worth remembering

These cost us time in this session. Future assistants should not re-debug them.

**MLflow on macOS arm64:**
- MLflow 2.22.5 (and other 2.x late versions) crash with SIGSEGV when starting workers on Apple Silicon. Root cause: pyarrow 19 + gunicorn fork model.
- Fix: pin `mlflow = "^3.12.0"` in pyproject.toml. Already applied.
- Note for the future: if a user reports SIGSEGV with MLflow on Mac, jump directly to "upgrade to 3.x".

**Typer/Click conflict:**
- In Typer ≥0.12 + Click, `typer.Option(default, "--format", ...)` triggers `TypeError: Secondary flag is not valid for non-boolean flag` when the parameter name differs from the flag.
- Symptoms: parameter `format: str = typer.Option(None, "--format", ...)` fails.
- Workarounds attempted (and abandoned): removing `from __future__ import annotations`, renaming param, adding `Optional[]`.
- **Final fix:** rewrote `scripts/download_dataset.py` to use stdlib argparse. Lesson: for utility scripts, argparse is more boring and predictable than Typer.

**Ultralytics SETTINGS["datasets_dir"]:**
- Ultralytics resolves the `path` field of dataset YAMLs against its global `SETTINGS["datasets_dir"]` (default `~/datasets/`), NOT against the YAML's own directory.
- Roboflow's default `path: .` and `train: ../train/images` both break because of this.
- **Fix:** in `train.py`, we materialize a "resolved" data.yaml inside the Hydra output dir with an absolute `path:`. Function: `_materialize_absolute_data_yaml()`.
- We also updated the source `data.yaml` to use `path: .` and `train: train/images` (clean relative names).

**Pre-commit:**
- Black hook initially had `language_version: python3.11`. User only has 3.12. Fixed to `python3` (any python3).
- Ruff B008 false positives for Typer's `typer.Option(...)` in defaults: added `extend-immutable-calls` whitelist in `pyproject.toml`.

**Hydra strict mode:**
- New keys not in the config can't be added via `key=value` CLI syntax — must use `+key=value`.
- This is why `device` is in `configs/training/default.yaml` as `device: null` (instead of being added on demand). Allows `training.device=mps` override without `+`.

**MLflow Model Registry with `file://` backend (USER HIT THIS):**
- The user's current MLflow tracking URI defaults to `file:./mlruns`. Model Registry **doesn't work fully with the file backend** — `register_model()` and aliases require a database-backed tracking server.
- This is why the user reports "no model registered" despite the registration code running.
- The training script catches the exception in `_maybe_register_model()` and logs a warning, so training doesn't fail.
- **Fix planned for the user's next step:** switch tracking URI to SQLite backend: `mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 8080`. Then Registry works. Or use remote tracking server when GCP is ready.
- Document this in conversation when picking up.

**Path / workspace divergence (RESOLVED):**
- Earlier in the session, edits went to the assistant's temp `outputs/` directory while the user worked from his Documents folder. This caused changes to not propagate.
- Now resolved: user mounted his `Documents/Repositorios_Github/football-player-tracker` folder via Cowork. All edits go directly to his repo.
- **Future assistants: write to `/Users/jmvazqueznicolas/Documents/Repositorios_Github/football-player-tracker/` only. The temp `outputs/` is a leftover that should NOT be used.**

---

## 8. Current state

**GCP:**
- Account created from scratch by the user.
- Status: waiting for GPU quota approval (T4 in us-central1).
- Bucket: NOT yet created.
- Service account: NOT yet created.
- DVC: NOT yet initialized.
- Plan: skip GCP-dependent steps (DVC, GCS, Compute Engine training) until quota approves. Use Colab in parallel.

**MLflow:**
- Running locally on user's Mac at `http://127.0.0.1:8080` (file backend).
- Sanity training visible with: experiment created, run logged with params, metrics, artifacts (config, weights, plots).
- Model Registry NOT working because of file backend limitation — user noted "no model registered". To be addressed in next step.

**Repo state:**
- All scaffolding committed and working.
- Training pipeline functional end-to-end on MPS.
- Sanity check completed.

**Hydra outputs:**
- Each `train.py` invocation creates `runs/<timestamp>/` with logs, resolved config, MLflow artifacts.
- These are gitignored.

**Roboflow API key:**
- User has set `ROBOFLOW_API_KEY` in his `.env` (not committed).

---

## 9. Immediate next steps (where to pick up)

In order:

1. **Address the Model Registry issue:** Switch tracking URI from `file:./mlruns` to `sqlite:///mlflow.db` so the Registry works. Easiest: update `.env` to `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` and instruct user to relaunch `mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 8080`. Re-run sanity-train to confirm model appears in Registry with `@candidate` alias.

2. **Configure ngrok (Task #19):**
   - User should already have a ngrok account and authtoken configured (we walked through it).
   - Commands he runs:
     - Terminal 1: `poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 8080`
     - Terminal 2: `ngrok http 8080`
   - User captures the `https://*.ngrok-free.app` URL.

3. **Create the Colab notebook (Task #20):**
   - Notebook at `notebooks/colab_train.ipynb` (in the repo) that:
     - Mounts (or pulls) the repo and dataset
     - Installs Poetry deps minimally (just `ultralytics`, `mlflow`, `python-dotenv`, `hydra-core`, `omegaconf`, `loguru`, `pyyaml`)
     - Sets `MLFLOW_TRACKING_URI` to the user's ngrok URL
     - Runs `python -m football_tracker.training.train training.epochs=30 training.device=0` (T4 GPU)
     - User sees runs appear LIVE in his local MLflow UI

4. **Full baseline training (Task #21):**
   - Launch 30 epochs on Colab T4
   - Verify MLflow logged everything via ngrok
   - Download `best.pt` and copy to `models/baseline/`
   - Document final metrics in README under "Results" section

5. **Close Session 1.** Commit everything, push to GitHub.

6. **Session 2 onward:** Already scoped in README and `architecture.md`. Next is integrating W&B in parallel with MLflow, then Optuna HPO.

---

## 10. Style notes for the next assistant

- Spanish, "tú" form, casual but technical.
- User likes step-by-step instructions with copy-paste-ready commands.
- He DOES want explanations — don't just say "run this", explain WHY.
- He pushes back when something feels like a shortcut to the MLOps learning goal. Honor those pushbacks.
- When he makes observations (active learning mode), validate the correct parts cleanly, correct the wrong parts directly without being mean, and add depth on what he missed.
- He responds well to interview-prep framing ("if asked in an interview..."). Use it.
- Avoid bullet-vomit. Mix prose with structure. Lists when they help; paragraphs when explaining.
- Tasks tool: he likes seeing the task list. Use it.
- TodoWrite / Task tools: maintain them rigorously. They're rendered in his UI.

---

## 11. Reference URLs

- Roboflow dataset: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-2frwp/dataset/1
- Roboflow API keys: https://app.roboflow.com/settings/api
- ngrok dashboard: https://dashboard.ngrok.com
- Hugging Face Spaces: https://huggingface.co/spaces
- Ultralytics docs: https://docs.ultralytics.com
- MLflow docs (3.x): https://mlflow.org/docs/latest/

---

## 12. Key file references for new assistant

If the next assistant needs to read just a few files to get context, in this order:
1. `docs/handoff.md` (this file)
2. `README.md` (project description + roadmap)
3. `docs/architecture.md` (stack table, reproducibility contract)
4. `src/football_tracker/training/train.py` (the heart of the pipeline)
5. `src/football_tracker/training/callbacks.py` (Ultralytics → MLflow bridge)
6. `configs/config.yaml` and `configs/training/default.yaml` (the contract)
7. `notebooks/01_eda.ipynb` (what we know about the data)
