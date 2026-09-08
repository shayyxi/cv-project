# Hosting the CV / PPE pipeline on Render

This guide walks through deploying this project to [Render](https://render.com)
as a **Background Worker** backed by **Render Postgres** and a **persistent disk**.
It is written for this codebase specifically, so it starts from what the app
actually is and works forward from there.

---

## 1. What you are deploying (and why it shapes the setup)

| Fact about this repo | Consequence on Render |
|---|---|
| Entry point is `python -m app.main`, an infinite `while True` polling loop (`app/application/runner.py`). It never binds an HTTP port. FastAPI/uvicorn are listed in `pyproject.toml` but never used. | Deploy as a **Background Worker**, not a Web Service. A Web Service expects a port and would be treated as unhealthy. |
| Needs PostgreSQL via `DATABASE_URL` in the form `postgresql+psycopg://...` (psycopg v3, see `app/storage/database.py`). | Use **Render Postgres**. Render hands you a `postgresql://` URL; you must rewrite the scheme (see step 5). |
| Schema is managed by Alembic; nothing in the app runs migrations automatically. | Run `alembic upgrade head` as the service's **Pre-Deploy Command**. |
| Writes images, PDFs, heatmaps, crops and the risk model to `data/{raw,processed,failed,analytics}` and stores those paths in the DB. The heatmap job reads raw frames back from disk. | Attach a **persistent disk** at `/data` and point the `LOCAL_*_DIR` variables at it. Without a disk, every deploy or restart wipes the files. |
| Model weights (`app/processing/cv/weights/*.pt`, ~150 MB) are gitignored, so they will not be in your repo. | Commit the small custom model (`best.pt`, 19 MB) and download the public `yolov8x.pt` (131 MB) at Docker build time. |
| `torch` is a bare dependency. On Linux, plain `pip install torch` pulls the multi-GB CUDA build. Render has no GPUs; the engine already falls back to CPU (`_select_device` in `ppe_vision_engine.py`). | Install the **CPU-only** PyTorch wheel first to keep the image small and the build fast. |
| YOLOv8x + SAHI (4x3 tiles at 1000 px) on CPU, with `torch.set_num_threads(2)` hard-coded. | Use at least a 2 GB instance; 2 CPU / 4 GB is the sensible choice. |
| Daily jobs fire on the first cycle at or after `REPORT_HOUR` using `datetime.now()` (local time). | Containers run in UTC. Set the `TZ` env var or your report will fire at the wrong hour. |

Resulting architecture:

```
Panomax cameras (HTTPS)  <----  Background Worker  ---->  Render Postgres
                                  |  /data (disk)
                                  |
                                  +---> WordPress webhook / SMTP (optional, outbound only)
```

---

## 2. Prerequisites

1. **A Git repository.** This folder is not a git repo yet. Render deploys from
   GitHub, GitLab or Bitbucket:

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

   `.gitignore` already excludes `.env`, `data/` and `*.pt`. Double-check that
   `.env` is **not** staged before the first push.

2. **A Render account with a payment method.** Background Workers and
   persistent disks have no free tier. (Free Postgres exists but expires after
   30 days, so use a paid database plan for anything real.)

3. **The custom PPE model** `app/processing/cv/weights/best.pt` on the machine
   you push from.

---

## 3. Prepare the repository (one-time changes)

### 3.1 Un-ignore the custom model so it ships with the repo

`best.pt` is 19 MB, well under GitHub's 100 MB per-file limit. Add this line to
`.gitignore` **after** the existing `*.pt` line:

```gitignore
# Model weights
*.pt
!app/processing/cv/weights/best.pt
```

`yolov8x.pt` (131 MB) stays ignored; the Dockerfile downloads it from the
official Ultralytics release instead.

### 3.2 Replace the Dockerfile

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        curl \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first, so `pip install .` below does not pull the CUDA build.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY pyproject.toml ./
RUN pip install .

COPY . .

# yolov8x.pt is gitignored (131 MB). Fetch it from the public Ultralytics release.
RUN curl -fsSL -o app/processing/cv/weights/yolov8x.pt \
      https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt

CMD ["python", "-m", "app.main"]
```

### 3.3 Trim the Docker build context

Add these to `.dockerignore` so the ~13 MB of test JPEGs in the repo root and
the coverage file are not uploaded on every build:

```
*.jpg
.coverage
docs/
```

### 3.4 (Optional but recommended) Accept Render's native database URL

Render's generated connection string starts with `postgresql://`, which makes
SQLAlchemy look for `psycopg2` (not installed) and crash with
`ModuleNotFoundError: No module named 'psycopg2'`. You can either paste a
rewritten URL by hand (step 5) or make the app normalise it. In
`app/config.py`:

```python
from pydantic import Field, field_validator
# ...
class Settings(BaseSettings):
    # ... existing fields ...

    @field_validator("database_url")
    @classmethod
    def _use_psycopg_driver(cls, value: str) -> str:
        for prefix in ("postgres://", "postgresql://"):
            if value.startswith(prefix):
                return "postgresql+psycopg://" + value[len(prefix):]
        return value
```

With this in place, `render.yaml` can wire `DATABASE_URL` straight from the
database with `fromDatabase`, and nothing has to be copied by hand.

Commit and push these changes before continuing.

---

## 4. Create the database

Dashboard: **New → Postgres**.

| Setting | Value |
|---|---|
| Name | `cv-ppe-db` |
| Database | `ai_vision` |
| User | `pipeline_user` |
| Region | Same region you will use for the worker (e.g. `frankfurt` if the cameras are in Europe). The internal URL only works within one region. |
| PostgreSQL version | 16 (matches `docker-compose.yml`) |
| Plan | Smallest paid plan is fine to start. `free` expires after 30 days. |

After creation, open the database page and copy the **Internal Database URL**.
It looks like:

```
postgresql://pipeline_user:<password>@dpg-xxxxxxxx-a/ai_vision
```

---

## 5. Create the Background Worker

Dashboard: **New → Background Worker**, connect the repository.

| Setting | Value |
|---|---|
| Name | `cv-ppe-pipeline` |
| Region | Same as the database |
| Runtime / Language | **Docker** (Render detects the `Dockerfile`) |
| Dockerfile path | `./Dockerfile` |
| Docker command | leave empty (uses the image `CMD`) |
| Instance type | `2c-4g` recommended. `1c-2g` is the minimum worth trying. |
| Pre-Deploy Command | `alembic upgrade head` |
| Auto-Deploy | Yes (deploys on every push to the branch) |

### 5.1 Persistent disk

In the service's **Disks** section (or during creation):

| Setting | Value |
|---|---|
| Name | `pipeline-data` |
| Mount path | `/data` |
| Size | 10 GB to start. You can grow it later, never shrink it. |

Raw frames are 2 to 5 MB each and processed frames are similar, so estimate
`cameras x frames per day x ~8 MB` and size accordingly, or add a cleanup job.

### 5.2 Environment variables

Set these in the service's **Environment** tab. They replace the local `.env`
(`pydantic-settings` reads real env vars first; `.env` is excluded from the
image anyway). Only `DATABASE_URL` and `CAMERA_IDS` are strictly required.

| Key | Value | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://pipeline_user:<password>@dpg-xxxxxxxx-a/ai_vision` | Paste the Internal Database URL and **change the scheme** to `postgresql+psycopg://` (skip the rewrite if you applied step 3.4). |
| `CAMERA_IDS` | `["12846","12875","12990"]` | Must be a JSON list of strings. Use your real camera IDs. |
| `APP_ENV` | `production` | |
| `LOG_LEVEL` | `INFO` | |
| `TZ` | e.g. `Europe/Vienna` | IANA zone name. Controls when `REPORT_HOUR` fires. |
| `STORAGE_PROVIDER` | `local` | Only provider implemented. |
| `LOCAL_RAW_DIR` | `/data/raw` | On the disk. |
| `LOCAL_PROCESSED_DIR` | `/data/processed` | On the disk. |
| `LOCAL_FAILED_DIR` | `/data/failed` | On the disk. |
| `LOCAL_ANALYTICS_DIR` | `/data/analytics` | On the disk. Reports, heatmaps, label queue and `risk_model.joblib` live here. |
| `FTP_POLL_INTERVAL_SECONDS` | `60` | Sleep between cycles. |
| `PROCESSOR_SLEEP_SECONDS` | `5` | |
| `REPORT_HOUR` | `18` | Local hour (0-23) for daily jobs. |
| `REPORT_DAYS` | `7` | Days covered by each report. |
| `REPORT_DELIVER_WEBHOOK` | `false` / `true` | Needs the two WordPress vars below when `true`. |
| `WORDPRESS_WEBHOOK_URL` | `https://...` | Optional. |
| `WORDPRESS_API_KEY` | secret | Optional. |
| `REPORT_DELIVER_EMAIL` | `false` / `true` | Needs the SMTP vars below when `true`. |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_TO` | as needed | Use port `587` (STARTTLS) or `465`. Outbound port 25 is not usable on Render. |

`POSTGRES_PASSWORD` is only used by `docker-compose.yml` and is not needed here.

Click **Create Background Worker**. The first build takes a while (PyTorch,
Ultralytics, SAHI, OpenCV, scikit-learn, plus the 131 MB weights download).

---

## 6. Migrations

The Pre-Deploy Command `alembic upgrade head` runs after every successful build
and before the new container starts. `alembic/env.py` reads
`settings.database_url`, so it uses the same `DATABASE_URL` you set above. If
the migration fails, the deploy is aborted and the previous version keeps
running.

To run migrations by hand from your own machine instead, use the database's
**External Database URL** (SSL required):

```powershell
$env:DATABASE_URL = "postgresql+psycopg://pipeline_user:<password>@dpg-xxxxxxxx-a.frankfurt-postgres.render.com/ai_vision?sslmode=require"
alembic upgrade head
```

---

## 7. Verify the deployment

1. **Logs** tab of the worker. A healthy start looks like:

   ```
   ... | INFO     | app.application.runner | Application started.
   ... | INFO     | app.ingestion.ingestion_service | Starting ingestion cycle for 3 cameras.
   ```

   Expect the first cycle to be slow: loading YOLOv8x on CPU and running 12
   SAHI tiles per frame can take a minute or more per image.

2. **Shell** tab (opens a shell inside the running container):

   ```bash
   ls /data/raw /data/processed
   python -m scripts.inspect_db
   python -m scripts.analytics score
   python -m scripts.analytics report          # generate a PDF on demand
   ```

3. **Database** page → *Metrics* shows connections; you can also connect with
   `psql` using the External URL.

---

## 8. Infrastructure as code: `render.yaml` (Blueprint)

Instead of clicking through the dashboard, commit this file to the repo root
and use **New → Blueprint**. Render creates the database, the worker, the disk
and the env vars from it, prompting for any value marked `sync: false`.

```yaml
# render.yaml
services:
  - type: worker
    name: cv-ppe-pipeline
    runtime: docker
    region: frankfurt            # must match the database region
    plan: 2c-4g                  # 1c-2g is the minimum worth trying
    dockerfilePath: ./Dockerfile
    dockerContext: .
    autoDeploy: true
    preDeployCommand: alembic upgrade head
    disk:
      name: pipeline-data
      mountPath: /data
      sizeGB: 10
    envVars:
      # Option A (no code change): paste the URL with the psycopg scheme when prompted.
      - key: DATABASE_URL
        sync: false
      # Option B (after applying step 3.4): let Render inject it.
      # - key: DATABASE_URL
      #   fromDatabase:
      #     name: cv-ppe-db
      #     property: connectionString
      - key: APP_ENV
        value: production
      - key: LOG_LEVEL
        value: INFO
      - key: TZ
        value: Europe/Vienna
      - key: STORAGE_PROVIDER
        value: local
      - key: LOCAL_RAW_DIR
        value: /data/raw
      - key: LOCAL_PROCESSED_DIR
        value: /data/processed
      - key: LOCAL_FAILED_DIR
        value: /data/failed
      - key: LOCAL_ANALYTICS_DIR
        value: /data/analytics
      - key: CAMERA_IDS
        value: '["12846","12875","12990"]'
      - key: FTP_POLL_INTERVAL_SECONDS
        value: "60"
      - key: PROCESSOR_SLEEP_SECONDS
        value: "5"
      - key: REPORT_HOUR
        value: "18"
      - key: REPORT_DAYS
        value: "7"
      - key: REPORT_DELIVER_WEBHOOK
        value: "false"
      - key: REPORT_DELIVER_EMAIL
        value: "false"
      - key: WORDPRESS_WEBHOOK_URL
        sync: false
      - key: WORDPRESS_API_KEY
        sync: false
      - key: SMTP_HOST
        sync: false
      - key: SMTP_PORT
        value: "587"
      - key: SMTP_USER
        sync: false
      - key: SMTP_PASSWORD
        sync: false
      - key: SMTP_TO
        sync: false

databases:
  - name: cv-ppe-db
    databaseName: ai_vision
    user: pipeline_user
    region: frankfurt
    plan: 0.1c-256mb             # smallest paid plan; "free" expires after 30 days
    postgresMajorVersion: "16"
```

---

## 9. Alternative: deploy a pre-built image

If you would rather not commit any weights, build the image locally (the
current `.dockerignore` does not exclude `*.pt`, so both weight files are
copied in by `COPY . .`), push it to Docker Hub or GHCR, and choose
**Deploy an existing image** when creating the Background Worker. You lose
auto-deploy on push and have to push a new image for every change, but the
repo stays lean. The disk, env vars and pre-deploy command work the same way.

---

## 10. Sizing and cost notes

- **Instance.** `torch.set_num_threads(2)` caps inference at two threads, so
  `2c-4g` is the sweet spot. Bigger CPU counts will not make a frame process
  faster; more RAM only matters if you raise the SAHI grid or image size.
- **Cheaper inference.** If cost matters more than accuracy, switch
  `models.person.path` in `app/processing/cv/config/vision_config.yaml` from
  `yolov8x.pt` to `yolov8m.pt` (or `yolov8n.pt`) and adjust the `curl` line in
  the Dockerfile. That alone lets `1c-2g` keep up comfortably.
- **Disk.** Billed per GB-month. Raw plus processed output is roughly 8 MB per
  frame per camera.
- **Database.** The tables are small (one row per frame plus detections);
  the smallest paid plan is plenty for a long time.
- **Cron Jobs are not a substitute.** `python -m app.main --run-once` would
  work as a Render Cron Job, but cron jobs cannot mount disks, so files would
  vanish between runs. Stick with the worker.
- Check the current prices at https://render.com/pricing.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'psycopg2'` at startup or during pre-deploy | `DATABASE_URL` starts with `postgresql://`. Rewrite it to `postgresql+psycopg://` or apply step 3.4. |
| `FileNotFoundError: ... weights/best.pt` | `best.pt` is still gitignored. Apply step 3.1 and confirm the file is in the repo (`git ls-files app/processing/cv/weights`). |
| `curl: (22) ... yolov8x.pt` during build | GitHub release download failed. Retry the deploy; if it persists, host the file yourself and change the URL. |
| Build is extremely slow or the image is several GB | CPU-only torch step missing; `pip install .` pulled the CUDA build. Use the Dockerfile in step 3.2. |
| Worker restarts with out-of-memory | Move to a larger instance, or use a smaller person model (step 10). |
| Files disappear after each deploy | Disk not attached, or `LOCAL_*_DIR` still point at `/app/data` instead of `/data/...`. |
| Daily report never runs, or runs at the wrong hour | `TZ` not set; the container clock is UTC and `AnalyticsJobs.run_due` uses local time. |
| Heatmap job logs "No raw images on disk" | Expected until the first frame for that camera has been saved to `/data/raw`. |
| `InFailedSqlTransaction` repeated every cycle | Fixed by the runner's rollback, but check the preceding stack trace in the logs for the real error. |
| Email delivery times out | Use SMTP port 587 or 465, not 25. |
| Deploy shows a short gap in processing | Services with a disk cannot do zero-downtime deploys. The loop resumes where it left off; the `job_runs` table prevents duplicate daily jobs. |
| Want to run two instances | Do not. A disk restricts the service to one instance, and two pollers would double-process frames. |

---

## 12. Day-to-day operations

- **Deploy a change:** push to the connected branch. Render builds, runs
  `alembic upgrade head`, then swaps the container.
- **Change a setting:** edit the env var in the dashboard; Render restarts the
  worker automatically.
- **Manual jobs:** open the Shell tab and use `python -m scripts.analytics ...`
  (`score`, `trends`, `repeat`, `heatmap`, `report`, `risk --train`,
  `export-crops`).
- **Reset data:** `python -m scripts.clear_database` and
  `python -m scripts.clear_storage` from the Shell tab.
- **Backups:** Render Postgres takes daily snapshots on paid plans. The disk
  has snapshots too, but you can also `tar` `/data` from the Shell and download
  it.
- **Logs:** everything goes to stdout (`app/utils/logging.py`), so the Logs tab
  and Render's log streams capture it all. Raise `LOG_LEVEL` to `DEBUG` when
  diagnosing a stuck cycle.
