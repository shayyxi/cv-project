# Retraining the PPE model with Roboflow

The pipeline already collects training material: once a day the
label-queue job writes crops of non-compliant workers to
`data/analytics/label_queue/`, together with `_annotations.coco.json`
holding the model's own helmet / vest / boots boxes as pre-labels.

This guide closes the loop:

```
label_queue/  --(scripts.roboflow_upload)-->  Roboflow project  -->  review labels
Roboflow version  --(scripts.train_ppe)-->  ultralytics  -->  app/processing/cv/weights/best.pt
```

Everything lives in `app/retraining/` and the two scripts; the
pipeline's own code is unchanged.

---

## 1. One-time setup

1. Make sure the Roboflow SDK is installed. It is a project dependency
   (`roboflow` in `pyproject.toml`), so `pip install .` and the Docker
   image already include it. Only an environment created before it was
   added needs a manual install:

   ```bash
   pip install roboflow
   ```

2. Create an **object detection** project in Roboflow with exactly these
   class names: `helmet`, `vest`, `boots`. The vision engine matches
   detections by class name (`classes.ppe` in
   `app/processing/cv/config/vision_config.yaml`), so other names would
   train a model that detects nothing usable.

3. Add to `.env` (or the Render environment):

   | Key | Value | Notes |
   |---|---|---|
   | `ROBOFLOW_API_KEY` | your private API key | Roboflow → Settings → API |
   | `ROBOFLOW_PROJECT` | project slug | last part of the project URL |
   | `ROBOFLOW_WORKSPACE` | workspace slug | optional; defaults to the key's workspace |
   | `ROBOFLOW_UPLOAD_AS_PREDICTION` | `true` (default) | pre-labels arrive as model predictions to review. `false` makes them ground truth immediately, with no human check. |
   | `ROBOFLOW_VALID_SPLIT_PERCENT` | `20` (default) | share of uploads assigned to the `valid` split |
   | `ROBOFLOW_BATCH_PREFIX` | `label-queue` (default) | uploads are batched as `<prefix>-YYYY-MM-DD` |

   These are read by `app/retraining/config.py` from the environment
   and `.env`; `app/config.py` ignores them.

---

## 2. Uploading the label queue

```bash
python -m scripts.roboflow_upload --dry-run   # how many crops are waiting
python -m scripts.roboflow_upload             # upload them
```

What one run does:

- Uploads every crop in the label queue that has not been uploaded
  before, in a batch named `label-queue-<today>`, tagged `label-queue`
  and `camera-<id>` (the camera id is the first part of the file name).
- Attaches that crop's boxes from `_annotations.coco.json` as
  pre-labels. Crops with no box go in unannotated.
- Assigns `train` / `valid` deterministically from the file name, so a
  re-exported crop keeps its split.
- Records each upload in `data/analytics/label_queue/_roboflow_uploads.json`
  (keyed per project). Delete an entry there to re-send a crop; Roboflow
  itself drops pixel-identical duplicates.
- A crop whose upload fails is not recorded, so the next run retries it.

### Making it automatic

The daily analytics job fills the queue at `REPORT_HOUR`. To get the
crops to Roboflow without anyone running anything, keep the uploader
running in watch mode next to the pipeline:

```bash
python -m scripts.roboflow_upload --watch                 # every 30 min
python -m scripts.roboflow_upload --watch --interval 600  # every 10 min
```

With Docker Compose, in a second terminal:

```bash
docker compose exec app python -m scripts.roboflow_upload --watch
```

or add a service that runs that command with the same `env_file` and
`volumes` as `app`. On Render, use the Shell tab or add a second
Background Worker with Docker command
`python -m scripts.roboflow_upload --watch` sharing the same disk.

A scheduled task also works: run the plain command (no `--watch`) from
cron or Windows Task Scheduler shortly after `REPORT_HOUR`.

`--export` runs the label-queue export first (same as
`scripts.analytics export-crops`, with `--days`, `--limit`, `--all`),
useful for a manual "export and upload now".

### Reviewing in Roboflow

Uploaded crops appear in the project's **Annotate** view with the
model's boxes drawn as suggestions. Approve or fix them; only reviewed
images become part of a dataset version. Set
`ROBOFLOW_UPLOAD_AS_PREDICTION=false` if you would rather trust the
model's boxes as-is (then errors are trained back in).

---

## 3. Exporting and training

```bash
python -m scripts.train_ppe --generate --install
```

Step by step, that:

1. Generates a new dataset version in Roboflow from everything labelled
   so far (no resize, no Roboflow augmentation; ultralytics does its
   own), waits for it, and downloads it in YOLOv8 format to
   `data/analytics/datasets/<project>-v<N>/`.
2. Rewrites `data.yaml` with absolute paths (Roboflow's relative ones
   only work from one directory) and checks the class names against
   `vision_config.yaml`.
3. Fine-tunes starting from the deployed
   `app/processing/cv/weights/best.pt` (falls back to `yolov8n.pt` when
   it is missing), 50 epochs at 640 px by default, into
   `data/analytics/training/ppe-<timestamp>/`.
4. Prints precision / recall / mAP50 / mAP50-95 and the path of
   `best.pt`.
5. Always compares: the deployed `best.pt` and the new one are
   evaluated on the same split of the same dataset version (test when
   the version has one, else valid) and a table of both is printed,
   headline metrics plus mAP50-95 per class. The result is written to
   `data/analytics/training/latest_comparison.json` (and
   `comparison.json` in the run folder) for the page.
6. Only with `--install`, meant for scripted runs, the new file is
   copied over the engine's weights, and only if its overall mAP50-95
   is not lower; the old file is kept as `best_backup_<timestamp>.pt`
   next to it. Without `--install` nothing is replaced: the page shows
   the numbers and a **Replace deployed model with this one** button.

Other ways to run it:

```bash
python -m scripts.train_ppe                     # latest existing version, compare only
python -m scripts.train_ppe --version 3         # a specific version
python -m scripts.train_ppe --data /path/data.yaml   # any local YOLO dataset
python -m scripts.train_ppe --epochs 100 --batch 32 --device 0 --generate --install
python -m scripts.train_ppe --base-model yolov8s.pt --generate   # start from scratch-ish
python -m scripts.train_ppe --weights /path/to/run/weights/best.pt
                                                  # no training: compare an existing file
```

`--version-settings settings.json` passes Roboflow preprocessing /
augmentation options to `--generate`; the keys are the ones the
Roboflow UI exposes (see the `generate_version` docstring in the
Roboflow SDK).

### Where to train

Training on the Render worker is not practical (CPU only, 2 threads).
Run `train_ppe` on a machine with a GPU, or on CPU for a small
dataset, then either:

- run it with `--install` on the machine you build the Docker image
  from and redeploy (the weights ship with the image), or
- copy the new `best.pt` into `app/processing/cv/weights/` on the host;
  `docker-compose.yml` mounts that folder read-only into the container,
  so restart the `app` service to load it.

The engine only reads the weights at start-up, so a restart is always
needed after installing new weights.

### Class order

Roboflow exports classes alphabetically (`boots, helmet, vest`), while
the deployed `best.pt` was trained as `helmet, vest, boots`. YOLO label
files store class indices, so that difference would make every box
train and score against the wrong class. Both `train_ppe` and the gate
handle this: before training or evaluating a model, the dataset is
rewritten into that model's class order under
`<dataset>/aligned-<order>/` (hard links to the images, remapped label
files), so a fine-tune keeps the deployed order and the comparison is
fair. Nothing to configure; it only kicks in when the orders differ.

### Sanity checks before installing

- The gate compares overall mAP50-95. Read the per-class rows too: a
  gain on helmets can hide a loss on boots, and the script warns when
  a class drops by more than 0.05.
- The split used for the comparison must contain real images. If the
  log says "no valid split - validating on the train split", set
  `ROBOFLOW_VALID_SPLIT_PERCENT` above zero, or move some images to
  `valid` or `test` in Roboflow, before trusting the numbers.
- Spot-check a few processed frames after the restart.

---

## 4. Buttons: the Streamlit page

For the two manual steps there is a page with buttons, so nobody has
to type the commands. It needs Streamlit (not in `pyproject.toml`):

```bash
pip install streamlit
```

Start it by double-clicking `retrain_ui.bat` in the project folder, or:

```bash
streamlit run scripts/retrain_ui.py
```

The browser opens on http://localhost:8501 with:

- **Upload label queue to Roboflow**: shows how many crops are waiting
  and runs `python -m scripts.roboflow_upload`. "Dry run" only counts.
- **Train and deploy**: epochs, device and "generate a new version",
  then runs `python -m scripts.train_ppe` with those flags. Nothing is
  replaced automatically; the comparison and the replace button are at
  the bottom of the page.

Each button starts the script in the background. The page shows a
progress bar built from the log (dataset download, epoch and batch,
validation, model comparison, outcome), a **Stop** button while a job
runs, and the last 200 log lines in an expander, refreshing every two
seconds. Only one job runs at a time; the buttons are disabled
meanwhile. Logs are kept under `data/analytics/ui_logs/`.

Stopping a training kills it and everything it spawned; the run folder
keeps `last.pt`, and nothing is installed.

Under the status, **Last model comparison** shows the most recent
comparison: deployed vs new for precision, recall, mAP50 and mAP50-95,
mAP50-95 per class with the delta, and a bar chart. It is read from
`data/analytics/training/latest_comparison.json`, which the training
script writes after every comparison (a copy sits in the run folder as
`comparison.json`).

Below the numbers is the decision: **Replace deployed model with this
one** copies the new `best.pt` over the engine's weights and keeps the
previous file as a backup. A model that scores lower than the deployed
one shows a red warning and needs a confirmation tick before the button
works. After a replacement the section says when it was installed, and
the pipeline must be restarted to load the new weights.

### Crop region per camera

The last section of the page replaces the Colab cell for marking a
camera's inference region. Pick the camera (or type a new id), upload a
frame or take the latest raw frame of that camera, and click the
corners of the area on the picture: two clicks make a rectangle, more
clicks a polygon. The saved region is drawn in blue, your new points in
red, and a preview shows exactly what the cropper will feed the model.
**Save region to config** writes the polygon, in full-image pixels,
into `app/processing/Image_cropper/config/image_crop_config.yaml` for
that camera and leaves the other cameras untouched.

No restart is needed: the cropper checks the file's modification time
before each frame and reloads it when it changed (a half-written or
invalid file keeps the previous regions). With docker-compose the
config folder is mounted from the host into both `app` and
`retrain-ui`, so a save from the page is picked up by the pipeline's
next frame. On Render the file is baked into the image: commit, push
and redeploy. The click component is the `streamlit-image-coordinates`
package, listed in `pyproject.toml`.

Run the page on the machine that should train (a GPU box or your PC
with `.env` and the weights folder). The label queue itself is filled
by the pipeline's daily job, so there is no export button.

### Running it with docker-compose

`docker-compose.yml` has two extra services next to `app`:
`retrain-ui` (the page, same image as the pipeline) and `caddy` (HTTPS
and a password in front of it). Both the pipeline and the page use
`/data` inside their containers, so they share the label queue and the
weights folder. Port 8501 is never published; Caddy is the only way in.

One-time setup on the server:

1. Point a DNS A record for your domain at the server and open ports
   80 and 443 in the firewall.
2. Create a password hash:

   ```bash
   docker compose run --rm caddy caddy hash-password --plaintext 'your-password'
   ```

3. In `Caddyfile`, replace the domain, the user name and the hash.
   Without a domain, use the commented `:80` block instead (plain HTTP
   on the office network at `http://<server-ip>`).
4. Build and start everything:

   ```bash
   docker compose up -d --build
   ```

Then open `https://retrain.yourdomain.com`, log in, and use the two
buttons. After the Train button installs a new `best.pt`, restart the
pipeline container so it loads it:

```bash
docker compose restart app
```

Training inside the container runs on CPU unless you add GPU
passthrough to the `retrain-ui` service. For regular retraining, a
GPU machine running the page locally is faster.

---

## 5. Files

| Path | Role |
|---|---|
| `app/retraining/config.py` | `ROBOFLOW_*` settings, Roboflow connection |
| `app/retraining/upload.py` | upload service with the per-project manifest |
| `app/retraining/dataset.py` | version generate / download, `data.yaml` fix, class check |
| `app/retraining/train.py` | ultralytics training, `install_weights` |
| `scripts/roboflow_upload.py` | upload CLI, `--watch`, `--export` |
| `scripts/train_ppe.py` | training CLI |
| `app/retraining/jobs.py` | background job runner behind the buttons |
| `scripts/retrain_ui.py`, `retrain_ui.bat` | Streamlit page and its launcher |
| `data/analytics/ui_logs/` | logs of button-started jobs (runtime data) |
| `data/analytics/label_queue/_roboflow_uploads.json` | what has been uploaded (runtime data) |
| `data/analytics/datasets/` | downloaded dataset versions (runtime data) |
| `data/analytics/training/` | training runs with `weights/best.pt` (runtime data) |
