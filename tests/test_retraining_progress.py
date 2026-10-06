from app.retraining.progress import parse_progress


def _ft(progress: dict) -> tuple:
    """(fraction, text) for compact assertions."""

    return progress["fraction"], progress["text"]


# Excerpts of real logs from this project's runs.
TRAIN_LOG = """\
python.exe -m scripts.train_ppe --generate --install --epochs 1

2026-10-01 19:14:35 | INFO     | app.retraining.dataset | Downloading Roboflow version 2 as yolov8 -> D:/x/PPE-crops-project-v2
Dataset: D:/x/PPE-crops-project-v2/data.yaml
Starting training for 2 epochs...
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/2         0G      1.677      7.235     0.9699         15        640:   3%|3         | 1/33 [01:56<1:01:59, 116.23s/it]
        1/2         0G      1.474      7.263     0.9561         19        640:   6%|6         | 2/33 [03:40<56:24, 109.18s/it]
"""

SETUP_LOG = (
    "2026-10-03 14:14:50 | INFO | app.retraining.dataset | Downloading Roboflow version 5 as yolov8 -> D:/x/v5\n"
    "Downloading Dataset Version Zip in D:/x/v5 to yolov8::  57%|#####7    | 21315/37233 [00:13<00:11, 1420.92it/s]\n"
)


def test_train_progress_follows_phases() -> None:
    download_only = "\n".join(TRAIN_LOG.splitlines()[:3])
    progress = parse_progress("train", download_only)
    assert _ft(progress) == (0.01, "Downloading the dataset from Roboflow")
    assert progress["phase"] == "download"

    progress = parse_progress("train", TRAIN_LOG)
    assert progress["text"] == "Epoch 1/2, batch 2/33"
    assert progress["phase"] == "train"
    assert (progress["epoch"], progress["epochs"]) == (1, 2)
    assert (progress["batch"], progress["batches"]) == (2, 33)
    assert abs(progress["phase_fraction"] - 2 / 33) < 1e-9
    # 5 % before training, then 85 % spread over the epochs.
    assert abs(progress["fraction"] - (0.05 + 0.85 * (2 / 33) / 2)) < 1e-9

    validating = TRAIN_LOG + (
        "                 Class     Images  Instances      Box(P          R"
        "      mAP50  mAP50-95):  40%|####      | 2/5 [00:25<00:37, 12.50s/it]\n"
    )
    progress = parse_progress("train", validating)
    assert progress["text"] == "Validating epoch 1/2"
    assert progress["phase"] == "validate"
    assert progress["phase_fraction"] == 0.4
    assert abs(progress["fraction"] - (0.05 + 0.85 * 0.5)) < 1e-9


def test_train_progress_covers_download_extract_and_scan_phases() -> None:
    progress = parse_progress("train", SETUP_LOG)
    assert progress["text"] == "Downloading the dataset 57%"
    assert progress["phase"] == "download"
    assert abs(progress["phase_fraction"] - 21315 / 37233) < 1e-9
    assert 0.01 < progress["fraction"] < 0.04

    extracting = SETUP_LOG + (
        "Extracting Dataset Version Zip to D:/x/v5 in yolov8::  54%|#####4    | 756/1391 [00:01<00:01, 626.07it/s]\n"
    )
    progress = parse_progress("train", extracting)
    assert progress["text"] == "Extracting the dataset 54%"
    assert progress["phase"] == "extract"
    assert 0.04 < progress["fraction"] < 0.05

    scanning = extracting + (
        "\x1b[34m\x1b[1mtrain: \x1b[0mScanning D:/x/labels... 303 images, 28 backgrounds, 0 corrupt:  62%|######1   | 303/491 [00:02<00:01, 145.70it/s]\n"
    )
    progress = parse_progress("train", scanning)
    assert progress["text"] == "Preparing training images 303/491"
    assert progress["phase"] == "prepare"
    assert abs(progress["phase_fraction"] - 303 / 491) < 1e-9
    assert progress["fraction"] == 0.05

    first_batches = scanning + (
        "Starting training for 7 epochs...\n"
        "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
        "  0%|          | 0/31 [00:00<?, ?it/s]\n"
        " 10%|#         | 3/31 [00:40<06:00, 12.87s/it]\n"
    )
    progress = parse_progress("train", first_batches)
    assert progress["text"] == "Epoch 1/7, batch 3/31"
    assert (progress["epoch"], progress["epochs"]) == (1, 7)
    assert abs(progress["phase_fraction"] - 3 / 31) < 1e-9
    assert abs(progress["fraction"] - (0.05 + 0.85 * (3 / 31) / 7)) < 1e-9


def test_train_progress_comparison_and_outcome() -> None:
    comparing = TRAIN_LOG + (
        "2026-10-01 19:25:44 | INFO | app.retraining.train | Evaluating D:/w/best.pt on the test split\n"
    )
    progress = parse_progress("train", comparing)
    assert _ft(progress) == (0.92, "Comparing with the deployed model (1/2)")
    assert progress["phase"] == "compare"
    assert progress["epochs"] == 2

    second = comparing + "... | Evaluating D:/r/best.pt on the test split\n"
    assert parse_progress("train", second)["fraction"] == 0.96

    kept = second + "Not installed: new mAP50-95 0.4328 is below the deployed 0.4852 on the test split.\n"
    progress = parse_progress("train", kept)
    assert _ft(progress) == (1.0, "Finished: new model not better, deployed weights kept")
    assert progress["phase"] == "done"
    assert progress["phase_fraction"] == 1.0

    installed = second + "Installed -> D:/w/best.pt\n"
    assert parse_progress("train", installed)["text"] == "Finished: new weights installed"

    no_install = TRAIN_LOG + "Weights not installed. Re-run with --install ...\n"
    assert parse_progress("train", no_install)["fraction"] == 1.0


def test_train_progress_reports_failures() -> None:
    crashed = TRAIN_LOG + "Traceback (most recent call last):\n  File x\nRuntimeError: boom\n"
    progress = parse_progress("train", crashed)
    assert progress["text"] == "Failed"
    assert progress["phase"] == "failed"
    assert progress["fraction"] is not None  # keeps the last known position

    exited = "Error: Roboflow is not configured; set ROBOFLOW_API_KEY\n"
    assert parse_progress("train", exited)["text"] == (
        "Failed: Roboflow is not configured; set ROBOFLOW_API_KEY"
    )


def test_upload_progress() -> None:
    progress = parse_progress("upload", "")
    assert _ft(progress) == (None, "Starting")
    assert progress["phase"] == "starting"

    log = (
        "2026-10-02 | INFO | app.retraining.upload | Roboflow upload 1/4: a.jpg\n"
        "2026-10-02 | INFO | app.retraining.upload | Roboflow upload 3/4: c.jpg\n"
    )
    progress = parse_progress("upload", log)
    assert _ft(progress) == (0.5, "Uploading crop 3 of 4")
    assert progress["phase_fraction"] == 0.5

    finished = log + (
        "2026-10-02 | INFO | app.retraining.upload | Roboflow upload: 4 uploaded, "
        "0 duplicates, 0 failed -> batch label-queue-2026-10-02 (ws/ppe)\n"
    )
    progress = parse_progress("upload", finished)
    assert progress["fraction"] == 1.0
    assert progress["phase"] == "done"
    assert progress["text"].startswith("Finished: 4 uploaded")

    dry = "0 crops pending upload (dry run; manifest: x)\n"
    assert parse_progress("upload", dry)["fraction"] == 1.0

    nothing = "... | Roboflow upload: nothing new in D:/q\n"
    assert parse_progress("upload", nothing)["text"] == "Nothing new to upload"


def test_unknown_job() -> None:
    progress = parse_progress("export", "anything")
    assert _ft(progress) == (None, "")
