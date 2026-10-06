"""
Turns a job's log into progress information for the Streamlit page.
Pure functions, no I/O.

parse_progress() returns:
    fraction        overall progress 0..1 (None before anything is known)
    text            one-line status, e.g. "Epoch 3/50, batch 12/31"
    phase           generate | download | extract | prepare | train |
                    validate | compare | done | failed | starting
    phase_fraction  progress of the current step 0..1 (None if unknown),
                    e.g. 0.67 while the dataset download is at 67 %
    epoch, epochs   current / total epochs during training (0 otherwise)
    batch, batches  current / total batches of the current epoch
"""

import re

# Terminal colour codes ultralytics prints (e.g. "\x1b[34m\x1b[1mtrain:").
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Roboflow SDK download / extraction, e.g.
# "Downloading Dataset Version Zip in ... to yolov8::  57%|#####7    | 21315/37233 ["
DOWNLOAD_LINE = re.compile(r"Downloading Dataset Version Zip.*?\|\s*(\d+)/(\d+)\s*\[")
EXTRACT_LINE = re.compile(r"Extracting Dataset Version Zip.*?\|\s*(\d+)/(\d+)\s*\[")

# ultralytics label scan, e.g. "train: Scanning ...labels... 303/491 ["
SCAN_LINE = re.compile(r"^(train|val): Scanning.*?\|\s*(\d+)/(\d+)\s*\[")

# ultralytics training line, e.g.
# "   3/50   0G   1.45   7.26   0.95   19   640:  36%|###  | 12/33 [..."
EPOCH_LINE = re.compile(r"^\s*(\d+)/(\d+)\s+\S+G\s.*?\|\s*(\d+)/(\d+)\s*\[")

# The very first batches print a bare bar before the epoch columns
# appear, e.g. "  0%|          | 0/31 [00:00<?, ?it/s]".
BARE_BAR_LINE = re.compile(r"^\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[")

# ultralytics validation line during training, e.g.
# "   Class  Images  Instances  Box(P  R  mAP50  mAP50-95):  20%|##  | 1/5 ["
VALIDATION_LINE = re.compile(r"^\s*Class\s+Images\s+Instances.*?\|\s*(\d+)/(\d+)\s*\[")

# app.retraining.upload logs "Roboflow upload 3/12: <file>" per crop.
UPLOAD_LINE = re.compile(r"Roboflow upload (\d+)/(\d+):")

# Where each phase of a training run sits on the overall bar.
GENERATE_AT = 0.01
DOWNLOAD_FROM, DOWNLOAD_TO = 0.01, 0.04
EXTRACT_FROM, EXTRACT_TO = 0.04, 0.05
TRAIN_START = 0.05
TRAIN_SPAN = 0.85
COMPARE_FIRST, COMPARE_SECOND = 0.92, 0.96


def _result(
    fraction: float | None,
    text: str,
    phase: str,
    phase_fraction: float | None = None,
    epoch: int = 0,
    epochs: int = 0,
    batch: int = 0,
    batches: int = 0,
) -> dict:
    return {
        "fraction": fraction,
        "text": text,
        "phase": phase,
        "phase_fraction": phase_fraction,
        "epoch": epoch,
        "epochs": epochs,
        "batch": batch,
        "batches": batches,
    }


def parse_progress(job: str, log_text: str) -> dict:
    """Progress information from the log so far (see module docstring)."""

    log_text = ANSI.sub("", log_text)

    if job == "train":
        return _train_progress(log_text)

    if job == "upload":
        return _upload_progress(log_text)

    return _result(None, "", "starting")


def _span(start: float, end: float, done: int, total: int) -> float:
    return start + (end - start) * (done / max(total, 1))


def _ratio(done: int, total: int) -> float:
    return done / max(total, 1)


def _pct(done: int, total: int) -> int:
    return 100 * done // max(total, 1)


def _train_progress(log_text: str) -> dict:
    current = _result(None, "Starting", "starting")

    failed = False
    error_line = ""

    epoch = epochs = 0
    training_started = False
    evaluations = 0

    for raw in log_text.splitlines():
        line = raw.rstrip()

        if not line:
            continue

        if "Traceback (most recent call last)" in line:
            failed = True
            continue

        if line.startswith("Error:"):
            failed = True
            error_line = line
            continue

        if "Roboflow is generating version" in line:
            current = _result(
                GENERATE_AT, "Generating a dataset version in Roboflow",
                "generate",
            )

        elif "Downloading Roboflow version" in line:
            current = _result(
                DOWNLOAD_FROM, "Downloading the dataset from Roboflow",
                "download", 0.0,
            )

        elif match := DOWNLOAD_LINE.search(line):
            done, total = int(match.group(1)), int(match.group(2))
            current = _result(
                _span(DOWNLOAD_FROM, DOWNLOAD_TO, done, total),
                f"Downloading the dataset {_pct(done, total)}%",
                "download", _ratio(done, total),
            )

        elif match := EXTRACT_LINE.search(line):
            done, total = int(match.group(1)), int(match.group(2))
            current = _result(
                _span(EXTRACT_FROM, EXTRACT_TO, done, total),
                f"Extracting the dataset {_pct(done, total)}%",
                "extract", _ratio(done, total),
            )

        elif "Aligned dataset class order" in line:
            current = _result(
                TRAIN_START, "Aligning class order to the deployed model",
                "prepare",
            )

        elif match := SCAN_LINE.match(line):
            split = "training" if match.group(1) == "train" else "validation"
            done, total = int(match.group(2)), int(match.group(3))
            current = _result(
                TRAIN_START,
                f"Preparing {split} images {done}/{total}",
                "prepare", _ratio(done, total),
            )

        elif "Transferred" in line and "pretrained weights" in line:
            current = _result(TRAIN_START, "Loading the model", "prepare")

        elif match := re.match(r"Starting training for (\d+) epochs", line):
            epochs = int(match.group(1))
            training_started = True
            current = _result(
                TRAIN_START, f"Starting training ({epochs} epochs)",
                "train", 0.0, 1, epochs, 0, 0,
            )

        elif match := EPOCH_LINE.match(line):
            epoch, epochs, batch, batches = (int(g) for g in match.groups())
            training_started = True

            done = (epoch - 1 + _ratio(batch, batches)) / max(epochs, 1)
            current = _result(
                TRAIN_START + TRAIN_SPAN * done,
                f"Epoch {epoch}/{epochs}, batch {batch}/{batches}",
                "train", _ratio(batch, batches), epoch, epochs, batch, batches,
            )

        elif (match := VALIDATION_LINE.match(line)) and epochs:
            done, total = int(match.group(1)), int(match.group(2))
            current = _result(
                TRAIN_START + TRAIN_SPAN * (epoch / max(epochs, 1)),
                f"Validating epoch {epoch}/{epochs}",
                "validate", _ratio(done, total), epoch, epochs, done, total,
            )

        elif (
            training_started
            and not epoch
            and (match := BARE_BAR_LINE.match(line))
        ):
            batch, batches = int(match.group(1)), int(match.group(2))
            current = _result(
                TRAIN_START + TRAIN_SPAN * _ratio(batch, batches) / max(epochs, 1),
                f"Epoch 1/{epochs}, batch {batch}/{batches}",
                "train", _ratio(batch, batches), 1, epochs, batch, batches,
            )

        elif "Evaluating" in line and " split" in line:
            evaluations += 1
            step = min(evaluations, 2)
            current = _result(
                COMPARE_FIRST if step == 1 else COMPARE_SECOND,
                f"Comparing with the deployed model ({step}/2)",
                "compare", (step - 1) / 2, epoch, epochs,
            )

        elif line.startswith("Installed ->"):
            current = _result(
                1.0, "Finished: new weights installed", "done", 1.0, epoch, epochs,
            )

        elif line.startswith("Not installed"):
            current = _result(
                1.0, "Finished: new model not better, deployed weights kept",
                "done", 1.0, epoch, epochs,
            )

        elif line.startswith("Weights not installed"):
            current = _result(
                1.0, "Finished: training done, weights not installed",
                "done", 1.0, epoch, epochs,
            )

    if failed:
        current["phase"] = "failed"
        current["text"] = "Failed" + (
            f": {error_line[len('Error:'):].strip()}" if error_line else ""
        )

    return current


def _upload_progress(log_text: str) -> dict:
    current = _result(None, "Starting", "starting")

    failed = False
    error_line = ""

    for raw in log_text.splitlines():
        line = raw.rstrip()

        if not line:
            continue

        if "Traceback (most recent call last)" in line:
            failed = True
            continue

        if line.startswith("Error:"):
            failed = True
            error_line = line
            continue

        if match := UPLOAD_LINE.search(line):
            index, total = int(match.group(1)), int(match.group(2))
            fraction = (index - 1) / max(total, 1)
            current = _result(
                fraction, f"Uploading crop {index} of {total}", "upload", fraction,
            )

        elif "Roboflow upload:" in line and "uploaded" in line:
            current = _result(
                1.0, "Finished: " + line.split("Roboflow upload:", 1)[1].strip(),
                "done", 1.0,
            )

        elif "nothing new" in line:
            current = _result(1.0, "Nothing new to upload", "done", 1.0)

        elif "crops pending upload" in line:
            current = _result(1.0, line.strip(), "done", 1.0)

    if failed:
        current["phase"] = "failed"
        current["text"] = "Failed" + (
            f": {error_line[len('Error:'):].strip()}" if error_line else ""
        )

    return current
