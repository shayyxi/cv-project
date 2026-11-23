"""
What the retraining page lists besides the latest comparison: the
training runs on disk, the deployed weights with their backups, and
the per-camera SAHI settings in vision_config.yaml. File reading and
one yaml write; training and installing stay in train.py.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from app.retraining.config import TRAINING_DIR
from app.retraining.train import VISION_CONFIG_PATH, ppe_weights_path

BACKUP_STAMP = "%Y%m%dT%H%M%S"  # as install_weights names its backups
_BACKUP_RE = re.compile(r"_backup_(\d{8}T\d{6})$")


# --- training runs ---------------------------------------------------------


@dataclass
class TrainingRun:
    name: str
    path: Path
    started: datetime
    epochs: int | None  # rows of ultralytics' results.csv
    comparison: dict | None  # the run's comparison.json, when it was compared
    has_weights: bool

    @property
    def map5095(self) -> tuple[float | None, float | None]:
        """(new, deployed) mAP50-95 from the comparison, or (None, None)."""

        if not self.comparison:
            return None, None
        new = (self.comparison.get("new") or {}).get("mAP50-95")
        deployed = (self.comparison.get("deployed") or {}).get("mAP50-95")
        return new, deployed


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _count_epochs(results: Path) -> int | None:
    if not results.is_file():
        return None
    try:
        with results.open(encoding="utf-8") as handle:
            return max(0, sum(1 for line in handle if line.strip()) - 1)
    except OSError:
        return None


def list_runs(training_dir: Path | str | None = None) -> list[TrainingRun]:
    """Every run folder under data/analytics/training/, newest first."""

    root = Path(training_dir) if training_dir else TRAINING_DIR

    if not root.is_dir():
        return []

    runs = []

    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        comparison = _read_json(folder / "comparison.json")
        started = None

        if comparison and comparison.get("timestamp"):
            try:
                started = datetime.strptime(comparison["timestamp"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                started = None

        if started is None:
            started = datetime.fromtimestamp(folder.stat().st_mtime)

        runs.append(TrainingRun(
            name=folder.name,
            path=folder,
            started=started,
            epochs=_count_epochs(folder / "results.csv"),
            comparison=comparison,
            has_weights=(folder / "weights" / "best.pt").is_file(),
        ))

    runs.sort(key=lambda run: run.started, reverse=True)
    return runs


# --- deployed weights and their backups --------------------------------------


@dataclass
class WeightsFile:
    path: Path
    size_bytes: int
    modified: datetime

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1e6


def _weights_file(path: Path, when: datetime | None = None) -> WeightsFile:
    stat = path.stat()
    return WeightsFile(path, stat.st_size, when or datetime.fromtimestamp(stat.st_mtime))


def deployed_weights(target: Path | str | None = None) -> WeightsFile | None:
    """The engine's PPE weights; its mtime is the install time."""

    target = Path(target) if target else ppe_weights_path()
    return _weights_file(target) if target.is_file() else None


def list_backups(target: Path | str | None = None) -> list[WeightsFile]:
    """
    The copies install_weights kept next to the weights
    (<stem>_backup_<stamp><suffix>), newest first.
    """

    target = Path(target) if target else ppe_weights_path()
    backups = []

    for path in target.parent.glob(f"{target.stem}_backup_*{target.suffix}"):
        match = _BACKUP_RE.search(path.stem)
        when = datetime.strptime(match.group(1), BACKUP_STAMP) if match else None
        backups.append(_weights_file(path, when))

    backups.sort(key=lambda item: item.modified, reverse=True)
    return backups


def file_sha256(path: Path | str, prefix: int = 12) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()[:prefix]


# --- per-camera SAHI settings (vision_config.yaml: sahi.default, sahi.cameras)


# key, lowest, highest, step. An int step means an integer field.
SAHI_FIELDS = (
    ("target_size", 320, 2400, 20),
    ("overlap_width", 0.0, 0.5, 0.05),
    ("overlap_height", 0.0, 0.5, 0.05),
    ("overlap", 0.0, 0.5, 0.05),
    ("merge_iou", 0.0, 1.0, 0.05),
    ("second_stage_merge_ios", 0.0, 1.0, 0.05),
    ("person_confidence", 0.0, 1.0, 0.01),
)
SAHI_FLAG = "second_stage_merge_enabled"
SAHI_LABELS = {
    "target_size": "Tile size (pixels)",
    "overlap_width": "Tile overlap, width",
    "overlap_height": "Tile overlap, height",
    "overlap": "Tile overlap",
    "merge_iou": "Merge threshold",
    "second_stage_merge_ios": "Second merge threshold",
    "person_confidence": "Person confidence",
}
# Used for a key the file's sahi.default leaves out.
DEFAULT_SAHI = {
    "target_size": 640,
    "overlap_width": 0.2,
    "overlap_height": 0.2,
    "overlap": 0.2,
    "merge_iou": 0.45,
    "second_stage_merge_ios": 0.7,
    "person_confidence": 0.5,
    SAHI_FLAG: False,
}


def _read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_sahi_settings(path: Path | str | None = None) -> tuple[dict, dict[str, dict]]:
    """
    (defaults, cameras): the default block completed with DEFAULT_SAHI,
    and camera id -> its own settings as written. The default person
    confidence falls back to the person model's confidence.
    """

    path = Path(path) if path else VISION_CONFIG_PATH
    config = _read_config(path) if path.is_file() else {}
    sahi = config.get("sahi") or {}

    defaults = {**DEFAULT_SAHI, **(sahi.get("default") or {})}

    if "person_confidence" not in (sahi.get("default") or {}):
        person = ((config.get("models") or {}).get("person") or {})
        if person.get("confidence") is not None:
            defaults["person_confidence"] = float(person["confidence"])

    cameras = {str(camera): dict(values or {}) for camera, values in (sahi.get("cameras") or {}).items()}
    return defaults, cameras


def save_sahi_settings(camera_id: str, values: dict, path: Path | str | None = None) -> Path:
    """
    Write one camera's entry under sahi.cameras, leaving the rest of the
    file as it is. Every field of SAHI_FIELDS must be present and within
    its range. Returns the file's path.
    """

    path = Path(path) if path else VISION_CONFIG_PATH
    camera_id = str(camera_id).strip()

    if not camera_id:
        raise ValueError("Camera id is empty.")

    entry: dict = {}

    for key, low, high, step in SAHI_FIELDS:
        if key not in values:
            raise ValueError(f"Missing setting: {key}.")
        value = int(values[key]) if isinstance(step, int) else round(float(values[key]), 4)
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}, got {value}.")
        entry[key] = value

    entry[SAHI_FLAG] = bool(values.get(SAHI_FLAG, False))

    config = _read_config(path) if path.is_file() else {}
    sahi = config.setdefault("sahi", {})
    cameras = sahi.setdefault("cameras", {}) or {}
    cameras[camera_id] = entry
    sahi["cameras"] = cameras

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

    return path
