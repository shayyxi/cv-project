"""
Fine-tunes the PPE detector with ultralytics and installs the result
as the weights file the vision engine loads.
"""

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from app.retraining.config import TRAINING_DIR
from app.retraining.dataset import align_dataset_to_names, ordered_names

logger = logging.getLogger(__name__)

CV_DIR = Path(__file__).resolve().parents[1] / "processing" / "cv"

VISION_CONFIG_PATH = CV_DIR / "config" / "vision_config.yaml"

# Fine-tuning start when the deployed PPE weights are not on disk.
# ultralytics downloads it on first use.
FALLBACK_BASE_MODEL = "yolov8n.pt"


def load_vision_config() -> dict:
    return yaml.safe_load(
        VISION_CONFIG_PATH.read_text(encoding="utf-8")
    ) or {}


def ppe_weights_path(config: dict | None = None) -> Path:
    """
    The PPE weights file the engine loads: models.ppe.path in
    vision_config.yaml, relative paths resolved against the cv
    package like PPEVisionEngine does.
    """

    config = config or load_vision_config()

    path = Path(config["models"]["ppe"]["path"])

    return path if path.is_absolute() else (CV_DIR / path).resolve()


def ppe_class_names(config: dict | None = None) -> list[str]:
    """
    The PPE class names the engine expects the model to emit
    (classes.ppe in vision_config.yaml), lower-cased.
    """

    config = config or load_vision_config()

    return [
        str(name).lower()
        for name in config["classes"]["ppe"].values()
    ]


def default_base_model() -> Path | str:
    """
    The deployed PPE weights when present, so every retraining starts
    from what is in production; else a pretrained YOLOv8n.
    """

    weights = ppe_weights_path()

    return weights if weights.exists() else FALLBACK_BASE_MODEL


@dataclass
class TrainingResult:
    best: Path
    last: Path
    save_dir: Path
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, float]:
        """
        The headline detection metrics, when ultralytics reported them.
        """

        wanted = {
            "precision": "metrics/precision(B)",
            "recall": "metrics/recall(B)",
            "mAP50": "metrics/mAP50(B)",
            "mAP50-95": "metrics/mAP50-95(B)",
        }

        return {
            label: round(self.metrics[key], 4)
            for label, key in wanted.items()
            if key in self.metrics
        }


def train_ppe_model(
    data_yaml: Path | str,
    base_model: Path | str | None = None,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    project_dir: Path | str | None = None,
    run_name: str | None = None,
    patience: int = 20,
    workers: int = 4,
    **overrides,
) -> TrainingResult:
    """
    Train with ultralytics and return where best.pt / last.pt landed.

    Runs go to data/analytics/training/<run_name>/. Extra keyword
    arguments are passed straight to ultralytics' train() (for
    example lr0=0.001 or freeze=10).
    """

    # Heavy import, kept off the module import path.
    from ultralytics import YOLO

    base_model = base_model or default_base_model()

    project_dir = Path(project_dir) if project_dir else TRAINING_DIR

    run_name = run_name or datetime.now().strftime("ppe-%Y%m%d-%H%M%S")

    project_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training PPE model from %s on %s "
        "(epochs=%d, imgsz=%d, batch=%d, device=%s)",
        base_model,
        data_yaml,
        epochs,
        imgsz,
        batch,
        device or "auto",
    )

    model = YOLO(str(base_model))

    # Fine-tuning keeps the pretrained head only if the label indices
    # mean the same classes; reorder the dataset to the model's order.
    data_yaml = align_dataset_to_names(data_yaml, ordered_names(model.names))

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
        "patience": patience,
        "workers": workers,
        "pretrained": True,
        "verbose": True,
    }

    if device is not None:
        train_kwargs["device"] = device

    train_kwargs.update(overrides)

    metrics = model.train(**train_kwargs)

    trainer = model.trainer

    result = TrainingResult(
        best=Path(trainer.best),
        last=Path(trainer.last),
        save_dir=Path(trainer.save_dir),
        metrics=_metrics_dict(metrics),
    )

    logger.info(
        "Training finished: %s %s", result.best, result.summary
    )

    return result


def _metrics_dict(metrics) -> dict[str, float]:
    results = getattr(metrics, "results_dict", None)

    if not isinstance(results, dict):
        return {}

    out: dict[str, float] = {}

    for key, value in results.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    return out


def install_weights(
    best: Path | str,
    target: Path | str | None = None,
) -> tuple[Path, Path | None]:
    """
    Copy best.pt over the engine's PPE weights, keeping a timestamped
    backup of the previous file next to it. Returns (target, backup).
    The pipeline loads the new weights on its next start.
    """

    best = Path(best)

    if not best.is_file():
        raise FileNotFoundError(f"No weights file at {best}")

    target = Path(target) if target else ppe_weights_path()

    target.parent.mkdir(parents=True, exist_ok=True)

    backup = None

    if target.exists():
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        backup = target.with_name(
            f"{target.stem}_backup_{stamp}{target.suffix}"
        )

        shutil.copy2(target, backup)

        logger.info("Backed up %s -> %s", target, backup)

    shutil.copy2(best, target)

    logger.info("Installed %s -> %s", best, target)

    return target, backup


# --- Install gate: is the new model at least as good as the deployed one? ---

# A per-class mAP50-95 drop larger than this is reported as a regression
# (informational; the gate itself looks at the overall mAP50-95).
REGRESSION_TOLERANCE = 0.05

HEADLINE_METRICS = ("precision", "recall", "mAP50", "mAP50-95")


def comparison_split(data_yaml: Path | str) -> str:
    """
    The split to compare models on: "test" when the dataset has one,
    else "val". The valid split picked the new model's best epoch, so
    it flatters the new model; test is the honest choice when present.
    """

    content = yaml.safe_load(
        Path(data_yaml).read_text(encoding="utf-8")
    ) or {}

    return "test" if content.get("test") else "val"


def metrics_summary(metrics) -> dict:
    """
    {"precision", "recall", "mAP50", "mAP50-95",
     "per_class": {class name: mAP50-95}} from ultralytics DetMetrics.
    """

    results = _metrics_dict(metrics)

    summary: dict = {
        "precision": results.get("metrics/precision(B)", 0.0),
        "recall": results.get("metrics/recall(B)", 0.0),
        "mAP50": results.get("metrics/mAP50(B)", 0.0),
        "mAP50-95": results.get("metrics/mAP50-95(B)", 0.0),
        "per_class": {},
    }

    names = getattr(metrics, "names", None) or {}
    maps = getattr(metrics, "maps", None)

    if maps is not None:
        for index, value in enumerate(maps):
            label = (
                str(names.get(index, index))
                if isinstance(names, dict)
                else str(index)
            )
            summary["per_class"][label] = float(value)

    return summary


def evaluate_model(
    weights: Path | str,
    data_yaml: Path | str,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    project_dir: Path | str | None = None,
    name: str | None = None,
) -> dict:
    """
    Validate a weights file on one split of a dataset. Returns the
    metrics_summary() dict. No plots are written.
    """

    from ultralytics import YOLO

    project_dir = Path(project_dir) if project_dir else TRAINING_DIR

    name = name or f"eval-{Path(weights).stem}"

    model = YOLO(str(weights))

    # Score each model against labels in its own class order.
    data_yaml = align_dataset_to_names(data_yaml, ordered_names(model.names))

    kwargs = {
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(project_dir),
        "name": name,
        "exist_ok": True,
        "plots": False,
        "verbose": False,
    }

    if device is not None:
        kwargs["device"] = device

    logger.info("Evaluating %s on the %s split", weights, split)

    metrics = model.val(**kwargs)

    return metrics_summary(metrics)


@dataclass
class ModelComparison:
    """Deployed ("old") versus freshly trained ("new") on one split."""

    split: str
    old: dict
    new: dict

    @property
    def better(self) -> bool:
        """New model is at least as good on overall mAP50-95."""

        return self.new["mAP50-95"] >= self.old["mAP50-95"]

    @property
    def regressed_classes(self) -> list[str]:
        """
        Classes whose mAP50-95 dropped by more than
        REGRESSION_TOLERANCE. Reported, not blocking.
        """

        old_per_class = self.old.get("per_class", {})
        new_per_class = self.new.get("per_class", {})

        return sorted(
            name
            for name, old_value in old_per_class.items()
            if name in new_per_class
            and old_value - new_per_class[name] > REGRESSION_TOLERANCE
        )

    def rows(self) -> list[tuple[str, float, float]]:
        """(label, old, new) for the headline metrics, then per class."""

        rows = [
            (key, float(self.old[key]), float(self.new[key]))
            for key in HEADLINE_METRICS
        ]

        old_per_class = self.old.get("per_class", {})
        new_per_class = self.new.get("per_class", {})

        for name in sorted(set(old_per_class) | set(new_per_class)):
            rows.append(
                (
                    f"mAP50-95 {name}",
                    float(old_per_class.get(name, 0.0)),
                    float(new_per_class.get(name, 0.0)),
                )
            )

        return rows


def compare_models(
    old_weights: Path | str,
    new_weights: Path | str,
    data_yaml: Path | str,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    project_dir: Path | str | None = None,
    run_name: str | None = None,
) -> ModelComparison:
    """
    Evaluate both weights files on the same split of the same dataset
    (see comparison_split). Validation output goes under
    <project_dir>/compare-<run_name>/{old,new}/.
    """

    split = comparison_split(data_yaml)

    project_dir = Path(project_dir) if project_dir else TRAINING_DIR

    base = f"compare-{run_name}" if run_name else "compare"

    logger.info(
        "Comparing deployed %s with new %s on the %s split",
        old_weights,
        new_weights,
        split,
    )

    old = evaluate_model(
        old_weights, data_yaml, split, imgsz, batch, device,
        project_dir, f"{base}/old",
    )

    new = evaluate_model(
        new_weights, data_yaml, split, imgsz, batch, device,
        project_dir, f"{base}/new",
    )

    comparison = ModelComparison(split=split, old=old, new=new)

    logger.info(
        "Comparison on %s: deployed mAP50-95 %.4f vs new %.4f -> %s",
        split,
        old["mAP50-95"],
        new["mAP50-95"],
        "new is at least as good" if comparison.better else "new is worse",
    )

    return comparison


# --- Comparison record for the Streamlit page ------------------------------

LATEST_COMPARISON_FILE = TRAINING_DIR / "latest_comparison.json"


def comparison_to_dict(
    comparison: ModelComparison,
    installed: bool,
    run_name: str | None = None,
    new_weights: Path | str | None = None,
    deployed_weights: Path | str | None = None,
) -> dict:
    """
    JSON-ready record of a comparison: headline metrics for both
    models, mAP50-95 per class with the delta, the outcome, and the
    two weights files so the page can install the new one later.
    """

    old_per_class = comparison.old.get("per_class", {})
    new_per_class = comparison.new.get("per_class", {})

    per_class = {}

    for name in sorted(set(old_per_class) | set(new_per_class)):
        old_value = float(old_per_class.get(name, 0.0))
        new_value = float(new_per_class.get(name, 0.0))

        per_class[name] = {
            "deployed": round(old_value, 4),
            "new": round(new_value, 4),
            "delta": round(new_value - old_value, 4),
        }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "run": run_name,
        "timestamp": now,
        "split": comparison.split,
        "better": comparison.better,
        "installed": installed,
        "installed_at": now if installed else None,
        "new_weights": str(Path(new_weights).resolve()) if new_weights else None,
        "deployed_weights": (
            str(Path(deployed_weights).resolve()) if deployed_weights else None
        ),
        "regressed_classes": comparison.regressed_classes,
        "deployed": {
            key: round(float(comparison.old[key]), 4)
            for key in HEADLINE_METRICS
        },
        "new": {
            key: round(float(comparison.new[key]), 4)
            for key in HEADLINE_METRICS
        },
        "per_class": per_class,
    }


def write_comparison(
    comparison: ModelComparison,
    installed: bool,
    run_name: str | None = None,
    run_dir: Path | str | None = None,
    latest_path: Path | str | None = None,
    new_weights: Path | str | None = None,
    deployed_weights: Path | str | None = None,
) -> Path:
    """
    Write the record to <run_dir>/comparison.json (when a run folder
    is given) and to the "latest" file the page reads. Returns the
    latest file's path.
    """

    import json

    record = comparison_to_dict(
        comparison, installed, run_name, new_weights, deployed_weights
    )
    payload = json.dumps(record, indent=2)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "comparison.json").write_text(payload, encoding="utf-8")

    latest = Path(latest_path) if latest_path else LATEST_COMPARISON_FILE
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(payload, encoding="utf-8")

    return latest


def mark_installed(latest_path: Path | str | None = None) -> dict:
    """
    Flag the latest comparison record as installed (after the page's
    replace button copied the weights). The run folder's copy, found
    next to the recorded new_weights, is updated too. Returns the
    updated record.
    """

    import json

    latest = Path(latest_path) if latest_path else LATEST_COMPARISON_FILE

    record = json.loads(latest.read_text(encoding="utf-8"))
    record["installed"] = True
    record["installed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = json.dumps(record, indent=2)
    latest.write_text(payload, encoding="utf-8")

    new_weights = record.get("new_weights")

    if new_weights:
        weights = Path(new_weights)

        if weights.parent.name == "weights":
            run_copy = weights.parent.parent / "comparison.json"

            if run_copy.exists():
                run_copy.write_text(payload, encoding="utf-8")

    return record
