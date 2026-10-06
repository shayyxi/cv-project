"""
Pulls a dataset version from Roboflow in YOLO format and makes its
data.yaml ready for ultralytics.
"""

import logging
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path

import yaml

from app.retraining.config import DATASETS_DIR, connect_project

logger = logging.getLogger(__name__)

EXPORT_FORMAT = "yolov8"

# Keep the crops at their native size: ultralytics letterboxes them to
# imgsz at train time, exactly as the engine does at inference. No
# Roboflow-side augmentation either; ultralytics applies its own.
DEFAULT_VERSION_SETTINGS = {
    "preprocessing": {"auto-orient": True},
    "augmentation": {},
}

# data.yaml key -> folder Roboflow's YOLO export uses for it.
SPLIT_DIRS = {
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
}


class RoboflowDatasetService:
    """
    Resolves, optionally generates, and downloads a Roboflow dataset
    version into data/analytics/datasets/<project>-v<N>/.
    """

    def __init__(
        self,
        project_factory: Callable[[], object] | None = None,
        datasets_dir: Path | str | None = None,
    ) -> None:
        self._project_factory = project_factory or connect_project

        self._datasets_dir = (
            Path(datasets_dir) if datasets_dir else DATASETS_DIR
        ) 

    def download(
        self,
        version_number: int | None = None,
        generate: bool = False,
        version_settings: dict | None = None,
        redownload: bool = False,
    ) -> Path:
        """
        Download a dataset version in YOLO format and return the path
        to its prepared data.yaml.

        generate=True first generates a new version from the project's
        currently annotated images and waits for it. Otherwise
        version_number, or the latest existing version, is used. An
        already downloaded version is reused unless redownload=True.
        """

        project = self._project_factory()

        if generate:
            version_number = self.generate_version(
                project, version_settings
            )
        elif version_number is None:
            version_number = self.latest_version_number(project)

            if version_number is None:
                raise RuntimeError(
                    "The Roboflow project has no dataset versions yet; "
                    "generate one in Roboflow or pass --generate."
                )

        version = project.version(version_number)

        location = (
            self._datasets_dir
            / f"{self._slug(project)}-v{version_number}"
        )

        self._datasets_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading Roboflow version %s as %s -> %s",
            version_number,
            EXPORT_FORMAT,
            location,
        )

        dataset = version.download(
            EXPORT_FORMAT,
            location=str(location),
            overwrite=redownload,
        )

        data_yaml = Path(dataset.location) / "data.yaml"

        if not data_yaml.exists():
            raise RuntimeError(
                f"Downloaded dataset has no data.yaml: {data_yaml}"
            )

        return prepare_data_yaml(data_yaml)

    @staticmethod
    def latest_version_number(project) -> int | None:
        numbers = []

        for info in project.get_version_information():
            number = _version_number(info.get("id"))

            if number is not None:
                numbers.append(number)

        return max(numbers) if numbers else None

    @staticmethod
    def generate_version(
        project,
        version_settings: dict | None = None,
    ) -> int:
        """
        Ask Roboflow to generate a new version; returns its number.
        The API needs both "preprocessing" and "augmentation" keys,
        so missing ones default to empty.
        """

        merged = {**DEFAULT_VERSION_SETTINGS, **(version_settings or {})}

        for key in ("preprocessing", "augmentation"):
            merged.setdefault(key, {})

        number = int(project.generate_version(settings=merged))

        logger.info("Roboflow is generating version %s", number)

        return number

    @staticmethod
    def _slug(project) -> str:
        raw = str(
            getattr(project, "name", None)
            or getattr(project, "id", None)
            or "dataset"
        )

        raw = raw.rsplit("/", 1)[-1]

        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")

        return slug or "dataset"


def _version_number(version_id) -> int | None:
    """"workspace/project/3" -> 3."""

    if version_id is None:
        return None

    try:
        return int(os.path.basename(str(version_id)))
    except ValueError:
        return None


def prepare_data_yaml(data_yaml: Path | str) -> Path:
    """
    Rewrite data.yaml in place with absolute split paths.

    Roboflow writes "../train/images"-style paths relative to the
    yaml's parent, which only resolve when the dataset sits in one
    specific place. A missing test split is dropped. A missing valid
    split falls back to train, since ultralytics needs one to
    validate on.
    """

    data_yaml = Path(data_yaml).resolve()

    root = data_yaml.parent

    content = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}

    resolved = {
        key: _resolve_split_dir(root, content.get(key), default)
        for key, default in SPLIT_DIRS.items()
    }

    if resolved["train"] is None:
        raise RuntimeError(
            f"No train/images folder found next to {data_yaml}"
        )

    if resolved["val"] is None:
        logger.warning(
            "%s has no valid split - validating on the train split. "
            "Set ROBOFLOW_VALID_SPLIT_PERCENT > 0 so future uploads "
            "give honest metrics.",
            data_yaml,
        )
        resolved["val"] = resolved["train"]

    content["path"] = str(root)
    content["train"] = str(resolved["train"])
    content["val"] = str(resolved["val"])

    if resolved["test"] is not None:
        content["test"] = str(resolved["test"])
    else:
        content.pop("test", None)

    data_yaml.write_text(
        yaml.safe_dump(content, sort_keys=False),
        encoding="utf-8",
    )

    return data_yaml


def _resolve_split_dir(
    root: Path,
    value: str | None,
    default_rel: str,
) -> Path | None:
    candidates: list[Path] = []

    if value:
        path = Path(str(value))

        candidates.append(path if path.is_absolute() else root / path)

        # "../train/images" written relative to the yaml really sits
        # next to it.
        parts = [part for part in path.parts if part not in ("..", ".")]

        if parts and not path.is_absolute():
            candidates.append(root.joinpath(*parts))

    candidates.append(root / default_rel)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    return None


def dataset_class_names(data_yaml: Path | str) -> list[str]:
    """
    The class names of a YOLO data.yaml, in index order, whether they
    are written as a list or as an {index: name} mapping.
    """

    content = yaml.safe_load(
        Path(data_yaml).read_text(encoding="utf-8")
    ) or {}

    names = content.get("names") or []

    if isinstance(names, dict):
        names = [
            name
            for _, name in sorted(
                names.items(), key=lambda item: int(item[0])
            )
        ]

    return [str(name) for name in names]


def check_class_names(
    names: Iterable[str],
    expected: Iterable[str],
) -> tuple[list[str], list[str]]:
    """
    (missing, extra): expected class names absent from the dataset,
    and dataset classes the engine would ignore. Case-insensitive,
    like the engine's reading of model class names.
    """

    have = {str(name).strip().lower() for name in names}
    want = {str(name).strip().lower() for name in expected}

    return sorted(want - have), sorted(have - want)


# --- Class order alignment ---------------------------------------------------
#
# Roboflow exports classes alphabetically (boots, helmet, vest). The
# deployed PPE model was trained with helmet, vest, boots. YOLO label
# files hold class *indices*, so evaluating or fine-tuning a model on a
# dataset with another order scores and trains every box against the
# wrong class. The helpers below rewrite a dataset into the model's
# order, by name, so both sides agree.


def ordered_names(names) -> list[str]:
    """
    Class names in index order, lower-cased, from either a YOLO
    data.yaml list or an ultralytics model.names {index: name} dict.
    """

    if isinstance(names, dict):
        return [
            str(names[index]).strip().lower()
            for index in sorted(names, key=int)
        ]

    return [str(name).strip().lower() for name in names]


def align_dataset_to_names(
    data_yaml: Path | str,
    target_names,
) -> Path:
    """
    Return a data.yaml whose class order matches target_names.

    Same order already: the given data.yaml. Same classes in another
    order: a sibling folder "aligned-<order>" holding hard links to
    the images (copies when linking fails) and label files with the
    indices remapped; reused on later calls. Different classes
    altogether: the given data.yaml, unchanged, with a log line.
    """

    data_yaml = Path(data_yaml).resolve()

    content = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}

    current = ordered_names(content.get("names") or [])
    target = ordered_names(target_names)

    if current == target:
        return data_yaml

    if sorted(current) != sorted(target):
        logger.info(
            "Dataset classes %s and model classes %s differ; keeping "
            "the dataset's class order.",
            current,
            target,
        )
        return data_yaml

    root = data_yaml.parent

    aligned_root = root / ("aligned-" + "-".join(target))
    aligned_yaml = aligned_root / "data.yaml"

    if aligned_yaml.exists():
        return aligned_yaml

    mapping = {
        old_index: target.index(name)
        for old_index, name in enumerate(current)
    }

    # Keep the dataset's own spelling of each name, just reordered.
    spelling = {
        str(name).strip().lower(): str(name)
        for name in content.get("names") or []
    }

    aligned = dict(content)

    for key in ("train", "val", "test"):
        source = content.get(key)

        if not source:
            continue

        images_dir = Path(str(source))

        if not images_dir.is_absolute():
            images_dir = (root / images_dir).resolve()

        if not images_dir.is_dir():
            continue

        split_dir = images_dir.parent

        target_images = aligned_root / split_dir.name / "images"
        target_labels = aligned_root / split_dir.name / "labels"

        _link_or_copy_files(images_dir, target_images)
        _remap_label_files(split_dir / "labels", target_labels, mapping)

        aligned[key] = str(target_images)

    aligned["path"] = str(aligned_root)
    aligned["names"] = [spelling[name] for name in target]
    aligned["nc"] = len(target)

    aligned_root.mkdir(parents=True, exist_ok=True)

    # Written last, so an interrupted run is redone next time.
    aligned_yaml.write_text(
        yaml.safe_dump(aligned, sort_keys=False),
        encoding="utf-8",
    )

    logger.info(
        "Aligned dataset class order %s -> %s at %s",
        current,
        target,
        aligned_yaml,
    )

    return aligned_yaml


def _link_or_copy_files(source_dir: Path, target_dir: Path) -> None:
    import os
    import shutil

    target_dir.mkdir(parents=True, exist_ok=True)

    for path in source_dir.iterdir():
        if not path.is_file() or path.suffix == ".cache":
            continue

        target = target_dir / path.name

        if target.exists():
            continue

        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def _remap_label_files(
    source_dir: Path,
    target_dir: Path,
    mapping: dict[int, int],
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.is_dir():
        return

    for path in source_dir.glob("*.txt"):
        lines = []

        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()

            if not parts:
                continue

            try:
                parts[0] = str(mapping[int(parts[0])])
            except (ValueError, KeyError):
                pass

            lines.append(" ".join(parts))

        (target_dir / path.name).write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
