from pathlib import Path

import pytest
import yaml

from app.retraining.dataset import (
    RoboflowDatasetService,
    check_class_names,
    dataset_class_names,
    prepare_data_yaml,
)


def _make_dataset(
    root: Path,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    names: tuple[str, ...] = ("boots", "helmet", "vest"),
) -> Path:
    """
    A folder shaped like Roboflow's YOLOv8 export, including its
    "../train/images" paths written relative to the yaml's parent.
    """

    for split in splits:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)

    content = {
        "names": list(names),
        "nc": len(names),
        "roboflow": {"version": 3, "project": "ppe-crops"},
        "test": "../test/images",
        "train": "../train/images",
        "val": "../valid/images",
    }

    root.mkdir(parents=True, exist_ok=True)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(content), encoding="utf-8")

    return data_yaml


def test_prepare_data_yaml_resolves_roboflow_relative_paths(tmp_path) -> None:
    root = tmp_path / "ds"
    data_yaml = _make_dataset(root)

    prepared = prepare_data_yaml(data_yaml)

    assert prepared == data_yaml.resolve()

    content = yaml.safe_load(prepared.read_text(encoding="utf-8"))

    assert Path(content["path"]) == root.resolve()
    assert Path(content["train"]) == (root / "train" / "images").resolve()
    assert Path(content["val"]) == (root / "valid" / "images").resolve()
    assert Path(content["test"]) == (root / "test" / "images").resolve()

    # Everything else is left alone.
    assert content["names"] == ["boots", "helmet", "vest"]
    assert content["nc"] == 3
    assert content["roboflow"]["version"] == 3


def test_prepare_data_yaml_accepts_sdk_rewritten_paths(tmp_path) -> None:
    root = tmp_path / "ds"
    data_yaml = _make_dataset(root)

    content = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    content["train"] = "train/images"
    content["val"] = "valid/images"
    content["test"] = "test/images"
    data_yaml.write_text(yaml.safe_dump(content), encoding="utf-8")

    content = yaml.safe_load(
        prepare_data_yaml(data_yaml).read_text(encoding="utf-8")
    )

    assert Path(content["train"]) == (root / "train" / "images").resolve()
    assert Path(content["val"]) == (root / "valid" / "images").resolve()


def test_prepare_data_yaml_drops_missing_test_and_falls_back_to_train(
    tmp_path,
) -> None:
    data_yaml = _make_dataset(tmp_path / "ds", splits=("train",))

    content = yaml.safe_load(
        prepare_data_yaml(data_yaml).read_text(encoding="utf-8")
    )

    assert "test" not in content
    assert content["val"] == content["train"]


def test_prepare_data_yaml_requires_train_split(tmp_path) -> None:
    root = tmp_path / "ds"
    root.mkdir()

    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": ["helmet"], "train": "../train/images"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="train/images"):
        prepare_data_yaml(root / "data.yaml")


def test_dataset_class_names_from_list_and_mapping(tmp_path) -> None:
    as_list = tmp_path / "list.yaml"
    as_list.write_text(
        yaml.safe_dump({"names": ["helmet", "vest"]}), encoding="utf-8"
    )

    as_dict = tmp_path / "dict.yaml"
    as_dict.write_text(
        yaml.safe_dump({"names": {1: "vest", 0: "helmet", 2: "boots"}}),
        encoding="utf-8",
    )

    assert dataset_class_names(as_list) == ["helmet", "vest"]
    assert dataset_class_names(as_dict) == ["helmet", "vest", "boots"]


def test_check_class_names_is_case_insensitive() -> None:
    expected = ["helmet", "vest", "boots"]

    assert check_class_names(["Helmet", "vest", "gloves"], expected) == (
        ["boots"],
        ["gloves"],
    )
    assert check_class_names(["boots", "helmet", "vest"], expected) == (
        [],
        [],
    )


class FakeDataset:
    def __init__(self, location: str) -> None:
        self.location = location


class FakeVersion:
    def __init__(self, number: int) -> None:
        self.number = number
        self.download_calls: list[tuple] = []

    def download(self, model_format, location=None, overwrite=False):
        self.download_calls.append((model_format, location, overwrite))

        _make_dataset(Path(location))

        return FakeDataset(location)


class FakeProject:
    name = "PPE Crops"
    id = "ws/ppe-crops"

    def __init__(self, versions: tuple[int, ...] = (1, 2, 3)) -> None:
        self.version_ids = [f"ws/ppe-crops/{n}" for n in versions]
        self.generated: list[dict] = []
        self.versions_requested: list[int] = []
        self.last_version: FakeVersion | None = None

    def get_version_information(self):
        return [
            {"id": version_id, "images": 10}
            for version_id in self.version_ids
        ]

    def generate_version(self, settings):
        self.generated.append(settings)

        numbers = [
            int(version_id.rsplit("/", 1)[1])
            for version_id in self.version_ids
        ]

        new = (max(numbers) + 1) if numbers else 1

        self.version_ids.append(f"ws/ppe-crops/{new}")

        return new

    def version(self, number):
        self.versions_requested.append(number)
        self.last_version = FakeVersion(number)
        return self.last_version


def _service(tmp_path, project) -> RoboflowDatasetService:
    return RoboflowDatasetService(
        project_factory=lambda: project,
        datasets_dir=tmp_path / "datasets",
    )


def test_download_uses_latest_version_by_default(tmp_path) -> None:
    project = FakeProject()

    data_yaml = _service(tmp_path, project).download()

    assert project.versions_requested == [3]

    expected = tmp_path / "datasets" / "PPE-Crops-v3" / "data.yaml"
    assert data_yaml == expected.resolve()

    content = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert Path(content["train"]).is_dir()
    assert Path(content["val"]).is_dir()

    assert project.last_version.download_calls == [
        ("yolov8", str(tmp_path / "datasets" / "PPE-Crops-v3"), False),
    ]


def test_download_generate_creates_and_uses_a_new_version(tmp_path) -> None:
    project = FakeProject()

    data_yaml = _service(tmp_path, project).download(
        generate=True,
        version_settings={"augmentation": {"flip": {"horizontal": True}}},
    )

    assert project.versions_requested == [4]
    assert project.generated == [
        {
            "preprocessing": {"auto-orient": True},
            "augmentation": {"flip": {"horizontal": True}},
        }
    ]
    assert data_yaml.parent.name == "PPE-Crops-v4"


def test_download_explicit_version_with_redownload(tmp_path) -> None:
    project = FakeProject()

    _service(tmp_path, project).download(version_number=2, redownload=True)

    assert project.versions_requested == [2]
    assert project.last_version.download_calls == [
        ("yolov8", str(tmp_path / "datasets" / "PPE-Crops-v2"), True),
    ]


def test_download_without_any_version_needs_generate(tmp_path) -> None:
    project = FakeProject(versions=())

    with pytest.raises(RuntimeError, match="no dataset versions"):
        _service(tmp_path, project).download()

    assert project.versions_requested == []


# --- class order alignment ---------------------------------------------------

from app.retraining.dataset import (  # noqa: E402
    align_dataset_to_names,
    ordered_names,
)


def test_ordered_names_from_list_and_model_dict() -> None:
    assert ordered_names(["Boots", "helmet", "vest "]) == ["boots", "helmet", "vest"]
    assert ordered_names({2: "boots", 0: "helmet", 1: "vest"}) == [
        "helmet", "vest", "boots",
    ]


def _labelled_dataset(root: Path) -> Path:
    """Roboflow-order dataset (boots=0, helmet=1, vest=2) with labels."""

    data_yaml = _make_dataset(root, splits=("train", "valid"))
    prepare_data_yaml(data_yaml)

    (root / "train" / "images" / "a.jpg").write_bytes(b"img-a")
    (root / "train" / "images" / "b.jpg").write_bytes(b"img-b")   # background
    (root / "train" / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n2 0.8 0.8 0.1 0.1\n",
        encoding="utf-8",
    )
    (root / "train" / "labels" / "labels.cache").write_bytes(b"stale")

    (root / "valid" / "images" / "c.jpg").write_bytes(b"img-c")
    (root / "valid" / "labels" / "c.txt").write_text(
        "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    return data_yaml


def test_align_returns_same_yaml_when_order_matches(tmp_path) -> None:
    data_yaml = _labelled_dataset(tmp_path / "ds")

    assert align_dataset_to_names(data_yaml, ["boots", "helmet", "vest"]) == data_yaml
    assert align_dataset_to_names(data_yaml, {0: "Boots", 1: "Helmet", 2: "Vest"}) == data_yaml


def test_align_keeps_dataset_order_for_different_classes(tmp_path) -> None:
    data_yaml = _labelled_dataset(tmp_path / "ds")

    # COCO-style model: not the same class set, nothing to align.
    assert align_dataset_to_names(data_yaml, ["person", "bicycle", "car"]) == data_yaml


def test_align_rewrites_labels_into_model_order(tmp_path) -> None:
    root = tmp_path / "ds"
    data_yaml = _labelled_dataset(root)

    # The deployed model's order.
    aligned = align_dataset_to_names(data_yaml, {0: "helmet", 1: "vest", 2: "boots"})

    assert aligned == root / "aligned-helmet-vest-boots" / "data.yaml"

    content = yaml.safe_load(aligned.read_text(encoding="utf-8"))
    assert content["names"] == ["helmet", "vest", "boots"]
    assert content["nc"] == 3
    assert Path(content["path"]) == aligned.parent
    assert Path(content["train"]) == aligned.parent / "train" / "images"
    assert Path(content["val"]) == aligned.parent / "valid" / "images"
    assert "test" not in content

    # boots 0 -> 2, helmet 1 -> 0, vest 2 -> 1; coordinates untouched.
    train_labels = aligned.parent / "train" / "labels"
    assert (train_labels / "a.txt").read_text(encoding="utf-8") == (
        "2 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n1 0.8 0.8 0.1 0.1\n"
    )
    assert (aligned.parent / "valid" / "labels" / "c.txt").read_text(
        encoding="utf-8"
    ) == "0 0.5 0.5 0.2 0.2\n"

    # Images are present (linked or copied); the stale cache is not.
    train_images = aligned.parent / "train" / "images"
    assert (train_images / "a.jpg").read_bytes() == b"img-a"
    assert (train_images / "b.jpg").read_bytes() == b"img-b"
    assert not (train_labels / "labels.cache").exists()

    # The original dataset is untouched.
    assert (root / "train" / "labels" / "a.txt").read_text(encoding="utf-8").startswith("0 ")


def test_align_is_reused_on_second_call(tmp_path) -> None:
    data_yaml = _labelled_dataset(tmp_path / "ds")

    first = align_dataset_to_names(data_yaml, ["helmet", "vest", "boots"])
    (first.parent / "marker").write_text("kept", encoding="utf-8")

    second = align_dataset_to_names(data_yaml, ["helmet", "vest", "boots"])

    assert second == first
    assert (first.parent / "marker").exists()
