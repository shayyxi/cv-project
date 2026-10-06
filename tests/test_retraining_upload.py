import json
from pathlib import Path

from app.retraining.config import RoboflowSettings
from app.retraining.upload import RoboflowUploadService

# Named like real crops: <camera>_<frame ts>_p<person>_<ts>.jpg
CROP_A = "12846_20260901T100000_p1_20260901T100000.jpg"
CROP_B = "12875_20260901T100100_p2_20260901T100100.jpg"
CROP_C = "12990_20260901T100200_p3_20260901T100200.jpg"
CROP_D = "12846_20260902T090000_p4_20260902T090000.jpg"

ALL_CROPS = [CROP_A, CROP_B, CROP_C]


class FakeProject:
    def __init__(
        self,
        fail_for: set[str] | None = None,
        duplicates: set[str] | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.fail_for = fail_for or set()
        self.duplicates = duplicates or set()

    def single_upload(self, **kwargs):
        self.calls.append(kwargs)

        name = Path(kwargs["image_path"]).name

        if name in self.fail_for:
            raise RuntimeError(f"upload failed: {name}")

        if name in self.duplicates:
            return {"image": {"id": f"dup-{name}", "duplicate": True}}

        return {"image": {"id": f"id-{name}", "success": True}}

    def uploaded_names(self) -> list[str]:
        return [Path(call["image_path"]).name for call in self.calls]


def _write_queue(queue_dir: Path, with_coco: bool = True) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)

    for name in ALL_CROPS:
        (queue_dir / name).write_bytes(b"\xff\xd8fake-jpeg")

    # Must be ignored: not an image.
    (queue_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    if not with_coco:
        return

    coco = {
        "info": {"description": "test"},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "helmet", "supercategory": "ppe"},
            {"id": 2, "name": "vest", "supercategory": "ppe"},
            {"id": 3, "name": "boots", "supercategory": "ppe"},
        ],
        "images": [
            {"id": 1, "file_name": CROP_A, "width": 512, "height": 640},
            {"id": 2, "file_name": CROP_B, "width": 512, "height": 640},
            {"id": 3, "file_name": CROP_C, "width": 512, "height": 640},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 50, 40],
                "area": 2000,
                "iscrowd": 0,
                "score": 0.9,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [20, 100, 200, 300],
                "area": 60000,
                "iscrowd": 0,
                "score": 0.8,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 3,
                "bbox": [0, 500, 100, 100],
                "area": 10000,
                "iscrowd": 0,
                "score": 0.7,
            },
        ],
    }

    (queue_dir / "_annotations.coco.json").write_text(
        json.dumps(coco), encoding="utf-8"
    )


def _config(**overrides) -> RoboflowSettings:
    values = {
        "api_key": "key",
        "project": "ppe",
        "workspace": "ws",
        "upload_as_prediction": True,
        "valid_split_percent": 0,
        "batch_prefix": "label-queue",
    }
    values.update(overrides)

    return RoboflowSettings(**values)


def _service(
    queue_dir: Path,
    project: FakeProject,
    **config_overrides,
) -> RoboflowUploadService:
    return RoboflowUploadService(
        config=_config(**config_overrides),
        queue_dir=queue_dir,
        project_factory=lambda: project,
    )


def _manifest(queue_dir: Path) -> dict:
    return json.loads(
        (queue_dir / "_roboflow_uploads.json").read_text(encoding="utf-8")
    )


def test_uploads_every_new_crop_with_its_own_prelabels(tmp_path) -> None:
    queue = tmp_path / "label_queue"
    _write_queue(queue)
    project = FakeProject()

    result = _service(queue, project).upload_new_crops()

    assert result["pending"] == 3
    assert result["uploaded"] == 3
    assert result["duplicates"] == 0
    assert result["failed"] == 0
    assert result["batch"].startswith("label-queue-")

    assert project.uploaded_names() == ALL_CROPS
    assert all(call["is_prediction"] is True for call in project.calls)
    assert all(call["batch_name"] == result["batch"] for call in project.calls)
    assert all(call["split"] == "train" for call in project.calls)

    first = project.calls[0]
    assert first["tag_names"] == ["label-queue", "camera-12846"]
    assert first["annotation_path"]["name"] == "annotation.coco.json"

    document = json.loads(first["annotation_path"]["rawText"])
    assert [image["file_name"] for image in document["images"]] == [CROP_A]
    assert [a["category_id"] for a in document["annotations"]] == [1, 2]
    assert all(a["image_id"] == 1 for a in document["annotations"])
    assert "score" not in document["annotations"][0]
    assert [c["name"] for c in document["categories"]] == [
        "helmet", "vest", "boots",
    ]

    # Only crop B's box, not A's.
    second = json.loads(project.calls[1]["annotation_path"]["rawText"])
    assert [a["bbox"] for a in second["annotations"]] == [[0, 500, 100, 100]]
    assert project.calls[1]["tag_names"] == ["label-queue", "camera-12875"]

    # No PPE box -> uploaded unannotated.
    third = project.calls[2]
    assert third["annotation_path"] is None

    uploads = _manifest(queue)["projects"]["ws/ppe"]
    assert set(uploads) == set(ALL_CROPS)
    assert uploads[CROP_A]["image_id"] == f"id-{CROP_A}"
    assert uploads[CROP_A]["annotations"] == 2
    assert uploads[CROP_C]["annotations"] == 0
    assert uploads[CROP_A]["duplicate"] is False


def test_second_run_only_uploads_new_crops(tmp_path) -> None:
    queue = tmp_path / "label_queue"
    _write_queue(queue)
    project = FakeProject()
    service = _service(queue, project)

    service.upload_new_crops()

    result = service.upload_new_crops()
    assert result["pending"] == 0
    assert result["uploaded"] == 0
    assert len(project.calls) == 3

    (queue / CROP_D).write_bytes(b"\xff\xd8new")

    result = _service(queue, project).upload_new_crops()
    assert result["pending"] == 1
    assert result["uploaded"] == 1
    assert project.uploaded_names()[-1] == CROP_D
    assert CROP_D in _manifest(queue)["projects"]["ws/ppe"]


def test_failed_upload_is_retried_on_next_run(tmp_path) -> None:
    queue = tmp_path / "label_queue"
    _write_queue(queue)
    project = FakeProject(fail_for={CROP_B})

    result = _service(queue, project).upload_new_crops()

    assert result["uploaded"] == 2
    assert result["failed"] == 1
    assert CROP_B not in _manifest(queue)["projects"]["ws/ppe"]

    project.fail_for.clear()

    result = _service(queue, project).upload_new_crops()

    assert result["pending"] == 1
    assert result["uploaded"] == 1
    assert project.uploaded_names()[-1] == CROP_B
    assert CROP_B in _manifest(queue)["projects"]["ws/ppe"]


def test_duplicates_count_as_uploaded(tmp_path) -> None:
    queue = tmp_path / "label_queue"
    _write_queue(queue)
    project = FakeProject(duplicates={CROP_A})

    result = _service(queue, project).upload_new_crops()

    assert result["uploaded"] == 2
    assert result["duplicates"] == 1

    uploads = _manifest(queue)["projects"]["ws/ppe"]
    assert uploads[CROP_A]["duplicate"] is True
    assert uploads[CROP_A]["image_id"] == f"dup-{CROP_A}"


def test_limit_caps_one_run(tmp_path) -> None:
    queue = tmp_path / "label_queue"
    _write_queue(queue)
    project = FakeProject()

    result = _service(queue, project).upload_new_crops(limit=2)

    assert result["pending"] == 2
    assert result["uploaded"] == 2
    assert project.uploaded_names() == [CROP_A, CROP_B]


def test_split_assignment_is_deterministic(tmp_path) -> None:
    queue = tmp_path / "label_queue"
    _write_queue(queue)
    project = FakeProject()

    all_train = _service(queue, project, valid_split_percent=0)
    all_valid = _service(queue, project, valid_split_percent=100)
    mixed = _service(queue, project, valid_split_percent=20)

    assert {all_train.split_for(name) for name in ALL_CROPS} == {"train"}
    assert {all_valid.split_for(name) for name in ALL_CROPS} == {"valid"}

    for name in ALL_CROPS:
        assert mixed.split_for(name) in {"train", "valid"}
        assert mixed.split_for(name) == mixed.split_for(name)

    all_valid.upload_new_crops()
    assert all(call["split"] == "valid" for call in project.calls)


def test_camera_id_comes_from_the_file_name() -> None:
    assert RoboflowUploadService.camera_id_for(CROP_A) == "12846"
    assert RoboflowUploadService.camera_id_for("no-underscore.jpg") is None


def test_dry_run_neither_connects_nor_writes(tmp_path) -> None:
    queue = tmp_path / "label_queue"
    _write_queue(queue)

    def factory():
        raise AssertionError("must not connect on a dry run")

    service = RoboflowUploadService(
        config=_config(),
        queue_dir=queue,
        project_factory=factory,
    )

    result = service.upload_new_crops(dry_run=True)

    assert result["pending"] == 3
    assert result["uploaded"] == 0
    assert not (queue / "_roboflow_uploads.json").exists()


def test_nothing_pending_does_not_connect(tmp_path) -> None:
    def factory():
        raise AssertionError("must not connect with nothing to upload")

    service = RoboflowUploadService(
        config=_config(),
        queue_dir=tmp_path / "missing_queue",
        project_factory=factory,
    )

    result = service.upload_new_crops()

    assert result["pending"] == 0
    assert result["uploaded"] == 0


def test_manifest_is_kept_per_project(tmp_path) -> None:
    queue = tmp_path / "label_queue"
    _write_queue(queue)
    project = FakeProject()

    _service(queue, project).upload_new_crops()

    other = RoboflowUploadService(
        config=_config(project="other"),
        queue_dir=queue,
        project_factory=lambda: project,
    )
    result = other.upload_new_crops()

    assert result["uploaded"] == 3
    assert len(project.calls) == 6

    assert set(_manifest(queue)["projects"]) == {"ws/ppe", "ws/other"}


def test_missing_coco_uploads_without_prelabels(tmp_path) -> None:
    queue = tmp_path / "label_queue"
    _write_queue(queue, with_coco=False)
    project = FakeProject()

    result = _service(queue, project).upload_new_crops()

    assert result["uploaded"] == 3
    assert all(call["annotation_path"] is None for call in project.calls)


def test_corrupt_manifest_starts_fresh(tmp_path) -> None:
    queue = tmp_path / "label_queue"
    _write_queue(queue)
    (queue / "_roboflow_uploads.json").write_text("{not json", encoding="utf-8")
    project = FakeProject()

    result = _service(queue, project).upload_new_crops()

    assert result["uploaded"] == 3
    assert set(_manifest(queue)["projects"]["ws/ppe"]) == set(ALL_CROPS)


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ROBOFLOW_API_KEY", "abc")
    monkeypatch.setenv("ROBOFLOW_PROJECT", "ppe-crops")
    monkeypatch.setenv("ROBOFLOW_WORKSPACE", "")
    monkeypatch.setenv("ROBOFLOW_UPLOAD_AS_PREDICTION", "false")
    monkeypatch.setenv("ROBOFLOW_VALID_SPLIT_PERCENT", "35")
    monkeypatch.delenv("ROBOFLOW_BATCH_PREFIX", raising=False)

    config = RoboflowSettings.from_env()

    assert config.configured is True
    assert config.project_name == "ppe-crops"
    assert config.upload_as_prediction is False
    assert config.valid_split_percent == 35
    assert config.batch_prefix == "label-queue"

    monkeypatch.setenv("ROBOFLOW_API_KEY", "")
    assert RoboflowSettings.from_env().configured is False
