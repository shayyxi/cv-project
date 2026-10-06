"""
Uploads new label-queue crops to Roboflow, once each, with the
model's PPE boxes attached as COCO pre-labels.
"""

import hashlib
import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from app.analytics.labeling_service import LabelingService
from app.retraining.config import (
    LABEL_QUEUE_DIR,
    RoboflowSettings,
    connect_project,
)

logger = logging.getLogger(__name__)

# Crops are named "<camera_id>_<timestamp>_p<person>_<ts>.jpg", after
# the raw frame "<camera_id>_<timestamp>.jpg" they were cut from.
CAMERA_PATTERN = re.compile(r"^([^_]+)_")


class RoboflowUploadService:
    """
    Pushes crops from the label queue folder to a Roboflow project.

    Every crop is uploaded exactly once: uploads are recorded per
    project in _roboflow_uploads.json next to the crops, so repeated
    runs (cron, --watch) only send what is new. A crop whose upload
    fails is not recorded and is retried on the next run.

    Pre-labels come from the queue's _annotations.coco.json. Each crop
    is sent with a COCO document holding only its own boxes, which is
    what Roboflow's per-image annotation API expects. Crops without
    any PPE box are uploaded unannotated.
    """

    MANIFEST_FILE = "_roboflow_uploads.json"
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
    TAG = "label-queue"

    def __init__(
        self,
        config: RoboflowSettings | None = None,
        queue_dir: Path | str | None = None,
        project_factory: Callable[[], object] | None = None,
    ) -> None:
        self._config = config or RoboflowSettings.from_env()

        self._queue_dir = (
            Path(queue_dir) if queue_dir else LABEL_QUEUE_DIR
        )

        self._project_factory = project_factory or (
            lambda: connect_project(self._config)
        )

        self._project = None

    def upload_new_crops(
        self,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Upload the crops not yet sent to the configured project.

        Returns {"pending": n, "uploaded": n, "duplicates": n,
        "failed": n, "batch": name, "manifest": path}. Duplicates are
        crops Roboflow already held (identical pixels); they count as
        uploaded. dry_run=True only reports what is pending.
        """

        manifest = self._load_manifest()

        uploads = manifest["projects"].setdefault(
            self._config.project_name, {}
        )

        pending = [
            name for name in self._crops_on_disk()
            if name not in uploads
        ]

        if limit is not None:
            pending = pending[:limit]

        batch_name = (
            f"{self._config.batch_prefix}-"
            f"{datetime.now(timezone.utc):%Y-%m-%d}"
        )

        summary = {
            "pending": len(pending),
            "uploaded": 0,
            "duplicates": 0,
            "failed": 0,
            "batch": batch_name,
            "manifest": self._queue_dir / self.MANIFEST_FILE,
        }

        if not pending:
            logger.info(
                "Roboflow upload: nothing new in %s", self._queue_dir
            )
            return summary

        if dry_run:
            logger.info(
                "Roboflow upload (dry run): %d crops pending in %s",
                len(pending),
                self._queue_dir,
            )
            return summary

        coco = self._load_coco()

        project = self._get_project()

        try:
            for index, name in enumerate(pending, start=1):

                # One line per crop; the Streamlit page turns these
                # into a progress bar.
                logger.info(
                    "Roboflow upload %d/%d: %s", index, len(pending), name
                )

                entry = coco["images"].get(name)

                split = self.split_for(name)

                tags = [self.TAG]

                camera_id = self.camera_id_for(name)

                if camera_id:
                    tags.append(f"camera-{camera_id}")

                annotation = self._annotation_for(name, entry, coco)

                try:
                    result = project.single_upload(
                        image_path=str(self._queue_dir / name),
                        annotation_path=annotation,
                        split=split,
                        batch_name=batch_name,
                        tag_names=tags,
                        is_prediction=self._config.upload_as_prediction,
                        num_retry_uploads=3,
                    )
                except Exception:
                    logger.exception(
                        "Roboflow upload failed for %s", name
                    )
                    summary["failed"] += 1
                    continue

                image = (result or {}).get("image") or {}

                duplicate = bool(image.get("duplicate"))

                uploads[name] = {
                    "image_id": image.get("id"),
                    "duplicate": duplicate,
                    "split": split,
                    "batch": batch_name,
                    "annotations": (
                        len(entry["annotations"]) if entry else 0
                    ),
                    "uploaded_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }

                summary["duplicates" if duplicate else "uploaded"] += 1

        finally:
            # Keep what succeeded even if the loop blew up.
            self._save_manifest(manifest)

        logger.info(
            "Roboflow upload: %d uploaded, %d duplicates, %d failed "
            "-> batch %s (%s)",
            summary["uploaded"],
            summary["duplicates"],
            summary["failed"],
            batch_name,
            self._config.project_name,
        )

        return summary

    def split_for(self, file_name: str) -> str:
        """
        Deterministic train/valid assignment from the file name, so a
        crop lands in the same split however often the queue is
        re-exported.
        """

        percent = self._config.valid_split_percent

        if percent <= 0:
            return "train"

        if percent >= 100:
            return "valid"

        digest = hashlib.sha1(file_name.encode("utf-8")).hexdigest()

        bucket = int(digest[:8], 16) % 100

        return "valid" if bucket < percent else "train"

    @staticmethod
    def camera_id_for(file_name: str) -> str | None:
        match = CAMERA_PATTERN.match(file_name)

        return match.group(1) if match else None

    def _get_project(self):
        if self._project is None:
            self._project = self._project_factory()

        return self._project

    def _crops_on_disk(self) -> list[str]:
        if not self._queue_dir.is_dir():
            return []

        return sorted(
            path.name
            for path in self._queue_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.IMAGE_SUFFIXES
        )

    def _annotation_for(
        self,
        name: str,
        entry: dict | None,
        coco: dict,
    ) -> dict | None:
        """
        A one-image COCO document for Roboflow's annotation upload
        ({"name", "rawText"}), or None when the crop has no boxes.
        """

        if not entry or not entry["annotations"]:
            return None

        annotations = []

        for index, annotation in enumerate(
            entry["annotations"], start=1
        ):
            bbox = list(annotation["bbox"])

            annotations.append(
                {
                    "id": index,
                    "image_id": 1,
                    "category_id": annotation["category_id"],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )

        document = {
            "info": coco["info"],
            "licenses": coco["licenses"],
            "categories": coco["categories"],
            "images": [
                {
                    "id": 1,
                    "file_name": name,
                    "width": entry["width"],
                    "height": entry["height"],
                }
            ],
            "annotations": annotations,
        }

        return {
            "name": "annotation.coco.json",
            "rawText": json.dumps(document),
        }

    def _load_coco(self) -> dict:
        """
        The queue's COCO file re-keyed by file name:
        {"info", "licenses", "categories",
         "images": {file_name: {"width", "height", "annotations"}}}.
        """

        path = self._queue_dir / LabelingService.ANNOTATIONS_FILE

        empty = {
            "info": {},
            "licenses": [],
            "categories": [
                {
                    "id": category_id,
                    "name": category,
                    "supercategory": "ppe",
                }
                for category, category_id
                in LabelingService.CATEGORIES.items()
            ],
            "images": {},
        }

        if not path.exists():
            logger.warning(
                "No %s in %s - uploading crops without pre-labels.",
                LabelingService.ANNOTATIONS_FILE,
                self._queue_dir,
            )
            return empty

        try:
            coco = json.loads(path.read_text(encoding="utf-8"))

            by_image_id: dict[int, list[dict]] = defaultdict(list)

            for annotation in coco.get("annotations", []):
                by_image_id[annotation["image_id"]].append(annotation)

            images = {}

            for image in coco.get("images", []):
                images[image["file_name"]] = {
                    "width": image["width"],
                    "height": image["height"],
                    "annotations": by_image_id.get(image["id"], []),
                }

            return {
                "info": coco.get("info", {}),
                "licenses": coco.get("licenses", []),
                "categories": (
                    coco.get("categories") or empty["categories"]
                ),
                "images": images,
            }

        except (OSError, ValueError, KeyError, TypeError):
            logger.warning(
                "Could not parse %s - uploading crops without "
                "pre-labels.",
                path,
            )
            return empty

    def _load_manifest(self) -> dict:
        path = self._queue_dir / self.MANIFEST_FILE

        if not path.exists():
            return {"projects": {}}

        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))

            if not isinstance(manifest.get("projects"), dict):
                raise ValueError("missing 'projects'")

            return manifest

        except (OSError, ValueError, TypeError):
            logger.warning(
                "Could not parse %s - starting a new manifest. Crops "
                "may be sent again; Roboflow de-duplicates identical "
                "images.",
                path,
            )
            return {"projects": {}}

    def _save_manifest(self, manifest: dict) -> None:
        self._queue_dir.mkdir(parents=True, exist_ok=True)

        (self._queue_dir / self.MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
