import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import yaml

from app.analytics.analytics_repository import AnalyticsRepository
from app.config import settings
from app.utils.smart_crop import smart_crop

logger = logging.getLogger(__name__)

VISION_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "processing"
    / "cv"
    / "config"
    / "vision_config.yaml"
)

DEFAULT_CROP_SETTINGS = {
    "width": 512,
    "height": 640,
    "scale_padding": 1.40,
}


def _load_crop_settings() -> dict:
    """
    The vision engine's crop settings, so exported crops match what
    the PPE model sees at inference time.
    """

    try:
        config = yaml.safe_load(
            VISION_CONFIG_PATH.read_text(encoding="utf-8")
        )
        crop = config["crop"]

        return {
            "width": int(crop["width"]),
            "height": int(crop["height"]),
            "scale_padding": float(crop["scale_padding"]),
        }
    except (OSError, KeyError, TypeError, ValueError):
        logger.warning(
            "Could not read crop settings from %s - using defaults.",
            VISION_CONFIG_PATH,
        )
        return dict(DEFAULT_CROP_SETTINGS)


class LabelingService:
    """
    Exports crops of non-compliant workers into a labelling queue
    folder, feeding the (external) retraining workflow.

    Alongside the crops it writes _annotations.coco.json with the
    model's helmet / vest / boots boxes (translated into crop
    coordinates) as COCO pre-labels for the annotation tool.

    Run soon after inference: it needs the source raw images still
    on disk.
    """

    ANNOTATIONS_FILE = "_annotations.coco.json"

    # Fixed COCO category ids, stable across runs.
    CATEGORIES = {
        "helmet": 1,
        "vest": 2,
        "boots": 3,
    }

    def __init__(
        self,
        repository: AnalyticsRepository,
    ) -> None:
        self._repository = repository

        self._output_dir = (
            settings.local_analytics_dir / "label_queue"
        )

        self._crop_settings = _load_crop_settings()

    def export_violation_crops(
        self,
        days: int = 7,
        limit: int = 200,
        include_compliant: bool = False,
    ) -> dict:
        """
        Returns {"saved": n, "skipped": n, "annotations": n,
        "output_dir": path, "coco_path": path}.
        Skipped records are those whose source image is gone from
        disk or whose box is degenerate.

        include_compliant=True also exports compliant persons, whose
        PPE boxes become pre-labels.
        """

        since = datetime.utcnow() - timedelta(days=days)

        records = self._repository.violation_records(
            since=since,
            limit=limit,
            include_compliant=include_compliant,
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)

        ppe_by_person = self._ppe_by_person(records)

        saved = 0
        skipped = 0

        # file_name -> {"width", "height", "annotations": [...]}
        entries: dict[str, dict] = {}

        for record in records:

            image_path = Path(record["raw_image_path"])

            image = (
                cv2.imread(str(image_path))
                if image_path.exists()
                else None
            )

            if image is None:
                skipped += 1
                continue

            height, width = image.shape[:2]

            x_min = max(0, int(record["x_min"]))
            y_min = max(0, int(record["y_min"]))
            x_max = min(width, int(record["x_max"]))
            y_max = min(height, int(record["y_max"]))

            if x_max <= x_min or y_max <= y_min:
                skipped += 1
                continue

            timestamp = record["ts"].strftime("%Y%m%dT%H%M%S")

            name = (
                f"{image_path.stem}"
                f"_p{record['person_id']}"
                f"_{timestamp}.jpg"
            )

            # Same padded, fixed-aspect crop the PPE model sees at
            # inference time, instead of the tight person box.
            crop, transform = smart_crop(
                image,
                x_min,
                y_min,
                x_max,
                y_max,
                crop_width=self._crop_settings["width"],
                crop_height=self._crop_settings["height"],
                scale_padding=self._crop_settings["scale_padding"],
            )

            cv2.imwrite(
                str(self._output_dir / name),
                crop,
            )

            saved += 1

            key = (record["image_job_id"], record["person_id"])

            crop_height, crop_width = crop.shape[:2]

            entries[name] = {
                "width": crop_width,
                "height": crop_height,
                "annotations": self._crop_annotations(
                    ppe_by_person.get(key, []),
                    transform=transform,
                    crop_size=(crop_width, crop_height),
                ),
            }

        coco_path, annotation_count = self._write_coco(entries)

        logger.info(
            "Label queue export: %d crops, %d PPE pre-labels -> %s "
            "(%d skipped)",
            saved,
            annotation_count,
            self._output_dir,
            skipped,
        )

        return {
            "saved": saved,
            "skipped": skipped,
            "annotations": annotation_count,
            "output_dir": self._output_dir,
            "coco_path": coco_path,
        }

    def _ppe_by_person(
        self,
        records: list[dict],
    ) -> dict[tuple, list[dict]]:
        """
        PPE boxes of the exported jobs, grouped by
        (image_job_id, person_id).
        """

        job_ids = sorted(
            {record["image_job_id"] for record in records}
        )

        grouped: dict[tuple, list[dict]] = defaultdict(list)

        for ppe in self._repository.ppe_records(job_ids):
            grouped[
                (ppe["image_job_id"], ppe["person_id"])
            ].append(ppe)

        return grouped

    def _crop_annotations(
        self,
        ppe_records: list[dict],
        transform: dict,
        crop_size: tuple[int, int],
    ) -> list[dict]:
        """
        PPE boxes translated from original-image coordinates into
        the smart crop's frame (shift by the crop origin, then apply
        the resize scale), clipped to it. Boxes fully outside the
        crop are dropped.
        """

        origin_x = transform["origin_x"]
        origin_y = transform["origin_y"]
        scale_x = transform["scale_x"]
        scale_y = transform["scale_y"]

        crop_w, crop_h = crop_size

        annotations = []

        for ppe in ppe_records:

            category = self._category_for(ppe["label"])

            if category is None:
                continue

            x_min = int(round((int(ppe["x_min"]) - origin_x) * scale_x))
            y_min = int(round((int(ppe["y_min"]) - origin_y) * scale_y))
            x_max = int(round((int(ppe["x_max"]) - origin_x) * scale_x))
            y_max = int(round((int(ppe["y_max"]) - origin_y) * scale_y))

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(crop_w, x_max)
            y_max = min(crop_h, y_max)

            if x_max <= x_min or y_max <= y_min:
                continue

            annotations.append(
                {
                    "category": category,
                    "bbox": [
                        x_min,
                        y_min,
                        x_max - x_min,
                        y_max - y_min,
                    ],
                    "score": round(float(ppe["confidence"]), 4),
                }
            )

        return annotations

    def _category_for(self, label: str) -> str | None:
        """
        Map a model class name ("helmet", "safety vest",
        "safety boots", ...) to a canonical COCO category.
        """

        label = label.lower()

        if "helmet" in label:
            return "helmet"
        if "vest" in label:
            return "vest"
        if "boot" in label:
            return "boots"

        return None

    def _write_coco(
        self,
        new_entries: dict[str, dict],
    ) -> tuple[Path, int]:
        """
        Write _annotations.coco.json covering the crops exported in
        this run plus previously annotated crops still on disk.
        Returns (path, number of annotations in this run's crops).
        """

        path = self._output_dir / self.ANNOTATIONS_FILE

        entries = self._existing_entries(path)
        entries.update(new_entries)

        images = []
        annotations = []
        annotation_id = 1

        for image_id, file_name in enumerate(
            sorted(entries), start=1
        ):
            entry = entries[file_name]

            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": entry["width"],
                    "height": entry["height"],
                }
            )

            for annotation in entry["annotations"]:
                bbox = annotation["bbox"]

                record = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": self.CATEGORIES[
                        annotation["category"]
                    ],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }

                if annotation.get("score") is not None:
                    record["score"] = annotation["score"]

                annotations.append(record)
                annotation_id += 1

        coco = {
            "info": {
                "description": (
                    "PPE violation crops with model pre-labels"
                ),
                "date_created": (
                    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                ),
            },
            "licenses": [],
            "categories": [
                {
                    "id": category_id,
                    "name": name,
                    "supercategory": "ppe",
                }
                for name, category_id in self.CATEGORIES.items()
            ],
            "images": images,
            "annotations": annotations,
        }

        path.write_text(
            json.dumps(coco, indent=2),
            encoding="utf-8",
        )

        new_count = sum(
            len(entry["annotations"])
            for entry in new_entries.values()
        )

        return path, new_count

    def _existing_entries(self, path: Path) -> dict[str, dict]:
        """
        Entries from a previous run's COCO file, for crops still on
        disk - so re-running the export never drops annotations of
        images it did not re-export.
        """

        if not path.exists():
            return {}

        try:
            coco = json.loads(path.read_text(encoding="utf-8"))

            id_to_category = {
                category["id"]: category["name"]
                for category in coco.get("categories", [])
            }

            by_image_id: dict[int, list[dict]] = defaultdict(list)

            for annotation in coco.get("annotations", []):
                category = id_to_category.get(
                    annotation["category_id"]
                )

                if category not in self.CATEGORIES:
                    continue

                by_image_id[annotation["image_id"]].append(
                    {
                        "category": category,
                        "bbox": annotation["bbox"],
                        "score": annotation.get("score"),
                    }
                )

            entries = {}

            for image in coco.get("images", []):
                file_name = image["file_name"]

                if not (self._output_dir / file_name).exists():
                    continue

                entries[file_name] = {
                    "width": image["width"],
                    "height": image["height"],
                    "annotations": by_image_id.get(
                        image["id"], []
                    ),
                }

            return entries

        except (OSError, ValueError, KeyError, TypeError):
            logger.warning(
                "Could not parse existing %s - rebuilding it "
                "from this run only.",
                path,
            )
            return {}
