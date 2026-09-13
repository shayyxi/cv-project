from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import math
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from app.dto import (
    BoundingBoxDTO,
    VisionDetectionDTO,
    VisionResultDTO,
    ComplianceDTO,
    PPEDetectionDTO,

)
from app.processing.cv.vision_engine import VisionEngine
from app.utils.smart_crop import smart_crop


class PPEVisionEngine(VisionEngine):

    def __init__(self):
        torch.backends.mkldnn.enabled = False
        torch.set_num_threads(2)
        self._config = self._load_config()
        self._device = self._select_device()

        self._person_model = None
        self._ppe_model = None

        self._models_loaded = False

        self._load_settings()

        # Load models once during engine initialization.
        self.load_models()


    def load_models(self) -> None:
        """
        Load all vision models.

        Models are loaded only once.
        """

        if self._models_loaded:
            return

        self._person_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self._person_model_path,
            confidence_threshold=self._person_confidence,
            device=self._device,
        )

        self._ppe_model = YOLO(self._ppe_model_path)

        if self._device == "cuda":
            self._ppe_model.to("cuda")
        else:
            self._ppe_model.to("cpu")

        self._models_loaded = True

    def process_image(
        self,
        image_bytes: bytes,
    ) -> VisionResultDTO:

        image = self._decode_image(image_bytes)

        image_height, image_width = image.shape[:2]        #keep this line might need this for later


        raw_persons = self._detect_persons(image)


        persons = self._merge_persons(
            raw_persons,
            self._merge_iou,
        )

        persons = self._filter_person_shapes(persons)

        detections = []



        for person_id, (person_bbox, person_confidence) in enumerate(
            persons
        ):

            x1, y1, x2, y2 = person_bbox

            crop, transform = self._smart_crop(
                image,
                x1,
                y1,
                x2,
                y2,
            )

            ppe_detections = self._detect_ppe(crop)

            ppe_detections = self._filter_ppe_to_person(
                ppe_detections,
                transform,
                image.shape,
                person_bbox,
            )

            if self._keep_best_per_class:
                ppe_detections = self._best_per_class(
                    ppe_detections
                )

            has_helmet = any(
                d["label"] == self._helmet_class
                for d in ppe_detections
            )

            has_vest = any(
                d["label"] == self._vest_class
                for d in ppe_detections
            )

            has_boots = any(
                d["label"] == self._boots_class
                for d in ppe_detections
            )

            compliant = (
                has_helmet
                and has_vest
                and has_boots
            )



            detections.append(
                VisionDetectionDTO(
                    label="person",
                    confidence=round(
                        float(person_confidence),
                        4,
                    ),
                    box=BoundingBoxDTO(
                        x_min=x1,
                        y_min=y1,
                        x_max=x2,
                        y_max=y2,
                    ),
                    is_sensitive=False,

                    person_id=person_id,

                    compliance=ComplianceDTO(
                        helmet=has_helmet,
                        vest=has_vest,
                        boots=has_boots,
                        compliant=compliant,
                    ),

                    ppe=self._build_ppe_detections(
                        ppe_detections,
                        transform,
                        image.shape,
                    ),
                )
            )

        return VisionResultDTO(
            worker_count=len(persons),
            detections=detections,
        )



    def _load_config(self) -> dict:
        """
        Load vision configuration exclusively for this engine.
        """

        config_path = (
            Path(__file__).resolve().parent
            / "config"
            / "vision_config.yaml"
        )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Vision configuration not found: {config_path}"
            )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return yaml.safe_load(f)

    def _load_settings(self) -> None:

        models = self._config["models"]
        person = models["person"]
        ppe = models["ppe"]

        self._person_model_path = self._resolve_path(
            person["path"]
        )

        self._ppe_model_path = self._resolve_path(
            ppe["path"]
        )

        self._person_confidence = float(
            person.get("confidence", 0.5)
        )

        self._person_min_aspect = float(
            person.get("min_aspect_ratio", 0.0)
        )

        self._ppe_confidence = float(
            ppe.get("confidence", 0.25)
        )

        self._keep_best_per_class = bool(
            ppe.get("keep_best_per_class", True)
        )

        self._min_person_overlap = float(
            ppe.get("min_person_overlap", 0.3)
        )

        sahi = self._config["sahi"]

        self._num_cols = int(
            sahi.get("num_cols")
        )

        self._num_rows = int(
            sahi.get("num_rows")
        )


        self._overlap_width = float(
            sahi.get("overlap_width")
        )

        self._overlap_height = float(
            sahi.get("overlap_height")
        )

        self._merge_iou = float(
            sahi.get("merge_iou")
        )

        self._target_size = int(
            sahi.get("target_size")
        )
        self._overlap=float(sahi.get("overlap"))



        crop = self._config["crop"]

        self._crop_width = int(
            crop.get("width")
        )

        self._crop_height = int(
            crop.get("height")
        )

        self._scale_padding = float(
            crop.get("scale_padding")
        )

        classes = self._config["classes"]["ppe"]

        self._helmet_class = classes["helmet"].lower()
        self._vest_class = classes["vest"].lower()
        self._boots_class = classes["boots"].lower()

    def _resolve_path(self, path: str) -> str:

        path = Path(path)

        if path.is_absolute():
            return str(path)

        # Relative paths are relative to the CV package. (adjust these shazi if some problems regarding path)
        base_dir = Path(__file__).resolve().parent

        return str(
            (base_dir / path).resolve()
        )

    def _select_device(self) -> str:

        prefer_cuda = self._config.get(
            "device",
            {},
        ).get(
            "prefer_cuda",
            True,
        )

        if prefer_cuda and torch.cuda.is_available():
            return "cuda"

        return "cpu"


    @staticmethod
    def _decode_image(
        image_bytes: bytes,
    ) -> np.ndarray:

        buffer = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            buffer,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "Unable to decode image bytes."
            )

        return image

    def _optimal_slice_size(self,dim, target_size, overlap):

        stride_guess = target_size * (1 - overlap)
        n_guess = max(1, round((dim - target_size) / stride_guess) + 1)

        best_n, best_size, best_diff = None, None, float("inf")

        # Check a small neighborhood around the guess to find the best fit
        for n in range(max(1, n_guess - 2), n_guess + 3):
            denom = n - (n - 1) * overlap
            if denom <= 0:
                continue
            size = dim / denom
            diff = abs(size - target_size)
            if diff < best_diff:
                best_n, best_size, best_diff = n, size, diff

        return math.ceil(best_size), best_n


    def _get_optimal_slice_params(self,image_width, image_height, target_size, overlap):
        slice_w, num_cols = self._optimal_slice_size(image_width, target_size, overlap)
        slice_h, num_rows = self._optimal_slice_size(image_height, target_size, overlap)


        return slice_w, slice_h


    def _detect_persons(
        self,
        image: np.ndarray,
    ):

        height, width = image.shape[:2]

        #slice_width = max(
         #   1,
          #  width // self._num_cols,
        #)

        #slice_height = max(
         #   1,
          #  height // self._num_rows,
        #)

        slice_width,slice_height=self._get_optimal_slice_params(width,height,self._target_size,self._overlap)

        result = get_sliced_prediction(
            image,
            self._person_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=self._overlap,
            overlap_width_ratio=self._overlap,
            perform_standard_pred=True,
        )

        return [
            obj
            for obj in result.object_prediction_list
            if obj.category.id == 0
        ]

    def _merge_persons(
        self,
        persons,
        iou_thresh: float,
    ):

        boxes = []

        for obj in persons:

            x1, y1, x2, y2 = map(
                int,
                obj.bbox.to_xyxy(),
            )

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                    float(obj.score.value),
                ]
            )

        boxes.sort(
            key=lambda box: box[4],
            reverse=True,
        )

        merged = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):

            if used[i]:
                continue

            bbox = boxes[i][:4]
            best_score = boxes[i][4]

            used[i] = True

            for j in range(i + 1, len(boxes)):

                if used[j]:
                    continue

                if self._iou(
                    bbox,
                    boxes[j][:4],
                ) > iou_thresh:

                    other = boxes[j][:4]

                    bbox = [
                        min(bbox[0], other[0]),
                        min(bbox[1], other[1]),
                        max(bbox[2], other[2]),
                        max(bbox[3], other[3]),
                    ]

                    best_score = max(
                        best_score,
                        boxes[j][4],
                    )

                    used[j] = True

            merged.append(
                (
                    bbox,
                    best_score,
                )
            )

        return merged

    @staticmethod
    def _iou(a, b) -> float:

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        intersection = iw * ih

        if intersection <= 0:
            return 0.0

        area_a = (
            ax2 - ax1
        ) * (
            ay2 - ay1
        )

        area_b = (
            bx2 - bx1
        ) * (
            by2 - by1
        )

        return intersection / (
            area_a + area_b - intersection
        )


    def _smart_crop(
        self,
        image,
        x1,
        y1,
        x2,
        y2,
    ):

        return smart_crop(
            image,
            x1,
            y1,
            x2,
            y2,
            crop_width=self._crop_width,
            crop_height=self._crop_height,
            scale_padding=self._scale_padding,
        )



    def _detect_ppe(
        self,
        crop: np.ndarray,
    ):

        results = self._ppe_model.predict(
            crop,
            conf=self._ppe_confidence,
            verbose=False,
        )[0]

        detections = []

        if results.boxes is None:
            return detections

        boxes = (
            results.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        confidences = (
            results.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        class_ids = (
            results.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        for bbox, confidence, class_id in zip(
            boxes,
            confidences,
            class_ids,
        ):

            x1, y1, x2, y2 = map(
                int,
                bbox,
            )

            label = str(
                results.names[class_id]
            ).lower()

            detections.append(
                {
                    "label": label,
                    "confidence": float(
                        confidence
                    ),
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                }
            )

        return detections


    def _filter_person_shapes(self, persons):
        """
        Drop person boxes that are wider than person-shaped
        (height / width below min_aspect_ratio). Kills common false
        positives such as gravel piles and equipment, which the
        detector reports with high confidence but in boxes no
        standing or crouching worker produces.
        """

        if self._person_min_aspect <= 0:
            return persons

        kept = []

        for bbox, confidence in persons:

            x1, y1, x2, y2 = bbox

            width = max(x2 - x1, 1)
            height = y2 - y1

            if height / width >= self._person_min_aspect:
                kept.append((bbox, confidence))

        return kept

    def _filter_ppe_to_person(
        self,
        ppe_detections,
        transform,
        image_shape,
        person_bbox,
    ):
        """
        Keep only PPE boxes that lie on this person: at least
        min_person_overlap of the PPE box area must fall inside the
        person bbox. Drops a neighbour's gear that is visible in the
        padded crop, which would otherwise make this person falsely
        compliant.
        """

        if self._min_person_overlap <= 0:
            return ppe_detections

        px1, py1, px2, py2 = person_bbox

        kept = []

        for detection in ppe_detections:

            ox1, oy1, ox2, oy2 = self._crop_bbox_to_original(
                detection["bbox"],
                transform,
                image_shape,
            )

            area = (ox2 - ox1) * (oy2 - oy1)

            if area <= 0:
                continue

            overlap_w = min(ox2, px2) - max(ox1, px1)
            overlap_h = min(oy2, py2) - max(oy1, py1)

            overlap = max(0, overlap_w) * max(0, overlap_h)

            if overlap / area >= self._min_person_overlap:
                kept.append(detection)

        return kept

    @staticmethod
    def _best_per_class(ppe_detections):
        """
        One detection per PPE class: the highest-confidence box.
        """

        best = {}

        for detection in ppe_detections:

            current = best.get(detection["label"])

            if (
                current is None
                or detection["confidence"] > current["confidence"]
            ):
                best[detection["label"]] = detection

        return list(best.values())


    def _crop_bbox_to_original(
        self,
        bbox,
        transform,
        image_shape,
    ):

        x1, y1, x2, y2 = bbox

        scale_x = transform["scale_x"]
        scale_y = transform["scale_y"]

        original_x1 = (
            x1 / scale_x
            + transform["origin_x"]
        )

        original_y1 = (
            y1 / scale_y
            + transform["origin_y"]
        )

        original_x2 = (
            x2 / scale_x
            + transform["origin_x"]
        )

        original_y2 = (
            y2 / scale_y
            + transform["origin_y"]
        )

        height, width = image_shape[:2]

        return [
            max(
                0,
                min(
                    width - 1,
                    int(round(original_x1)),
                ),
            ),
            max(
                0,
                min(
                    height - 1,
                    int(round(original_y1)),
                ),
            ),
            max(
                0,
                min(
                    width - 1,
                    int(round(original_x2)),
                ),
            ),
            max(
                0,
                min(
                    height - 1,
                    int(round(original_y2)),
                ),
            ),
        ]


    def _build_ppe_detections(
        self,
        ppe_detections,
        transform,
        image_shape,
    ):

        results = []

        for detection in ppe_detections:

            crop_bbox = detection["bbox"]

            original_bbox = (
                self._crop_bbox_to_original(
                    crop_bbox,
                    transform,
                    image_shape,
                )
            )

            #DTO's have been modified by me

            results.append(
                {
                    "label": detection["label"],
                    "confidence": round(
                        detection["confidence"],
                        4,
                    ),
                    "box": BoundingBoxDTO(
                        x_min=original_bbox[0],
                        y_min=original_bbox[1],
                        x_max=original_bbox[2],
                        y_max=original_bbox[3],
                    ),
                    "crop_box": BoundingBoxDTO(
                        x_min=crop_bbox[0],
                        y_min=crop_bbox[1],
                        x_max=crop_bbox[2],
                        y_max=crop_bbox[3],
                    ),
                }
            )

        return results