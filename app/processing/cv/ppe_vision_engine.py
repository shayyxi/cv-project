from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

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


class PPEVisionEngine(VisionEngine):

    def __init__(self):
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

        self._ppe_confidence = float(
            ppe.get("confidence", 0.25)
        )

        sahi = self._config["sahi"]

        self._num_cols = int(
            sahi.get("num_cols", 4)
        )

        self._num_rows = int(
            sahi.get("num_rows", 3)
        )

        self._overlap_width = float(
            sahi.get("overlap_width", 0.2)
        )

        self._overlap_height = float(
            sahi.get("overlap_height", 0.2)
        )

        self._merge_iou = float(
            sahi.get("merge_iou", 0.45)
        )

        crop = self._config["crop"]

        self._crop_width = int(
            crop.get("width", 512)
        )

        self._crop_height = int(
            crop.get("height", 640)
        )

        self._scale_padding = float(
            crop.get("scale_padding", 1.4)
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


    def _detect_persons(
        self,
        image: np.ndarray,
    ):

        height, width = image.shape[:2]

        slice_width = max(
            1,
            width // self._num_cols,
        )

        slice_height = max(
            1,
            height // self._num_rows,
        )

        result = get_sliced_prediction(
            image,
            self._person_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=self._overlap_height,
            overlap_width_ratio=self._overlap_width,
            perform_standard_pred=False,
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

        image_height, image_width = image.shape[:2]

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        scale = max(
            bbox_width / self._crop_width,
            bbox_height / self._crop_height,
        )

        scale = max(
            scale * self._scale_padding,
            1.2,
        )

        half_width = int(
            self._crop_width * scale
        ) // 2

        half_height = int(
            self._crop_height * scale
        ) // 2

        crop_x1 = center_x - half_width
        crop_y1 = center_y - half_height
        crop_x2 = center_x + half_width
        crop_y2 = center_y + half_height

        pad_left = max(
            0,
            -crop_x1,
        )

        pad_top = max(
            0,
            -crop_y1,
        )

        pad_right = max(
            0,
            crop_x2 - image_width,
        )

        pad_bottom = max(
            0,
            crop_y2 - image_height,
        )

        actual_x1 = max(
            0,
            crop_x1,
        )

        actual_y1 = max(
            0,
            crop_y1,
        )

        actual_x2 = min(
            image_width,
            crop_x2,
        )

        actual_y2 = min(
            image_height,
            crop_y2,
        )

        crop = image[
            actual_y1:actual_y2,
            actual_x1:actual_x2,
        ].copy()

        if (
            pad_left
            or pad_top
            or pad_right
            or pad_bottom
        ):
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        padded_height, padded_width = crop.shape[:2]

        crop = cv2.resize(
            crop,
            (
                self._crop_width,
                self._crop_height,
            ),
            interpolation=cv2.INTER_LANCZOS4,
        )

        transform = {
            "origin_x": crop_x1,
            "origin_y": crop_y1,
            "scale_x": self._crop_width / padded_width,
            "scale_y": self._crop_height / padded_height,
        }

        return crop, transform



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