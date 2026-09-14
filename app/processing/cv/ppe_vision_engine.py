from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import math
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ultralytics import YOLO

from app.dto import (
    BoundingBoxDTO,
    PointDTO,
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
            self._merge_overlap,
        )

        persons = self._filter_person_shapes(persons)

        detections = []



        for person_id, (
            person_bbox,
            person_confidence,
            person_polygons,
        ) in enumerate(persons):

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
                    mask=self._build_mask(
                        person_polygons,
                        image.shape,
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

        self._merge_overlap = float(
            sahi.get("merge_overlap")
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

    @classmethod
    def _merge_persons(
        cls,
        persons,
        overlap_thresh: float,
    ):
        """
        Collapse fragments of the same worker into one detection.

        SAHI's own merge is greedy and box-based: a slice fragment that
        does not directly overlap the highest-scoring fragment survives
        as its own "person" (legs-only, torso-only). Here every pair is
        compared with intersection-over-smaller on the silhouettes
        (boxes when no silhouette exists) and merging repeats until
        nothing overlaps above overlap_thresh.

        Overlap-over-smaller catches a torso or legs fragment sitting
        inside a full-body detection, which IoU never does. Comparing
        silhouettes instead of boxes keeps two neighbouring workers
        apart even when their boxes overlap heavily.

        Returns [(bbox, score, polygons)] sorted by score, polygons in
        COCO flat format ([x1, y1, x2, y2, ...] per polygon).
        """

        entries = [cls._merge_entry(obj) for obj in persons]

        entries.sort(
            key=lambda entry: entry["score"],
            reverse=True,
        )

        changed = True

        while changed:

            changed = False
            merged = []
            used = [False] * len(entries)

            for i, current in enumerate(entries):

                if used[i]:
                    continue

                used[i] = True

                for j in range(i + 1, len(entries)):

                    if used[j]:
                        continue

                    overlap = cls._overlap_over_smaller(
                        current,
                        entries[j],
                    )

                    if overlap >= overlap_thresh:
                        current = cls._merge_pair(
                            current,
                            entries[j],
                        )
                        used[j] = True
                        changed = True

                merged.append(current)

            entries = merged

        return [
            (
                entry["bbox"],
                entry["score"],
                (
                    cls._shape_to_polygons(entry["shape"])
                    if entry["shape"] is not None
                    else entry["polygons"]
                ),
            )
            for entry in entries
        ]

    @classmethod
    def _merge_entry(cls, obj) -> dict:

        x1, y1, x2, y2 = map(
            int,
            obj.bbox.to_xyxy(),
        )

        # SAHI attaches a Mask (COCO polygons, full-image coords)
        # only when the person weight is a segmentation model.
        mask = getattr(obj, "mask", None)

        polygons = (
            list(mask.segmentation)
            if mask is not None
            else []
        )

        return {
            "bbox": [x1, y1, x2, y2],
            "score": float(obj.score.value),
            "polygons": polygons,
            "shape": cls._polygons_to_shape(polygons),
        }

    @staticmethod
    def _polygons_to_shape(polygons):
        """
        COCO flat polygons -> one shapely geometry (or None when there
        is no valid polygon). buffer(0) repairs self-intersections that
        contour tracing produces.
        """

        shapes = []

        for polygon in polygons:

            coords = list(polygon)

            if len(coords) < 6:
                continue

            shape = Polygon(
                list(zip(coords[0::2], coords[1::2]))
            ).buffer(0)

            if not shape.is_empty:
                shapes.append(shape)

        if not shapes:
            return None

        return unary_union(shapes)

    @staticmethod
    def _shape_to_polygons(shape):
        """
        Shapely geometry -> COCO flat polygons (exterior rings only).
        """

        geoms = (
            shape.geoms
            if hasattr(shape, "geoms")
            else [shape]
        )

        polygons = []

        for geom in geoms:

            if geom.is_empty or geom.geom_type != "Polygon":
                continue

            # Drop the closing point shapely repeats.
            coords = list(geom.exterior.coords)[:-1]

            if len(coords) < 3:
                continue

            polygons.append(
                [
                    value
                    for point in coords
                    for value in point
                ]
            )

        return polygons

    @classmethod
    def _overlap_over_smaller(cls, a: dict, b: dict) -> float:

        if a["shape"] is not None and b["shape"] is not None:

            smaller = min(a["shape"].area, b["shape"].area)

            if smaller > 0:
                return (
                    a["shape"].intersection(b["shape"]).area
                    / smaller
                )

        return cls._box_overlap_over_smaller(
            a["bbox"],
            b["bbox"],
        )

    @staticmethod
    def _merge_pair(a: dict, b: dict) -> dict:

        if a["shape"] is not None and b["shape"] is not None:
            shape = a["shape"].union(b["shape"])
        else:
            shape = a["shape"] if a["shape"] is not None else b["shape"]

        return {
            "bbox": [
                min(a["bbox"][0], b["bbox"][0]),
                min(a["bbox"][1], b["bbox"][1]),
                max(a["bbox"][2], b["bbox"][2]),
                max(a["bbox"][3], b["bbox"][3]),
            ],
            "score": max(a["score"], b["score"]),
            "polygons": a["polygons"] + b["polygons"],
            "shape": shape,
        }

    @staticmethod
    def _build_mask(
        polygons,
        image_shape,
    ):
        """
        Convert SAHI COCO polygons ([x1, y1, x2, y2, ...] per polygon)
        into PointDTO lists clamped to the image. Returns None when the
        detector produced no usable silhouette (detect-only weights, or
        masks with fewer than three vertices), so the renderer falls
        back to the bounding box.
        """

        height, width = image_shape[:2]

        result = []

        for polygon in polygons:

            coords = list(polygon)

            if len(coords) < 6:
                continue

            points = [
                PointDTO(
                    x=max(
                        0,
                        min(width - 1, int(round(coords[i]))),
                    ),
                    y=max(
                        0,
                        min(height - 1, int(round(coords[i + 1]))),
                    ),
                )
                for i in range(0, len(coords) - 1, 2)
            ]

            result.append(points)

        return result or None

    @staticmethod
    def _box_overlap_over_smaller(a, b) -> float:

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))

        intersection = iw * ih

        if intersection <= 0:
            return 0.0

        smaller = min(
            (ax2 - ax1) * (ay2 - ay1),
            (bx2 - bx1) * (by2 - by1),
        )

        if smaller <= 0:
            return 0.0

        return intersection / smaller


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

        for bbox, confidence, polygons in persons:

            x1, y1, x2, y2 = bbox

            width = max(x2 - x1, 1)
            height = y2 - y1

            if height / width >= self._person_min_aspect:
                kept.append((bbox, confidence, polygons))

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