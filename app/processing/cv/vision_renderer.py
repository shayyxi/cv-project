from pathlib import Path

import cv2
import numpy as np
import yaml

from app.dto import VisionDetectionDTO

# (x_min, y_min, x_max, y_max) of an area already occupied by a label.
Rect = tuple[int, int, int, int]


class VisionRenderer:

    REGION_COLOR = (255, 0, 0)

    CROP_STATUS_BAR_HEIGHT = 32

    # Opacity of the tint inside a person silhouette (0 = none, 1 = solid).
    MASK_FILL_ALPHA = 0.25

    MASK_OUTLINE_THICKNESS = 4

    def __init__(self):

        self._config = self._load_config()

        colors = self._config["colors"]

        self._compliant_color = tuple(
            colors["compliant"]
        )

        self._non_compliant_color = tuple(
            colors["non_compliant"]
        )

        self._ppe_colors = {
            "helmet": tuple(
                colors["ppe"]["helmet"]
            ),
            "vest": tuple(
                colors["ppe"]["vest"]
            ),
            "boots": tuple(
                colors["ppe"]["boots"]
            ),
        }

    # ==================================================================
    # Public API
    # ==================================================================

    def draw_original(
            self,
            image_bytes: bytes,
            result,
    ) -> bytes:

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "Could not decode image bytes."
            )

        annotated = image.copy()

        # Pass 1: every silhouette/box. Pass 2: every label, so labels
        # sit on top of all shapes and can be placed without overlapping
        # each other.
        self._draw_region_polygon(
            annotated,
            result,
        )

        self._draw_persons(
            annotated,
            result.detections,
        )

        for person in result.detections:
            for detection in person.ppe:
                self._draw_box(
                    annotated,
                    detection.box,
                    self._ppe_color(detection),
                    3,
                )

        placed: list[Rect] = []

        self._draw_region_label(
            annotated,
            result,
            placed,
        )

        for person in result.detections:
            self._draw_person_label(
                annotated,
                person,
                placed,
            )

            for detection in person.ppe:
                self._draw_ppe_label(
                    annotated,
                    detection,
                    detection.box,
                    placed,
                )

        success, encoded = cv2.imencode(
            ".jpg",
            annotated,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if not success:
            raise ValueError(
                "Could not encode annotated image."
            )

        return encoded.tobytes()

    def draw_crop(
        self,
        crop: np.ndarray,
        person: VisionDetectionDTO,
    ) -> np.ndarray:

        annotated = crop.copy()

        for detection in person.ppe:
            self._draw_box(
                annotated,
                detection.crop_box,
                self._ppe_color(detection),
                3,
            )

        # Reserve the status bar so no label ends up hidden under it.
        placed: list[Rect] = [
            (0, 0, annotated.shape[1], self.CROP_STATUS_BAR_HEIGHT),
        ]

        for detection in person.ppe:
            self._draw_ppe_label(
                annotated,
                detection,
                detection.crop_box,
                placed,
            )

        self._draw_crop_status(
            annotated,
            person,
        )

        return annotated

    # ==================================================================
    # Region
    # ==================================================================

    @staticmethod
    def _region_polygon(result):

        region = result.processing_region

        if region is None or not region.polygon:
            return None

        return np.array(
            [
                [point.x, point.y]
                for point in region.polygon
            ],
            dtype=np.int32,
        )

    def _draw_region_polygon(
            self,
            image,
            result,
    ):
        polygon = self._region_polygon(result)

        if polygon is None:
            return

        cv2.polylines(
            image,
            [polygon],
            isClosed=True,
            color=self.REGION_COLOR,
            thickness=5,
        )

    def _draw_region_label(
            self,
            image,
            result,
            placed: list[Rect],
    ):
        polygon = self._region_polygon(result)

        if polygon is None:
            return

        # Label near the first polygon point
        x = int(polygon[0][0])
        y = int(polygon[0][1])

        self._draw_label(
            image,
            "DETECTION REGION",
            x,
            y - 10,
            self.REGION_COLOR,
            scale=0.55,
            thickness=1,
            placed=placed,
        )

    # ==================================================================
    # People and PPE
    # ==================================================================

    def _draw_persons(
        self,
        image,
        persons,
    ):
        """
        Segmented persons get a translucent fill plus an outline in the
        compliance color. Persons without a mask (detect-only weights)
        keep the rectangle. All fills are blended in a single pass so
        overlapping workers do not stack tints.
        """

        overlay = None

        for person in persons:

            polygons = self._mask_polygons(person.mask)

            if polygons is None:
                self._draw_box(
                    image,
                    person.box,
                    self._person_color(person),
                    4,
                )
                continue

            if overlay is None:
                overlay = image.copy()

            cv2.fillPoly(
                overlay,
                polygons,
                self._person_color(person),
            )

        if overlay is None:
            return

        cv2.addWeighted(
            overlay,
            self.MASK_FILL_ALPHA,
            image,
            1 - self.MASK_FILL_ALPHA,
            0,
            dst=image,
        )

        for person in persons:

            polygons = self._mask_polygons(person.mask)

            if polygons is None:
                continue

            cv2.polylines(
                image,
                polygons,
                isClosed=True,
                color=self._person_color(person),
                thickness=self.MASK_OUTLINE_THICKNESS,
            )

    @staticmethod
    def _mask_polygons(mask):
        """
        DTO mask (list of PointDTO polygons) -> list of int32 Nx2 arrays
        for cv2. None when there is nothing drawable.
        """

        if not mask:
            return None

        polygons = [
            np.array(
                [
                    [point.x, point.y]
                    for point in polygon
                ],
                dtype=np.int32,
            )
            for polygon in mask
            if len(polygon) >= 3
        ]

        return polygons or None

    def _person_color(
        self,
        person,
    ):
        return (
            self._compliant_color
            if person.compliance.compliant
            else self._non_compliant_color
        )

    def _ppe_color(
        self,
        detection,
    ):
        return self._ppe_colors.get(
            detection.label.lower(),
            (200, 200, 200),
        )

    def _draw_person_label(
        self,
        image,
        person,
        placed: list[Rect],
    ):

        status = (
            "COMPLIANT"
            if person.compliance.compliant
            else "NON-COMPLIANT"
        )

        text = (
            f"P{person.person_id} {person.confidence:.2f} | {status} | "
            f"H:{'Y' if person.compliance.helmet else 'N'} "
            f"V:{'Y' if person.compliance.vest else 'N'} "
            f"B:{'Y' if person.compliance.boots else 'N'}"
        )

        self._draw_label(
            image,
            text,
            person.box.x_min,
            person.box.y_min - 10,
            self._person_color(person),
            scale=0.55,
            thickness=1,
            placed=placed,
        )

    def _draw_ppe_label(
        self,
        image,
        detection,
        bbox,
        placed: list[Rect],
    ):

        self._draw_label(
            image,
            (
                f"{self._display_label(detection.label)} "
                f"{detection.confidence:.2f}"
            ),
            bbox.x_min,
            bbox.y_min - 5,
            self._ppe_color(detection),
            placed=placed,
        )

    def _draw_crop_status(
        self,
        image,
        person,
    ):

        color = self._person_color(person)

        status = (
            "COMPLIANT"
            if person.compliance.compliant
            else "NON-COMPLIANT"
        )

        text = (
            f"P{person.person_id} {person.confidence:.2f} | {status} | "
            f"Hat:{'YES' if person.compliance.helmet else 'NO'} "
            f"Vest:{'YES' if person.compliance.vest else 'NO'} "
            f"Boots:{'YES' if person.compliance.boots else 'NO'}"
        )

        cv2.rectangle(
            image,
            (0, 0),
            (image.shape[1], self.CROP_STATUS_BAR_HEIGHT),
            color,
            -1,
        )

        cv2.putText(
            image,
            text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ==================================================================
    # Primitives
    # ==================================================================

    @staticmethod
    def _draw_box(
        image,
        bbox,
        color,
        thickness,
    ):

        cv2.rectangle(
            image,
            (
                bbox.x_min,
                bbox.y_min,
            ),
            (
                bbox.x_max,
                bbox.y_max,
            ),
            color,
            thickness,
        )

    @staticmethod
    def _draw_label(
        image,
        text,
        x,
        y,
        color,
        scale=0.45,
        thickness=1,
        placed: list[Rect] | None = None,
    ) -> Rect:
        """
        Draw `text` in a filled box whose baseline is near (x, y).

        The box is kept inside the image. When `placed` is given, the
        box is moved down past any rectangle already in it so labels
        never overlap each other, and the final rectangle is appended.
        """

        (tw, th), base = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )

        box_w = tw + 4
        box_h = th + base + 4

        image_h, image_w = image.shape[:2]

        x = max(0, min(int(x), image_w - box_w))
        top = max(0, int(y) - th - base - 2)

        if placed:
            # Each pass moves the box below one rectangle it overlaps.
            # A box never re-hits a rectangle it was moved below, so
            # len(placed) passes are enough to clear all of them.
            for _ in range(len(placed)):
                rect = (x, top, x + box_w, top + box_h)

                hit = next(
                    (
                        other
                        for other in placed
                        if VisionRenderer._overlaps(rect, other)
                    ),
                    None,
                )

                if hit is None:
                    break

                top = hit[3] + 1

        top = max(0, min(top, image_h - box_h))

        rect = (x, top, x + box_w, top + box_h)

        cv2.rectangle(
            image,
            (rect[0], rect[1]),
            (rect[2], rect[3]),
            color,
            -1,
        )

        text_color = (
            (0, 0, 0)
            if sum(color) > 400
            else (255, 255, 255)
        )

        cv2.putText(
            image,
            text,
            (x + 2, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

        if placed is not None:
            placed.append(rect)

        return rect

    @staticmethod
    def _overlaps(
        a: Rect,
        b: Rect,
    ) -> bool:
        return (
            a[0] < b[2]
            and b[0] < a[2]
            and a[1] < b[3]
            and b[1] < a[3]
        )

    @staticmethod
    def _display_label(
        label: str,
    ):

        mapping = {
            "helmet": "Helmet",
            "vest": "High Visibility Vest",
            "boots": "Safety Boots",
        }

        return mapping.get(
            label.lower(),
            label,
        )

    # ==================================================================
    # Config
    # ==================================================================

    @staticmethod
    def _load_config():

        config_path = (
            Path(__file__).resolve().parent
            / "config"
            / "vision_config.yaml"
        )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            return yaml.safe_load(f)
