from pathlib import Path

import cv2
import numpy as np
import yaml

from app.dto import VisionDetectionDTO


class VisionRenderer:

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
            "safety vest": tuple(
                colors["ppe"]["safety_vest"]
            ),
            "safety boots": tuple(
                colors["ppe"]["safety_boots"]
            ),
        }

    # ==================================================================
    # Public API
    # ==================================================================

    def draw_original(
        self,
        image: np.ndarray,
        result,
    ) -> np.ndarray:

        annotated = image.copy()

        for person in result.detections:

            self._draw_person(
                annotated,
                person,
            )

            self._draw_ppe(
                annotated,
                person,
            )

        return annotated

    def draw_crop(
        self,
        crop: np.ndarray,
        person: VisionDetectionDTO,
    ) -> np.ndarray:

        annotated = crop.copy()

        for detection in person.ppe:

            bbox = detection.crop_box

            color = self._ppe_colors.get(
                detection.label.lower(),
                (200, 200, 200),
            )

            self._draw_box(
                annotated,
                bbox,
                color,
                3,
            )

            self._draw_label(
                annotated,
                (
                    f"{self._display_label(detection.label)} "
                    f"{detection.confidence:.2f}"
                ),
                bbox.x_min,
                max(
                    bbox.y_min - 5,
                    20,
                ),
                color,
            )

        self._draw_crop_status(
            annotated,
            person,
        )

        return annotated


    def _draw_person(
        self,
        image,
        person,
    ):

        color = (
            self._compliant_color
            if person.compliance.compliant
            else self._non_compliant_color
        )

        self._draw_box(
            image,
            person.box,
            color,
            4,
        )

        status = (
            "COMPLIANT"
            if person.compliance.compliant
            else "NON-COMPLIANT"
        )

        text = (
            f"P{person.person_id} | {status} | "
            f"H:{'Y' if person.compliance.helmet else 'N'} "
            f"V:{'Y' if person.compliance.vest else 'N'} "
            f"B:{'Y' if person.compliance.boots else 'N'}"
        )

        self._draw_label(
            image,
            text,
            person.box.x_min,
            max(
                person.box.y_min - 10,
                25,
            ),
            color,
            scale=0.8,
            thickness=2,
        )


    def _draw_ppe(
        self,
        image,
        person,
    ):

        for detection in person.ppe:

            bbox = detection.box

            color = self._ppe_colors.get(
                detection.label.lower(),
                (200, 200, 200),
            )

            self._draw_box(
                image,
                bbox,
                color,
                3,
            )

            self._draw_label(
                image,
                (
                    f"{self._display_label(detection.label)} "
                    f"{detection.confidence:.2f}"
                ),
                bbox.x_min,
                max(
                    bbox.y_min - 5,
                    20,
                ),
                color,
            )



    def _draw_crop_status(
        self,
        image,
        person,
    ):

        color = (
            self._compliant_color
            if person.compliance.compliant
            else self._non_compliant_color
        )

        status = (
            "COMPLIANT"
            if person.compliance.compliant
            else "NON-COMPLIANT"
        )

        text = (
            f"P{person.person_id} | {status} | "
            f"Hat:{'YES' if person.compliance.helmet else 'NO'} "
            f"Vest:{'YES' if person.compliance.vest else 'NO'} "
            f"Boots:{'YES' if person.compliance.boots else 'NO'}"
        )

        cv2.rectangle(
            image,
            (0, 0),
            (image.shape[1], 45),
            color,
            -1,
        )

        cv2.putText(
            image,
            text,
            (8, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )



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
        scale=0.6,
        thickness=2,
    ):

        (tw, th), base = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )

        y = max(
            y,
            th + 4,
        )

        cv2.rectangle(
            image,
            (x, y - th - base - 2),
            (x + tw + 4, y + 2),
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
            (x + 2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _display_label(
        label: str,
    ):

        mapping = {
            "helmet": "Helmet",
            "safety vest": "High Visibility Vest",
            "safety boots": "Safety Boots",
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