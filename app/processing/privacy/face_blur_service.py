from app.processing.privacy.privacy_service import PrivacyService
from abc import ABC, abstractmethod

import cv2
import numpy as np

from app.dto import VisionResultDTO

class FaceBlurPrivacyService(PrivacyService):
    """
    Applies privacy pixelation to the estimated head region
    of every detected person.

    Person detection is NOT performed here.

    The person bounding boxes are taken from VisionResultDTO.
    """

    HEAD_FRAC = 0.22
    PIXELATION_BLOCKS = 8

    def apply_privacy_blur(
        self,
        image_bytes: bytes,
        vision_result: VisionResultDTO,
    ) -> bytes:

        # --------------------------------------------------
        # Decode image bytes
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Work on a copy.
        # The original image is never modified.
        # --------------------------------------------------

        output = image.copy()

        image_height, image_width = output.shape[:2]

        # --------------------------------------------------
        # Blur every detected person's head
        # --------------------------------------------------

        for detection in vision_result.detections:

            box = detection.box

            x1 = max(0, min(image_width, box.x_min))
            y1 = max(0, min(image_height, box.y_min))
            x2 = max(0, min(image_width, box.x_max))
            y2 = max(0, min(image_height, box.y_max))

            person_width = x2 - x1
            person_height = y2 - y1

            if person_width <= 0 or person_height <= 0:
                continue

            # --------------------------------------------------
            # Estimate head region.
            #
            # Top 22% of the person bounding box.
            # Horizontally use approximately 70% of
            # the person's width, centered.
            # --------------------------------------------------

            head_y1 = y1

            head_y2 = min(
                image_height,
                y1 + max(
                    12,
                    int(person_height * self.HEAD_FRAC),
                ),
            )

            center_x = (x1 + x2) // 2

            half_width = max(
                10,
                int(person_width * 0.35),
            )

            head_x1 = max(
                0,
                center_x - half_width,
            )

            head_x2 = min(
                image_width,
                center_x + half_width,
            )

            if head_x2 <= head_x1 or head_y2 <= head_y1:
                continue

            # --------------------------------------------------
            # Extract head region
            # --------------------------------------------------

            roi = output[
                head_y1:head_y2,
                head_x1:head_x2,
            ]

            if roi.size == 0:
                continue

            roi_height, roi_width = roi.shape[:2]

            # --------------------------------------------------
            # Pixelation
            # --------------------------------------------------

            small_width = max(
                1,
                roi_width // self.PIXELATION_BLOCKS,
            )

            small_height = max(
                1,
                roi_height // self.PIXELATION_BLOCKS,
            )

            small = cv2.resize(
                roi,
                (small_width, small_height),
                interpolation=cv2.INTER_LINEAR,
            )

            pixelated = cv2.resize(
                small,
                (roi_width, roi_height),
                interpolation=cv2.INTER_NEAREST,
            )

            output[
                head_y1:head_y2,
                head_x1:head_x2,
            ] = pixelated

        # --------------------------------------------------
        # Encode processed image back to bytes
        #
        # JPEG is used because your input images are JPEGs
        # and this is appropriate for the backend image flow.
        # --------------------------------------------------

        success, encoded = cv2.imencode(
            ".jpg",
            output,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if not success:
            raise ValueError(
                "Could not encode privacy-processed image."
            )

        return encoded.tobytes()