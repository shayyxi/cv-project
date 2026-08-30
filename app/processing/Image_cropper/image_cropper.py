# from pathlib import Path
#
# import cv2
# import numpy as np
# import yaml
#
# from app.dto import BoundingBoxDTO
#
#
# class ImageCropper:
#     """
#     Provides camera-specific regions of interest (ROI).
#
#     The ROI is used only for computer-vision inference.
#     The original image remains the source image for rendering
#     and final output.
#     """
#
#     def __init__(self) -> None:
#         self._config = self._load_config()
#
#         self._camera_regions = (
#             self._config.get("camera_crops", {})
#         )
#
#     def crop(
#         self,
#         image_bytes: bytes,
#         camera_id: str,
#     ) -> tuple[bytes, BoundingBoxDTO | None]:
#         """
#         Crop the image according to camera configuration.
#
#         Returns:
#             cropped_image_bytes
#             crop_region in original-image coordinates
#
#         If the camera has no configured crop:
#             - original image bytes are returned
#             - crop_region is None
#         """
#
#         crop_config = self._camera_regions.get(
#             str(camera_id)
#         )
#
#         # --------------------------------------------------
#         # No crop configured
#         # --------------------------------------------------
#
#         if crop_config is None:
#             return image_bytes, None
#
#         # --------------------------------------------------
#         # Decode image
#         # --------------------------------------------------
#
#         image_array = np.frombuffer(
#             image_bytes,
#             dtype=np.uint8,
#         )
#
#         image = cv2.imdecode(
#             image_array,
#             cv2.IMREAD_COLOR,
#         )
#
#         if image is None:
#             raise ValueError(
#                 "Could not decode image for cropping."
#             )
#
#         image_height, image_width = image.shape[:2]
#
#         # --------------------------------------------------
#         # Read coordinates
#         # --------------------------------------------------
#
#         x_min = int(crop_config["x_min"])
#         y_min = int(crop_config["y_min"])
#         x_max = int(crop_config["x_max"])
#         y_max = int(crop_config["y_max"])
#
#         # --------------------------------------------------
#         # Clamp to image boundaries
#         # --------------------------------------------------
#
#         x_min = max(0, min(x_min, image_width))
#         x_max = max(0, min(x_max, image_width))
#
#         y_min = max(0, min(y_min, image_height))
#         y_max = max(0, min(y_max, image_height))
#
#         if x_max <= x_min or y_max <= y_min:
#             raise ValueError(
#                 f"Invalid crop region for camera_id={camera_id}: "
#                 f"({x_min}, {y_min}, {x_max}, {y_max})"
#             )
#
#         # --------------------------------------------------
#         # Crop ROI
#         # --------------------------------------------------
#
#         cropped = image[
#             y_min:y_max,
#             x_min:x_max,
#         ].copy()
#
#         # --------------------------------------------------
#         # Encode crop
#         # --------------------------------------------------
#
#         success, encoded = cv2.imencode(
#             ".jpg",
#             cropped,
#             [
#                 cv2.IMWRITE_JPEG_QUALITY,
#                 95,
#             ],
#         )
#
#         if not success:
#             raise ValueError(
#                 "Could not encode cropped image."
#             )
#
#         region = BoundingBoxDTO(
#             x_min=x_min,
#             y_min=y_min,
#             x_max=x_max,
#             y_max=y_max,
#         )
#
#         return encoded.tobytes(), region
#
#     @staticmethod
#     def _load_config() -> dict:
#         config_path = (
#             Path(__file__).resolve().parent
#             / "config"
#             / "image_crop_config.yaml"
#         )
#
#         with config_path.open(
#             "r",
#             encoding="utf-8",
#         ) as f:
#             return yaml.safe_load(f) or {}
#
#     def translate_result_to_original(
#             self,
#             result,
#             crop_region: BoundingBoxDTO | None,
#     ):
#         """
#         Translate detection coordinates from ROI coordinates
#         into original-image coordinates.
#
#         crop_box remains in ROI coordinates.
#         box becomes original-image coordinates.
#         """
#
#         if crop_region is None:
#             result.processing_region = None
#             return result
#
#         offset_x = crop_region.x_min
#         offset_y = crop_region.y_min
#
#         for person in result.detections:
#
#             person.box = BoundingBoxDTO(
#                 x_min=person.box.x_min + offset_x,
#                 y_min=person.box.y_min + offset_y,
#                 x_max=person.box.x_max + offset_x,
#                 y_max=person.box.y_max + offset_y,
#             )
#
#             for ppe in person.ppe:
#                 ppe.box = BoundingBoxDTO(
#                     x_min=ppe.box.x_min + offset_x,
#                     y_min=ppe.box.y_min + offset_y,
#                     x_max=ppe.box.x_max + offset_x,
#                     y_max=ppe.box.y_max + offset_y,
#                 )
#
#         result.processing_region = crop_region
#
#         return result


################################################################################
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.dto import (
    PointDTO,
    ProcessingRegionDTO,
    VisionResultDTO,
)


class ImageCropper:
    """
    Crops an image according to a configured polygon for each camera.

    The polygon defines the region in the ORIGINAL image where vision
    inference is allowed to operate.

    The returned crop is rectangular, based on the polygon's bounding
    rectangle, with pixels outside the polygon masked out.

    Detection coordinates produced inside the crop can subsequently be
    translated back to original-image coordinates.
    """

    def __init__(self):
        self._config = self._load_config()

        self._regions = self._config.get("regions", {})

    # ==================================================================
    # Public API
    # ==================================================================

    def crop(
        self,
        image_bytes: bytes,
        camera_id: str,
    ):
        """
        Crop the image according to the polygon configured for camera_id.

        Returns:
            cropped_image_bytes
            crop_region

        crop_region contains:
            x_offset
            y_offset
            polygon
        """

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

        polygon = self._get_polygon(camera_id)

        polygon_array = np.array(
            polygon,
            dtype=np.int32,
        )

        # --------------------------------------------------------------
        # Bounding rectangle around polygon
        # --------------------------------------------------------------

        x, y, width, height = cv2.boundingRect(
            polygon_array
        )

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid crop region for camera_id={camera_id}"
            )

        # --------------------------------------------------------------
        # Crop the bounding rectangle
        # --------------------------------------------------------------

        crop = image[
            y:y + height,
            x:x + width,
        ].copy()

        # --------------------------------------------------------------
        # Translate polygon into crop coordinates
        # --------------------------------------------------------------

        crop_polygon = polygon_array.copy()

        crop_polygon[:, 0] -= x
        crop_polygon[:, 1] -= y

        # --------------------------------------------------------------
        # Mask everything outside polygon
        # --------------------------------------------------------------

        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        cv2.fillPoly(
            mask,
            [crop_polygon],
            255,
        )

        crop[mask == 0] = 0

        # --------------------------------------------------------------
        # Encode cropped image
        # --------------------------------------------------------------

        success, encoded = cv2.imencode(
            ".jpg",
            crop,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if not success:
            raise ValueError(
                "Could not encode cropped image."
            )

        crop_region = {
            "x_offset": x,
            "y_offset": y,
            "polygon": polygon,
        }

        return encoded.tobytes(), crop_region

    # ==================================================================
    # Coordinate translation
    # ==================================================================

    def translate_result_to_original(
            self,
            result: VisionResultDTO,
            crop_region,
    ) -> VisionResultDTO:
        """
        Translate detection coordinates from crop coordinates into
        original-image coordinates.

        Person and PPE boxes are translated.

        crop_box remains in crop coordinates.

        The processing polygon is attached to the result so that
        the renderer can visualize the detection region on the
        original image.
        """

        x_offset = crop_region["x_offset"]
        y_offset = crop_region["y_offset"]

        # --------------------------------------------------------------
        # Translate person and PPE detections
        # --------------------------------------------------------------

        for person in result.detections:

            person.box.x_min += x_offset
            person.box.x_max += x_offset
            person.box.y_min += y_offset
            person.box.y_max += y_offset

            for ppe in person.ppe:
                ppe.box.x_min += x_offset
                ppe.box.x_max += x_offset
                ppe.box.y_min += y_offset
                ppe.box.y_max += y_offset

        # --------------------------------------------------------------
        # Add processing polygon to result
        # --------------------------------------------------------------

        result.processing_region = ProcessingRegionDTO(
            polygon=[
                PointDTO(
                    x=int(x),
                    y=int(y),
                )
                for x, y in crop_region["polygon"]
            ]
        )

        return result

    # ==================================================================
    # Config
    # ==================================================================

    def _get_polygon(
        self,
        camera_id: str,
    ):
        """
        Return the configured polygon for a camera.
        """

        camera_config = self._regions.get(
            str(camera_id)
        )

        if camera_config is None:
            raise ValueError(
                f"No crop region configured for camera_id={camera_id}"
            )

        polygon = camera_config.get("polygon")

        if not polygon:
            raise ValueError(
                f"No polygon configured for camera_id={camera_id}"
            )

        return [
            (
                int(point[0]),
                int(point[1]),
            )
            for point in polygon
        ]

    @staticmethod
    def _load_config():

        config_path = (
            Path(__file__).resolve().parent
            / "config"
            / "image_crop_config.yaml"
        )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return yaml.safe_load(f)