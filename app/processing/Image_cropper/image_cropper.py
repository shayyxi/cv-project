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
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.dto import (
    PointDTO,
    ProcessingRegionDTO,
    VisionResultDTO,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "config"
    / "image_crop_config.yaml"
)


class ImageCropper:
    """
    Crops an image according to the labelled crops configured for each
    camera (see app.retraining.regions for the yaml).

    The drawn crops (`person detection`, `safety`, `priority`; `person`
    and `include` from earlier saves count as `person detection`)
    define where in the ORIGINAL image vision inference is allowed to
    operate; with none configured the whole frame is used. Everything
    outside them is blacked out. The `exclude` entries the page writes
    describe that outside for the file's readers and are ignored here.

    The returned crop is rectangular, the bounding rectangle of the
    drawn crops, with pixels outside them masked out.

    Detection coordinates produced inside the crop can subsequently be
    translated back to original-image coordinates.
    """

    def __init__(self):
        self._config_path = CONFIG_PATH

        self._config = self._load_config(self._config_path) or {}

        self._regions = self._config.get("regions", {}) or {}

        self._config_mtime = self._config_path.stat().st_mtime

    def _reload_config_if_changed(self) -> None:
        """
        Re-read image_crop_config.yaml when its modification time has
        changed, so a polygon saved from the retraining page is used
        for the next frame without a restart. An unreadable or invalid
        file (for example, half-written) keeps the last good config.
        """

        try:
            mtime = self._config_path.stat().st_mtime
        except OSError:
            return

        if mtime == self._config_mtime:
            return

        try:
            config = self._load_config(self._config_path) or {}
        except (OSError, yaml.YAMLError):
            logger.warning(
                "Could not reload %s - keeping the previous crop regions.",
                self._config_path,
            )
            return

        self._config = config
        self._regions = config.get("regions", {}) or {}
        self._config_mtime = mtime

        logger.info(
            "Crop regions reloaded from %s (%d cameras).",
            self._config_path,
            len(self._regions),
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def crop(
        self,
        image_bytes: bytes,
        camera_id: str,
    ):
        """
        Crop the image according to the crops configured for camera_id.

        Returns:
            cropped_image_bytes
            crop_region

        crop_region contains:
            x_offset
            y_offset
            polygon (the first include crop, or the whole frame)
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

        drawn = self._get_crops(camera_id)

        image_height, image_width = image.shape[:2]

        # --------------------------------------------------------------
        # Bounding rectangle around the drawn crops (the whole frame
        # when there is none)
        # --------------------------------------------------------------

        if drawn:
            x, y, width, height = cv2.boundingRect(
                np.array(
                    [point for polygon in drawn for point in polygon],
                    dtype=np.int32,
                )
            )

            x, y = max(x, 0), max(y, 0)
            width = min(width, image_width - x)
            height = min(height, image_height - y)
        else:
            x, y, width, height = 0, 0, image_width, image_height

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
        # Mask: inside an include crop and outside every exclude crop
        # --------------------------------------------------------------

        def in_crop_coordinates(polygons):
            return [
                (np.array(polygon, dtype=np.int32) - (x, y)).astype(np.int32)
                for polygon in polygons
            ]

        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        # One fillPoly call per polygon: a single call with several
        # polygons fills by the even-odd rule, which would punch a hole
        # where a safety or priority zone lies inside the person crop.
        if drawn:
            for polygon in in_crop_coordinates(drawn):
                cv2.fillPoly(mask, [polygon], 255)
        else:
            mask[:] = 255

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

        # The outline drawn on the delivered image: the first person
        # crop (person crops come first), or the whole frame.
        if drawn:
            polygon = drawn[0]
        else:
            polygon = [
                (x, y),
                (x + width - 1, y),
                (x + width - 1, y + height - 1),
                (x, y + height - 1),
            ]

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

    def _get_crops(
        self,
        camera_id: str,
    ) -> list[list[tuple[int, int]]]:
        """
        The drawn polygons of a camera, from the current contents of
        the config file: the person detection, safety and priority
        crops (`person` and `include` from earlier saves count as
        person detection), person detection crops first so the outline
        is one of them. Reads the camera's `crops` list; a camera with
        only the older single `polygon` key is one person detection
        crop. Exclude entries, and any polygon with fewer than three
        points, have no effect here.
        """

        self._reload_config_if_changed()

        camera_config = self._regions.get(
            str(camera_id)
        )

        if camera_config is None:
            raise ValueError(
                f"No crop region configured for camera_id={camera_id}"
            )

        crops = camera_config.get("crops")

        if crops is None:
            polygon = camera_config.get("polygon")
            crops = [{"label": "person detection", "polygon": polygon}] if polygon else []

        persons: list[list[tuple[int, int]]] = []
        zones: list[list[tuple[int, int]]] = []

        for item in crops or []:
            item = item or {}

            polygon = [
                (int(point[0]), int(point[1]))
                for point in (item.get("polygon") or [])
            ]

            if len(polygon) < 3:
                continue

            label = str(item.get("label", "person detection")).strip().lower()

            if label in ("person detection", "person", "include"):
                persons.append(polygon)
            elif label in ("safety", "priority"):
                zones.append(polygon)

        return persons + zones

    @staticmethod
    def _load_config(config_path: Path | None = None):

        config_path = config_path or CONFIG_PATH

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return yaml.safe_load(f)