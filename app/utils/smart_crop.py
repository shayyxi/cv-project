"""
Person-centred smart crop shared by the PPE vision engine and the
label-queue export, so exported training crops match what the
PPE model sees at inference time.
"""

import cv2
import numpy as np


def smart_crop(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    crop_width: int,
    crop_height: int,
    scale_padding: float,
) -> tuple[np.ndarray, dict]:
    """
    Fixed-aspect crop centred on the bbox, enlarged by scale_padding
    (at least 1.2x), black-padded where it leaves the image, then
    resized to (crop_width, crop_height).

    Returns (crop, transform); transform maps original-image
    coordinates into crop coordinates:

        crop_x = (x - origin_x) * scale_x
        crop_y = (y - origin_y) * scale_y
    """

    image_height, image_width = image.shape[:2]

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    scale = max(
        bbox_width / crop_width,
        bbox_height / crop_height,
    )

    scale = max(
        scale * scale_padding,
        1.2,
    )

    half_width = int(crop_width * scale) // 2
    half_height = int(crop_height * scale) // 2

    crop_x1 = center_x - half_width
    crop_y1 = center_y - half_height
    crop_x2 = center_x + half_width
    crop_y2 = center_y + half_height

    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - image_width)
    pad_bottom = max(0, crop_y2 - image_height)

    actual_x1 = max(0, crop_x1)
    actual_y1 = max(0, crop_y1)
    actual_x2 = min(image_width, crop_x2)
    actual_y2 = min(image_height, crop_y2)

    crop = image[
        actual_y1:actual_y2,
        actual_x1:actual_x2,
    ].copy()

    if pad_left or pad_top or pad_right or pad_bottom:
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
        (crop_width, crop_height),
        interpolation=cv2.INTER_LANCZOS4,
    )

    transform = {
        "origin_x": crop_x1,
        "origin_y": crop_y1,
        "scale_x": crop_width / padded_width,
        "scale_y": crop_height / padded_height,
    }

    return crop, transform
