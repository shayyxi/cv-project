import cv2
import numpy as np

from app.dto import (
    BoundingBoxDTO,
    ComplianceDTO,
    PointDTO,
    VisionDetectionDTO,
    VisionResultDTO,
)
from app.processing.cv.vision_renderer import VisionRenderer
from app.processing.Image_cropper.image_cropper import ImageCropper


CANVAS = 300

# Box and a diamond inscribed in it. The box's bottom-left corner
# (40, 260) lies well outside the diamond, and the diamond's bottom
# vertex (100, 260) lies on its outline. Both probes sit far from the
# person label, which is anchored above the box's top-left corner.
BOX = BoundingBoxDTO(x_min=40, y_min=40, x_max=160, y_max=260)

DIAMOND = [
    PointDTO(x=100, y=40),
    PointDTO(x=160, y=150),
    PointDTO(x=100, y=260),
    PointDTO(x=40, y=150),
]

BOX_CORNER = (260, 40)      # (row, col) of the box's bottom-left corner
DIAMOND_CENTRE = (150, 100)
DIAMOND_BOTTOM = (258, 100)  # just inside the bottom vertex, on the outline


def _person(mask=None, compliant=True):
    return VisionDetectionDTO(
        label="person",
        confidence=0.9,
        box=BOX.model_copy(),
        mask=mask,
        is_sensitive=False,
        person_id=0,
        compliance=ComplianceDTO(
            helmet=compliant,
            vest=compliant,
            boots=compliant,
            compliant=compliant,
        ),
    )


def _render(result) -> np.ndarray:
    blank = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)

    ok, encoded = cv2.imencode(".png", blank)
    assert ok

    rendered = VisionRenderer().draw_original(
        encoded.tobytes(),
        result,
    )

    return cv2.imdecode(
        np.frombuffer(rendered, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def _bgr(image, probe):
    row, col = probe
    return [int(v) for v in image[row, col]]


def test_masked_person_is_drawn_as_silhouette_not_box():
    result = VisionResultDTO(
        worker_count=1,
        detections=[_person(mask=[DIAMOND])],
    )

    image = _render(result)

    # Translucent green tint inside the diamond (compliant colour is
    # pure green, blended at MASK_FILL_ALPHA onto black).
    b, g, r = _bgr(image, DIAMOND_CENTRE)
    assert g > 40, (b, g, r)
    assert b < 30 and r < 30, (b, g, r)

    # Solid green outline on the diamond edge.
    b, g, r = _bgr(image, DIAMOND_BOTTOM)
    assert g > 150, (b, g, r)

    # No rectangle: the box corner outside the diamond stays black.
    b, g, r = _bgr(image, BOX_CORNER)
    assert max(b, g, r) < 40, (b, g, r)


def test_non_compliant_silhouette_uses_non_compliant_colour():
    result = VisionResultDTO(
        worker_count=1,
        detections=[_person(mask=[DIAMOND], compliant=False)],
    )

    image = _render(result)

    # Non-compliant colour is pure red in BGR.
    b, g, r = _bgr(image, DIAMOND_BOTTOM)
    assert r > 150, (b, g, r)
    assert g < 60 and b < 60, (b, g, r)


def test_person_without_mask_still_gets_rectangle():
    result = VisionResultDTO(
        worker_count=1,
        detections=[_person(mask=None)],
    )

    image = _render(result)

    b, g, r = _bgr(image, BOX_CORNER)
    assert g > 150, (b, g, r)


def test_translate_result_shifts_mask_with_box():
    polygon = [
        PointDTO(x=1, y=2),
        PointDTO(x=3, y=4),
        PointDTO(x=5, y=6),
    ]

    result = VisionResultDTO(
        worker_count=1,
        detections=[_person(mask=[polygon])],
    )

    ImageCropper().translate_result_to_original(
        result,
        {
            "x_offset": 10,
            "y_offset": 20,
            "polygon": [(0, 0), (1, 0), (1, 1)],
        },
    )

    person = result.detections[0]

    assert person.box.x_min == BOX.x_min + 10
    assert person.box.y_min == BOX.y_min + 20

    assert [(p.x, p.y) for p in person.mask[0]] == [
        (11, 22),
        (13, 24),
        (15, 26),
    ]


def test_translate_result_tolerates_missing_mask():
    result = VisionResultDTO(
        worker_count=1,
        detections=[_person(mask=None)],
    )

    ImageCropper().translate_result_to_original(
        result,
        {
            "x_offset": 10,
            "y_offset": 20,
            "polygon": [(0, 0), (1, 0), (1, 1)],
        },
    )

    assert result.detections[0].mask is None
    assert result.detections[0].box.x_min == BOX.x_min + 10
