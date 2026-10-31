import itertools
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.processing.Image_cropper.image_cropper import CONFIG_PATH, ImageCropper
from app.retraining.regions import (
    Crop,
    as_crops,
    normalise_polygon,
    render_regions_yaml,
    save_region,
)


_BUMPS = itertools.count(1)


def _bump_mtime(path: Path) -> None:
    """
    Make sure the change is visible even within the same second. Each
    bump moves the mtime further ahead, so two writes in one second do
    not land on the same value (the cropper reloads on any change).
    """

    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 2 * next(_BUMPS)))


def _write(cropper: ImageCropper, regions: dict) -> None:
    # Two-point crops become rectangles, as the page writes them. Labels
    # are not checked here, so exclude entries can be written as the
    # page does, without computing them.
    normalised = {
        camera: [Crop(crop.label, normalise_polygon(crop.points)) for crop in as_crops(crops)]
        for camera, crops in regions.items()
    }
    cropper._config_path.write_text(render_regions_yaml(normalised), encoding="utf-8")
    _bump_mtime(cropper._config_path)


def _white_image(width: int = 100, height: int = 80) -> bytes:
    success, encoded = cv2.imencode(".png", np.full((height, width, 3), 255, np.uint8))
    assert success
    return encoded.tobytes()


def _decode(image_bytes: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)


@pytest.fixture()
def cropper(tmp_path: Path) -> ImageCropper:
    config = tmp_path / "image_crop_config.yaml"
    shutil.copy(CONFIG_PATH, config)

    cropper = ImageCropper()
    # Point the instance at the scratch copy, as if it were the real file.
    cropper._config_path = config
    cropper._config_mtime = None  # force one reload from the scratch copy

    return cropper


def test_crops_follow_changes_to_the_config_file(cropper: ImageCropper) -> None:
    drawn = cropper._get_crops("12846")
    assert drawn[0][0] == (7, 1649)

    save_region("12846", [(100, 200), (3000, 2500)], path=cropper._config_path)
    _bump_mtime(cropper._config_path)

    assert cropper._get_crops("12846") == [
        [(100, 200), (3000, 200), (3000, 2500), (100, 2500)],
    ]

    # Other cameras are still there, and a new one appears without a restart.
    assert cropper._get_crops("12990")[0][0] == (24, 2487)

    # Exclude entries are in the file for its readers, not for the cropper.
    _write(
        cropper,
        {
            "55555": [
                Crop("person detection", [(0, 0), (10, 10)]),
                Crop("exclude", [(10, 0), (99, 79)]),
            ],
        },
    )

    assert cropper._get_crops("55555") == [[(0, 0), (10, 0), (10, 10), (0, 10)]]


def test_unchanged_file_is_not_reread(cropper: ImageCropper, monkeypatch) -> None:
    cropper._get_crops("12846")

    calls = []
    monkeypatch.setattr(
        ImageCropper,
        "_load_config",
        staticmethod(lambda path=None: calls.append(path) or {"regions": {}}),
    )

    cropper._get_crops("12846")
    assert calls == []


def test_broken_file_keeps_the_last_good_config(cropper: ImageCropper) -> None:
    good = cropper._get_crops("12875")

    cropper._config_path.write_text("regions: [not: valid: yaml", encoding="utf-8")
    _bump_mtime(cropper._config_path)

    assert cropper._get_crops("12875") == good

    # Once the file is valid again it is picked up.
    _write(cropper, {"12875": [(1, 1), (9, 1), (9, 9), (1, 9)]})

    assert cropper._get_crops("12875") == [[(1, 1), (9, 1), (9, 9), (1, 9)]]


def test_unknown_camera_raises(cropper: ImageCropper) -> None:
    with pytest.raises(ValueError, match="camera_id=404"):
        cropper.crop(_white_image(), "404")


def test_crop_is_the_union_of_the_drawn_crops(cropper: ImageCropper) -> None:
    person = [(10, 10), (50, 10), (50, 40), (10, 40)]
    _write(
        cropper,
        {
            "1": [
                # A zone listed before the person crop: the outline is
                # still the person crop.
                Crop("safety", [(15, 15), (25, 25)]),
                Crop("person detection", person),
                Crop("priority", [(70, 60), (90, 75)]),
            ],
        },
    )

    cropped_bytes, region = cropper.crop(_white_image(), "1")
    cropped = _decode(cropped_bytes)

    # The bounding box of every drawn crop, offset into the frame.
    assert cropped.shape == (66, 81, 3)
    assert (region["x_offset"], region["y_offset"]) == (10, 10)
    assert region["polygon"] == person

    # Inside the person, safety and priority crops stays white; between
    # them is black (JPEG leaves a little noise).
    assert cropped[15, 20].min() > 200
    assert cropped[10, 10].min() > 200
    assert cropped[58, 70].min() > 200
    assert cropped[40, 50].max() < 30


def test_exclude_entries_are_ignored(cropper: ImageCropper) -> None:
    # The page writes the rest of the frame as exclude entries; the mask
    # already blacks that part out, so they change nothing here (even one
    # covering the whole frame).
    person = [(10, 10), (50, 10), (50, 40), (10, 40)]
    _write(
        cropper,
        {"1": [Crop("person detection", person), Crop("exclude", [(0, 0), (99, 79)])]},
    )

    cropped_bytes, region = cropper.crop(_white_image(), "1")
    cropped = _decode(cropped_bytes)

    assert cropped.shape == (31, 41, 3)
    assert (region["x_offset"], region["y_offset"]) == (10, 10)
    assert region["polygon"] == person
    assert cropped[15, 20].min() > 200


def test_crop_without_any_crop_is_the_whole_frame(cropper: ImageCropper) -> None:
    _write(cropper, {"1": []})

    cropped_bytes, region = cropper.crop(_white_image(), "1")
    cropped = _decode(cropped_bytes)

    assert cropped.shape == (80, 100, 3)
    assert (region["x_offset"], region["y_offset"]) == (0, 0)
    assert region["polygon"] == [(0, 0), (99, 0), (99, 79), (0, 79)]
    assert cropped[10, 10].min() > 200
    assert cropped[70, 90].min() > 200


def test_legacy_polygon_entry_still_crops(cropper: ImageCropper) -> None:
    cropper._config_path.write_text(
        'regions:\n  "1":\n    polygon:\n      - [10, 10]\n      - [60, 10]\n'
        '      - [60, 50]\n      - [10, 50]\n',
        encoding="utf-8",
    )
    _bump_mtime(cropper._config_path)

    cropped_bytes, region = cropper.crop(_white_image(), "1")

    assert _decode(cropped_bytes).shape == (41, 51, 3)
    assert region["polygon"] == [(10, 10), (60, 10), (60, 50), (10, 50)]
