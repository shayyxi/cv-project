import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from app.retraining.regions import (
    CROP_CONFIG_PATH,
    draw_overlay,
    fit_for_display,
    load_regions,
    normalise_polygon,
    polygon_bbox,
    polygon_preview,
    save_region,
    to_full_image,
)


def test_normalise_polygon() -> None:
    assert normalise_polygon([(500.4, 1200), (5800, 3900)]) == [
        (500, 1200), (5800, 1200), (5800, 3900), (500, 3900),
    ]
    # Order of the two corners does not matter.
    assert normalise_polygon([(5800, 3900), (500, 1200)]) == [
        (500, 1200), (5800, 1200), (5800, 3900), (500, 3900),
    ]
    assert normalise_polygon([(1, 2), (3, 4), (5, 6)]) == [(1, 2), (3, 4), (5, 6)]

    with pytest.raises(ValueError):
        normalise_polygon([(1, 2)])


def test_polygon_bbox() -> None:
    assert polygon_bbox([(10, 50), (200, 20), (150, 300)]) == (10, 20, 200, 300)


def test_fit_for_display_and_click_scaling() -> None:
    assert fit_for_display(800, 600) == (800, 600, 1.0)

    disp_w, disp_h, scale = fit_for_display(6912, 3888, max_width=1200)
    assert disp_w == 1200
    assert disp_h == 675
    assert abs(scale - 6912 / 1200) < 1e-9

    # A click at the displayed bottom-right corner maps to the full frame.
    assert to_full_image(1200, 675, scale) == (6912, 3888)


def test_save_region_replaces_one_camera_and_keeps_the_rest(tmp_path) -> None:
    path = tmp_path / "image_crop_config.yaml"
    shutil.copy(CROP_CONFIG_PATH, path)

    before = load_regions(path)
    assert set(before) == {"12846", "12875", "12990"}

    written = save_region("12875", [(100, 200), (3000, 2500)], path=path)

    assert written == [(100, 200), (3000, 200), (3000, 2500), (100, 2500)]

    after = load_regions(path)
    assert after["12875"] == written
    assert after["12846"] == before["12846"]
    assert after["12990"] == before["12990"]

    # Still the cropper's shape, and the file's own style.
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert content["regions"]["12875"]["polygon"] == [list(p) for p in written]
    text = path.read_text(encoding="utf-8")
    assert '  "12875":\n    polygon:\n      - [100, 200]\n' in text


def test_save_region_adds_a_camera_and_clamps_to_the_image(tmp_path) -> None:
    path = tmp_path / "image_crop_config.yaml"
    shutil.copy(CROP_CONFIG_PATH, path)

    save_region(
        " 99999 ",
        [(-5, 10), (7000, 10), (7000, 5000)],
        path=path,
        image_size=(6912, 3888),
    )

    regions = load_regions(path)
    assert len(regions) == 4
    assert regions["99999"] == [(0, 10), (6911, 10), (6911, 3887)]

    with pytest.raises(ValueError):
        save_region("", [(1, 1), (2, 2)], path=path)


def test_save_region_into_missing_file(tmp_path) -> None:
    path = tmp_path / "new" / "image_crop_config.yaml"

    save_region("1", [(0, 0), (10, 10)], path=path)

    assert load_regions(path) == {"1": [(0, 0), (10, 0), (10, 10), (0, 10)]}


def test_draw_overlay_resizes_and_returns_scale() -> None:
    frame = Image.new("RGB", (2400, 1200), (90, 90, 90))

    canvas, scale = draw_overlay(frame, [(0, 0), (2400, 0), (2400, 1200)], [(100, 100), (500, 500)])

    assert canvas.size == (1200, 600)
    assert scale == 2.0
    # A marked point leaves red pixels where it was drawn.
    assert canvas.getpixel((50, 50)) == (255, 59, 48)


def test_polygon_preview_is_the_bbox_with_outside_black() -> None:
    frame = Image.new("RGB", (400, 300), (200, 200, 200))

    preview = polygon_preview(frame, [(100, 50), (300, 50), (200, 250)])

    assert preview.size == (201, 201)
    # Inside the triangle keeps the frame colour; a corner outside is black.
    assert preview.getpixel((100, 20)) == (200, 200, 200)
    assert preview.getpixel((0, 200)) == (0, 0, 0)
