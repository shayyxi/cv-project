import shutil

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from app.retraining.regions import (
    CROP_CONFIG_PATH,
    Crop,
    crop_preview,
    draw_overlay,
    exclude_crops,
    fit_for_display,
    load_regions,
    nearest_point_index,
    normalise_crops,
    normalise_polygon,
    polygon_bbox,
    render_regions_yaml,
    save_region,
    to_full_image,
)

GREEN = (138, 198, 70)
AMBER = (255, 180, 0)
BLUE = (0, 174, 239)
GREY = (120, 120, 120)


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


def test_nearest_point_index() -> None:
    points = [(0, 0), (100, 100), (200, 0)]

    assert nearest_point_index(points, 95, 90) == 1
    assert nearest_point_index(points, 190, 5) == 2
    # Too far from every point for the given reach.
    assert nearest_point_index(points, 95, 90, max_distance=5) is None
    assert nearest_point_index(points, 95, 90, max_distance=20) == 1
    assert nearest_point_index([], 10, 10) is None


def test_fit_for_display_and_click_scaling() -> None:
    assert fit_for_display(800, 600) == (800, 600, 1.0)

    disp_w, disp_h, scale = fit_for_display(6912, 3888, max_width=1200)
    assert disp_w == 1200
    assert disp_h == 675
    assert abs(scale - 6912 / 1200) < 1e-9

    # A click at the displayed bottom-right corner maps to the full frame.
    assert to_full_image(1200, 675, scale) == (6912, 3888)


def test_normalise_crops() -> None:
    crops = [
        Crop("person detection", [(100.4, 50), (300, 250)]),
        Crop(" Safety ", [(-5, 10), (20, 10), (20, 5000)]),
        Crop("safety", []),
        # Exclude crops passed in are dropped and computed again.
        Crop("exclude", [(0, 0), (1, 1), (0, 1)]),
    ]

    normalised = normalise_crops(crops, image_size=(400, 300))

    assert normalised[:2] == [
        Crop("person detection", [(100, 50), (300, 50), (300, 250), (100, 250)]),
        Crop("safety", [(0, 10), (20, 10), (20, 299)]),
    ]
    # With the frame size known, the rest of the frame follows as exclude crops.
    assert normalised[2:] == exclude_crops(normalised[:2], (400, 300))
    assert [crop.label for crop in normalised[2:]] == ["exclude"]

    # Without a frame size there is nothing to derive them from.
    assert all(crop.label != "exclude" for crop in normalise_crops(crops))

    # A plain list of points is one person detection crop.
    assert normalise_crops([(0, 0), (10, 10)]) == [
        Crop("person detection", [(0, 0), (10, 0), (10, 10), (0, 10)]),
    ]

    # "include" and "person", the earlier names of the crop, are read as
    # person detection.
    for old in (" Include ", "person"):
        assert normalise_crops([Crop(old, [(0, 0), (10, 10)])]) == [
            Crop("person detection", [(0, 0), (10, 0), (10, 10), (0, 10)]),
        ]

    with pytest.raises(ValueError, match="single point"):
        normalise_crops([Crop("person detection", [(1, 1)])])

    with pytest.raises(ValueError, match="Unknown crop label"):
        normalise_crops([Crop("roi", [(1, 1), (2, 2)])])

    # Exclude crops alone are nothing drawn.
    with pytest.raises(ValueError, match="at least one crop"):
        normalise_crops([Crop("exclude", [(1, 1), (2, 2)])])

    with pytest.raises(ValueError):
        normalise_crops([Crop("person detection", [])])

    with pytest.raises(ValueError):
        normalise_crops([])


def test_load_regions_legacy_polygon_is_one_person_crop(tmp_path) -> None:
    path = tmp_path / "image_crop_config.yaml"
    path.write_text(
        'regions:\n  "7":\n    polygon:\n      - [1, 2]\n      - [3, 4]\n      - [5, 6]\n',
        encoding="utf-8",
    )

    assert load_regions(path) == {"7": [Crop("person detection", [(1, 2), (3, 4), (5, 6)])]}

    # The shipped file reads the same way, whichever form it is in.
    for crops in load_regions(CROP_CONFIG_PATH).values():
        assert any(crop.label == "person detection" and len(crop.points) >= 3 for crop in crops)


def test_load_regions_reads_crops_with_labels(tmp_path) -> None:
    path = tmp_path / "image_crop_config.yaml"
    path.write_text(
        'regions:\n'
        '  "7":\n'
        '    crops:\n'
        '      - label: Include\n'
        '        polygon:\n'
        '          - [1, 2]\n'
        '          - [3, 4]\n'
        '      - label: priority\n'
        '        polygon:\n'
        '          - [5, 6]\n'
        '          - [7, 8]\n'
        '          - [9, 10]\n'
        '      - label: safety\n'
        '        polygon: []\n'
        '  "8":\n'
        '    crops: []\n',
        encoding="utf-8",
    )

    # "Include" is the earlier name of the person crop.
    assert load_regions(path) == {
        "7": [
            Crop("person detection", [(1, 2), (3, 4)]),
            Crop("priority", [(5, 6), (7, 8), (9, 10)]),
            Crop("safety", []),
        ],
        "8": [],
    }

    path.write_text(
        'regions:\n  "7":\n    crops:\n      - label: roi\n        polygon:\n          - [1, 2]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="roi"):
        load_regions(path)

    # Exclude entries (written by save) and the earlier "person" label load too.
    path.write_text(
        'regions:\n  "7":\n    crops:\n      - label: person\n        polygon:\n'
        '          - [1, 2]\n          - [3, 4]\n      - label: exclude\n'
        '        polygon:\n          - [0, 0]\n          - [9, 0]\n          - [9, 9]\n',
        encoding="utf-8",
    )

    assert load_regions(path) == {
        "7": [
            Crop("person detection", [(1, 2), (3, 4)]),
            Crop("exclude", [(0, 0), (9, 0), (9, 9)]),
        ],
    }


def test_render_regions_yaml_crops_layout() -> None:
    text = render_regions_yaml(
        {
            "7": [Crop("person detection", [(1, 2), (3, 4)]), Crop("safety", [(5, 6)])],
            "8": [(0, 0), (9, 9)],
            "9": [],
        },
        extra={"note": "x"},
    )

    assert text == (
        'regions:\n'
        '  "7":\n'
        '    crops:\n'
        '      - label: person detection\n'
        '        polygon:\n'
        '          - [1, 2]\n'
        '          - [3, 4]\n'
        '      - label: safety\n'
        '        polygon:\n'
        '          - [5, 6]\n'
        '\n'
        '  "8":\n'
        '    crops:\n'
        '      - label: person detection\n'
        '        polygon:\n'
        '          - [0, 0]\n'
        '          - [9, 9]\n'
        '\n'
        '  "9":\n'
        '    crops: []\n'
        '\n'
        'note: x\n'
    )

    content = yaml.safe_load(text)
    assert content["regions"]["7"]["crops"][1] == {"label": "safety", "polygon": [[5, 6]]}
    assert content["regions"]["9"] == {"crops": []}
    assert content["note"] == "x"


def test_save_region_replaces_one_camera_and_keeps_the_rest(tmp_path) -> None:
    path = tmp_path / "image_crop_config.yaml"
    shutil.copy(CROP_CONFIG_PATH, path)

    before = load_regions(path)
    assert set(before) == {"12846", "12875", "12990"}

    written = save_region("12875", [(100, 200), (3000, 2500)], path=path)

    assert written == [
        Crop("person detection", [(100, 200), (3000, 200), (3000, 2500), (100, 2500)]),
    ]

    after = load_regions(path)
    assert after["12875"] == written
    assert after["12846"] == before["12846"]
    assert after["12990"] == before["12990"]

    # The crops form, in the file's own style, and no legacy key left.
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert content["regions"]["12875"]["crops"] == [
        {"label": "person detection", "polygon": [list(p) for p in written[0].points]},
    ]
    assert "polygon" not in content["regions"]["12875"]
    text = path.read_text(encoding="utf-8")
    assert (
        '  "12875":\n    crops:\n      - label: person detection\n        polygon:\n'
        '          - [100, 200]\n'
    ) in text


def test_save_region_writes_several_crops(tmp_path) -> None:
    path = tmp_path / "image_crop_config.yaml"
    shutil.copy(CROP_CONFIG_PATH, path)

    crops = [
        Crop("person detection", [(0, 0), (100, 100)]),
        Crop("safety", [(10, 10), (20, 10), (20, 20)]),
        Crop("safety", [(50, 50), (60, 60)]),
        Crop("safety", []),
    ]

    written = save_region("55555", crops, path=path, image_size=(200, 200))

    # The drawn crops, then the rest of the frame as one exclude crop.
    assert [crop.label for crop in written] == [
        "person detection", "safety", "safety", "exclude",
    ]

    regions = load_regions(path)
    assert regions["55555"] == written

    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert content["regions"]["55555"]["crops"][1] == {
        "label": "safety", "polygon": [[10, 10], [20, 10], [20, 20]],
    }
    assert content["regions"]["55555"]["crops"][3]["label"] == "exclude"
    # Every camera is now in the crops form.
    for config in content["regions"].values():
        assert "crops" in config and "polygon" not in config


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
    # The clamped crop, then the rest of the frame as one exclude crop.
    assert regions["99999"][0] == Crop("person detection", [(0, 10), (6911, 10), (6911, 3887)])
    assert [crop.label for crop in regions["99999"][1:]] == ["exclude"]

    with pytest.raises(ValueError):
        save_region("", [(1, 1), (2, 2)], path=path)


def test_save_region_into_missing_file(tmp_path) -> None:
    path = tmp_path / "new" / "image_crop_config.yaml"

    save_region("1", [(0, 0), (10, 10)], path=path)

    assert load_regions(path) == {
        "1": [Crop("person detection", [(0, 0), (10, 0), (10, 10), (0, 10)])],
    }


def test_draw_overlay_resizes_and_returns_scale() -> None:
    frame = Image.new("RGB", (2400, 1200), (90, 90, 90))
    crops = [
        Crop("person detection", [(100, 100), (500, 500)]),
        Crop("safety", [(1000, 200), (1400, 600)]),
    ]

    canvas, scale = draw_overlay(frame, crops, active=0)

    assert canvas.size == (1200, 600)
    assert scale == 2.0
    # The active crop's points are dots in its label colour.
    assert canvas.getpixel((50, 50)) == GREEN
    # The other crop is outlined in its own colour (left edge of its rectangle).
    assert canvas.getpixel((500, 200)) == AMBER
    # Inside a crop the frame shows through.
    assert canvas.getpixel((150, 150)) == (90, 90, 90)


def test_draw_overlay_shades_outside_the_drawn_crops() -> None:
    frame = Image.new("RGB", (1200, 600), (200, 200, 200))
    crops = [
        Crop("person detection", [(100, 100), (500, 500)]),
        # The derived exclude crop is not a drawn one: it does not keep
        # anything bright.
        Crop("exclude", [(0, 0), (1199, 599)]),
    ]

    canvas, _ = draw_overlay(frame, crops, active=0, shade=0.5)

    # Inside the crop the frame shows as it is; outside it is half dark.
    assert canvas.getpixel((300, 300)) == (200, 200, 200)
    assert canvas.getpixel((800, 300)) == (100, 100, 100)

    # Without a drawn crop nothing is darkened (the whole frame is used).
    plain, _ = draw_overlay(frame, [Crop("person detection", [(1, 1)])], active=0, shade=0.5)
    assert plain.getpixel((800, 300)) == (200, 200, 200)


def test_draw_overlay_colours_follow_the_label() -> None:
    frame = Image.new("RGB", (1200, 600), (90, 90, 90))

    for label, colour in [
        ("person detection", GREEN), ("safety", AMBER), ("priority", BLUE), ("exclude", GREY),
    ]:
        canvas, _ = draw_overlay(frame, [Crop(label, [(100, 100), (500, 500)])], active=0)

        assert canvas.getpixel((100, 100)) == colour


def test_draw_overlay_rings_the_highlighted_point() -> None:
    frame = Image.new("RGB", (2400, 1200), (90, 90, 90))
    crops = [Crop("safety", [(100, 100), (500, 100), (500, 500)])]

    plain, _ = draw_overlay(frame, crops, active=0)
    ringed, _ = draw_overlay(frame, crops, active=0, highlight=0)

    # 10 px below the first point (shown at 50, 50): only the ring is there.
    assert plain.getpixel((50, 60)) == (90, 90, 90)
    assert ringed.getpixel((50, 60)) == AMBER
    # The other points are drawn the same either way.
    assert ringed.getpixel((250, 260)) == plain.getpixel((250, 260))


def test_crop_preview_is_the_person_bbox_with_outside_black() -> None:
    frame = Image.new("RGB", (400, 300), (200, 200, 200))

    preview = crop_preview(frame, [Crop("person detection", [(100, 50), (300, 50), (200, 250)])])

    assert preview.size == (201, 201)
    # Inside the triangle keeps the frame colour; a corner outside is black.
    assert preview.getpixel((100, 20)) == (200, 200, 200)
    assert preview.getpixel((0, 200)) == (0, 0, 0)


def test_crop_preview_is_the_union_of_the_drawn_crops() -> None:
    frame = Image.new("RGB", (400, 300), (200, 200, 200))
    crops = [
        Crop("person detection", [(100, 50), (300, 250)]),
        # A priority zone outside the person crop is seen too; a crop
        # still being started is not.
        Crop("priority", [(320, 20), (380, 60)]),
        Crop("person detection", [(1, 1)]),
    ]

    preview = crop_preview(frame, crops)

    # The bounding box of both crops: (100, 20) to (380, 250).
    assert preview.size == (281, 231)
    # Full-frame (120, 60) is inside the person crop.
    assert preview.getpixel((20, 40)) == (200, 200, 200)
    # Full-frame (350, 40) is inside the priority crop.
    assert preview.getpixel((250, 20)) == (200, 200, 200)
    # Full-frame (310, 200) and (120, 30) are in neither: black.
    assert preview.getpixel((210, 180)) == (0, 0, 0)
    assert preview.getpixel((20, 10)) == (0, 0, 0)


def test_crop_preview_with_a_safety_crop_only_is_that_crop() -> None:
    frame = Image.new("RGB", (400, 300), (200, 200, 200))

    preview = crop_preview(frame, [Crop("safety", [(100, 100), (200, 200)])])

    assert preview.size == (101, 101)
    assert preview.getpixel((0, 0)) == (200, 200, 200)
    assert preview.getpixel((100, 100)) == (200, 200, 200)


def test_crop_preview_without_any_crop_is_the_whole_frame() -> None:
    frame = Image.new("RGB", (400, 300), (200, 200, 200))

    preview = crop_preview(frame, [Crop("person detection", [(1, 1)])])

    assert preview.size == (400, 300)
    assert preview.getpixel((10, 10)) == (200, 200, 200)
    assert preview.getpixel((300, 200)) == (200, 200, 200)


def _coverage(crops, size) -> tuple[int, int]:
    """(pixels in neither a drawn nor an exclude crop, pixels in both)."""

    drawn = Image.new("1", size, 0)
    excluded = Image.new("1", size, 0)

    for crop in crops:
        target = excluded if crop.label == "exclude" else drawn
        ImageDraw.Draw(target).polygon(normalise_polygon(crop.points), fill=1)

    neither = both = 0

    for a, b in zip(drawn.getdata(), excluded.getdata()):
        neither += not a and not b
        both += bool(a and b)

    return neither, both


def _coverage_cv2(crops, size) -> tuple[int, int]:
    """As _coverage, filled with OpenCV, the pipeline's rasteriser."""

    width, height = size
    drawn = np.zeros((height, width), np.uint8)
    excluded = np.zeros((height, width), np.uint8)

    for crop in crops:
        target = excluded if crop.label == "exclude" else drawn
        cv2.fillPoly(target, [np.array(normalise_polygon(crop.points), np.int32)], 1)

    neither = int((~drawn.astype(bool) & ~excluded.astype(bool)).sum())

    return neither, int((drawn & excluded).sum())


def test_exclude_crops_of_an_edge_touching_crop_is_one_polygon() -> None:
    # The floor from edge to edge, as the live cameras: the sky above is
    # one polygon.
    crops = [
        Crop("person detection", [(0, 150), (200, 120), (399, 160), (399, 299), (0, 299)]),
    ]

    excluded = exclude_crops(crops, (400, 300))

    assert [crop.label for crop in excluded] == ["exclude"]
    assert len(excluded[0].points) == 5
    assert set(excluded[0].points) == {(0, 0), (399, 0), (0, 150), (200, 120), (399, 160)}

    # Together they cover the frame; they only share their boundary.
    neither, both = _coverage(crops + excluded, (400, 300))
    assert neither == 0
    assert both < 2 * (400 + 300)


def test_exclude_crops_of_a_floating_crop_is_one_ring_with_a_bridge() -> None:
    crops = [Crop("person detection", [(100, 100), (200, 200)])]

    excluded = exclude_crops(crops, (400, 300))

    # Still one polygon: the frame outline, then the crop reached through
    # a bridge from the nearest corner, traced, and the bridge walked
    # back, so its two ends appear twice (4 + 4 + 2 points).
    assert len(excluded) == 1
    points = excluded[0].points
    assert len(points) == 10
    assert set(points) == {
        (0, 0), (399, 0), (399, 299), (0, 299),
        (100, 100), (200, 100), (200, 200), (100, 200),
    }
    assert points.count((0, 299)) == 2 and points.count((100, 200)) == 2

    # Either rasteriser fills it to exactly the black area.
    for rasterise in (_coverage, _coverage_cv2):
        neither, both = rasterise(crops + excluded, (400, 300))
        assert neither == 0
        assert both < 2 * (400 + 300)


def test_exclude_crops_of_a_band_joins_both_pieces_in_one_ring() -> None:
    # A crop from edge to edge leaves a piece above and one below; the
    # ring walks from one to the other along the frame edge and back.
    crops = [Crop("person detection", [(0, 100), (399, 200)])]

    excluded = exclude_crops(crops, (400, 300))

    assert len(excluded) == 1
    assert len(excluded[0].points) == 10

    for rasterise in (_coverage, _coverage_cv2):
        neither, both = rasterise(crops + excluded, (400, 300))
        assert neither == 0
        assert both < 2 * (400 + 300)


def test_exclude_crops_without_a_drawn_crop_is_empty() -> None:
    assert exclude_crops([], (400, 300)) == []
    assert exclude_crops([Crop("person detection", [(1, 1)])], (400, 300)) == []
    # Exclude crops passed in do not count as drawn.
    assert exclude_crops([Crop("exclude", [(0, 0), (10, 10)])], (400, 300)) == []
