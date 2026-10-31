"""
Crop regions per camera: read and write the cropper's
image_crop_config.yaml, and the image helpers the Streamlit page uses
to mark crops by clicking. Pure functions, no Streamlit.

A camera has any number of crops, each with a label:

    regions:
      "12846":
        crops:
          - label: person detection
            polygon:
              - [7, 1649]
              - [2548, 1684]
          - label: safety
            polygon:
              - [100, 100]
              - [400, 300]
          - label: exclude
            polygon:
              - [0, 0]
              - [6911, 0]
              - [6911, 1649]
              - [0, 1649]

person detection = where people are detected, safety = a safety zone,
priority = a priority zone. These are the drawn crops: the model sees
their union and everything outside them is blacked out (no crop at
all means the whole frame). exclude is never drawn: it is the rest of
the frame, computed from the drawn crops and the frame size on every
save and written as one polygon so the file states what is cut out
(holes and separate pieces are reached through zero-width bridges, so
any polygon fill paints exactly the black area). Coordinates are full-image
pixels. The older form with a single `polygon:` under the camera is
still read as one person detection crop, and the labels `include` and
`person` from earlier saves as `person detection`; the first save
rewrites the file in the current form.

The same file holds the PPE classes switched off per camera; the
engine counts a class listed here as present, so it is never a
violation on that camera. A camera (or the key) that is absent checks
all three:

    disabled_ppe:
      "12846": [boots]
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

CROP_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "processing"
    / "Image_cropper"
    / "config"
    / "image_crop_config.yaml"
)

DISPLAY_WIDTH = 1200

# The labels a crop can be drawn with; exclude is derived on save.
LABELS = ("person detection", "safety", "priority")
DEFAULT_LABEL = "person detection"
EXCLUDE_LABEL = "exclude"
STORED_LABELS = LABELS + (EXCLUDE_LABEL,)

# The labels earlier saves used for the person detection crop.
LEGACY_LABELS = {"include": DEFAULT_LABEL, "person": DEFAULT_LABEL}

# person detection = FINGO green, safety = amber, priority = blue,
# exclude = grey.
LABEL_COLOURS = {
    "person detection": (138, 198, 70),
    "safety": (255, 180, 0),
    "priority": (0, 174, 239),
    "exclude": (120, 120, 120),
}

# The classes compliance is judged on (the classes.ppe keys in
# vision_config.yaml); each can be switched off per camera.
PPE_CLASSES = ("helmet", "vest", "boots")

Point = tuple[int, int]


@dataclass
class Crop:
    """One labelled polygon of a camera. Two points mean a rectangle."""

    label: str = DEFAULT_LABEL
    points: list[Point] = field(default_factory=list)


def _as_point(point) -> Point:
    x, y = point
    return int(round(float(x))), int(round(float(y)))


def _as_label(value, where: str = "") -> str:
    label = str(DEFAULT_LABEL if value is None else value).strip().lower()
    label = LEGACY_LABELS.get(label, label)

    if label not in STORED_LABELS:
        raise ValueError(
            f"Unknown crop label {value!r}{where}; use one of {', '.join(LABELS)}."
        )

    return label


def as_crops(value) -> list[Crop]:
    """
    A list of Crop as it is; a plain list of points as one person crop
    (the form the single-polygon callers use).
    """

    items = list(value or [])

    if not items:
        return []

    if all(isinstance(item, Crop) for item in items):
        return items

    if any(isinstance(item, Crop) for item in items):
        raise TypeError("Pass either crops or points, not both.")

    return [Crop(DEFAULT_LABEL, [_as_point(p) for p in items])]


def drawn_crops(crops) -> list[Crop]:
    """The crops that were drawn: every crop except the derived exclude ones."""

    return [crop for crop in as_crops(crops) if crop.label != EXCLUDE_LABEL]


def _read_camera(camera: str, config: dict) -> list[Crop]:
    if "crops" in config:
        return [
            Crop(
                _as_label((item or {}).get("label"), f" for camera {camera}"),
                [_as_point(p) for p in ((item or {}).get("polygon") or [])],
            )
            for item in (config.get("crops") or [])
        ]

    polygon = config.get("polygon") or []

    return [Crop(DEFAULT_LABEL, [_as_point(p) for p in polygon])] if polygon else []


def load_regions(path: Path | str | None = None) -> dict[str, list[Crop]]:
    """
    Camera id -> crops, from the yaml. Missing file -> {}. A camera
    with the older single `polygon:` is one person crop. An unknown
    label raises ValueError.
    """

    path = Path(path or CROP_CONFIG_PATH)

    if not path.exists():
        return {}

    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    regions = content.get("regions") or {}

    return {
        str(camera): _read_camera(str(camera), config or {})
        for camera, config in regions.items()
    }


def normalise_polygon(points) -> list[Point]:
    """
    Two points are the opposite corners of a rectangle and become its
    four corners; three or more are used as they are. Fewer than two
    raise ValueError.
    """

    polygon = [_as_point(p) for p in points]

    if len(polygon) < 2:
        raise ValueError(
            "Mark at least two points (two = opposite corners of a rectangle)."
        )

    if len(polygon) == 2:
        (ax, ay), (bx, by) = polygon
        x1, x2 = min(ax, bx), max(ax, bx)
        y1, y2 = min(ay, by), max(ay, by)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    return polygon


def clamp_polygon(polygon, width: int, height: int) -> list[Point]:
    return [
        (min(max(x, 0), width - 1), min(max(y, 0), height - 1))
        for x, y in (_as_point(p) for p in polygon)
    ]


def _polygons_of(geometry) -> list:
    """The Polygon parts of a shapely geometry (a Polygon, Multi or collection)."""

    if geometry.is_empty:
        return []

    if geometry.geom_type == "Polygon":
        return [geometry]

    return [part for part in getattr(geometry, "geoms", []) if part.geom_type == "Polygon"]


def _integer_ring(coords) -> list[Point]:
    """A shapely ring as integer points, without repeats or the closing point."""

    points: list[Point] = []

    for x, y in coords:
        point = _as_point((x, y))

        if not points or point != points[-1]:
            points.append(point)

    if len(points) > 1 and points[0] == points[-1]:
        points.pop()

    return points


def _single_ring(polygons) -> list[Point]:
    """
    One ring that traces every polygon given, holes included. The
    first polygon's outline starts the ring; every further outline or
    hole is reached from the nearest vertex of the ring so far through
    a zero-width bridge (there and back). No polygon fill paints a
    bridge, so under the even-odd and the non-zero rule the ring fills
    exactly the polygons minus their holes. Outlines run counter-
    clockwise, holes clockwise.
    """

    rings: list[list[Point]] = []

    for polygon in sorted(polygons, key=lambda polygon: polygon.bounds):
        polygon = orient(polygon, 1.0)
        rings.append(_integer_ring(polygon.exterior.coords))
        rings.extend(_integer_ring(hole.coords) for hole in polygon.interiors)

    rings = [ring for ring in rings if len(ring) >= 3]

    if not rings:
        return []

    result = list(rings[0])

    for ring in rings[1:]:
        at, start = min(
            ((i, j) for i in range(len(result)) for j in range(len(ring))),
            key=lambda ij: math.dist(result[ij[0]], ring[ij[1]]),
        )
        detour = ring[start:] + ring[:start]
        result[at + 1:at + 1] = detour + [detour[0], result[at]]

    return _integer_ring(result)


def exclude_crops(crops, image_size: tuple[int, int]) -> list[Crop]:
    """
    The excluded part of the frame as one exclude crop: the frame
    rectangle minus the union of the drawn crops (two points = a
    rectangle), traced as a single ring with integer coordinates. When
    that part has holes or several pieces the ring reaches them through
    zero-width bridges (see _single_ring), so one polygon still paints
    exactly the black area. Without a drawn crop of two or more points
    the whole frame is used, so nothing is excluded and the list is
    empty; so is a rest smaller than one pixel.
    """

    width, height = image_size

    shapes = [
        Polygon(normalise_polygon(crop.points)).buffer(0)
        for crop in drawn_crops(crops)
        if len(crop.points) >= 2
    ]

    if not shapes:
        return []

    rest = box(0, 0, width - 1, height - 1).difference(unary_union(shapes))

    ring = _single_ring(_polygons_of(rest))

    if len(ring) < 3 or rest.area < 1:
        return []

    return [Crop(EXCLUDE_LABEL, ring)]


def normalise_crops(crops, image_size: tuple[int, int] | None = None) -> list[Crop]:
    """
    The crops as they are saved: labels checked, empty crops dropped,
    two points made a rectangle, points clamped to image_size when
    given. Exclude crops passed in are dropped (they are derived) and,
    when image_size is given, computed again from the drawn crops and
    appended. A drawn crop with a single point, or no drawn crop left,
    raises ValueError. Returns new Crop objects.
    """

    result: list[Crop] = []

    for number, crop in enumerate(as_crops(crops), start=1):
        label = _as_label(crop.label, f" on crop {number}")

        if label == EXCLUDE_LABEL:
            continue

        points = [_as_point(p) for p in crop.points]

        if not points:
            continue

        if len(points) == 1:
            raise ValueError(
                f"Crop {number} ({label}) has a single point; add another "
                "or remove the crop."
            )

        polygon = normalise_polygon(points)

        if image_size:
            polygon = clamp_polygon(polygon, *image_size)

        result.append(Crop(label, polygon))

    if not result:
        raise ValueError("Mark at least one crop with two or more points.")

    if image_size:
        result += exclude_crops(result, image_size)

    return result


def polygon_bbox(polygon) -> tuple[int, int, int, int]:
    """(x_min, y_min, x_max, y_max) of a polygon."""

    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]

    return min(xs), min(ys), max(xs), max(ys)


def nearest_point_index(
    points,
    x: float,
    y: float,
    max_distance: float | None = None,
) -> int | None:
    """
    Index of the point closest to (x, y). None when there are no points
    or the closest one is farther away than max_distance.
    """

    best: int | None = None
    best_distance = math.inf

    for index, (px, py) in enumerate(points):
        distance = math.hypot(px - x, py - y)

        if distance < best_distance:
            best, best_distance = index, distance

    if best is None or (max_distance is not None and best_distance > max_distance):
        return None

    return best


def render_regions_yaml(regions: dict, extra: dict | None = None) -> str:
    """
    The yaml text in the file's style: quoted camera keys, one crop
    per "- label:" entry and one "- [x, y]" row per point. A camera's
    value may be a list of Crop or a plain list of points (one person
    crop). Any other top-level keys are appended with the standard
    dumper.
    """

    lines = ["regions:"]

    for camera, value in regions.items():
        crops = as_crops(value)

        lines.append(f'  "{camera}":')
        lines.append("    crops:" + (" []" if not crops else ""))

        for crop in crops:
            lines.append(f"      - label: {crop.label}")
            lines.append("        polygon:")

            for x, y in crop.points:
                lines.append(f"          - [{x}, {y}]")

        lines.append("")

    text = "\n".join(lines).rstrip("\n") + "\n"

    if extra:
        text += "\n" + yaml.safe_dump(extra, sort_keys=False)

    return text


def save_region(
    camera_id: str,
    crops,
    path: Path | str | None = None,
    image_size: tuple[int, int] | None = None,
) -> list[Crop]:
    """
    Replace (or add) one camera's crops in the yaml, leaving the other
    cameras as they are. `crops` is a list of Crop, or a plain list of
    points for one person crop. Crops are normalised (two points =
    rectangle, empty crops dropped) and clamped to image_size when
    given. Returns the crops that were written.
    """

    path = Path(path or CROP_CONFIG_PATH)

    camera_id = str(camera_id).strip()

    if not camera_id:
        raise ValueError("Camera id is empty.")

    written = normalise_crops(crops, image_size)

    content: dict = {}

    if path.exists():
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    regions = load_regions(path)
    regions[camera_id] = written

    extra = {key: value for key, value in content.items() if key != "regions"}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_regions_yaml(regions, extra), encoding="utf-8", newline="\n")

    return written


def load_disabled_ppe(path: Path | str | None = None) -> dict[str, set[str]]:
    """
    Camera id -> PPE classes switched off for it, from the yaml's
    `disabled_ppe` key. Missing file or key -> {}.
    """

    path = Path(path or CROP_CONFIG_PATH)

    if not path.exists():
        return {}

    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return {
        str(camera): {str(name).strip().lower() for name in names or []}
        for camera, names in (content.get("disabled_ppe") or {}).items()
    }


def save_disabled_ppe(
    camera_id: str,
    disabled,
    path: Path | str | None = None,
) -> None:
    """
    Replace one camera's switched-off PPE classes in the yaml, leaving
    the crops and the other cameras as they are. An empty `disabled`
    removes the camera's entry (all classes checked again).
    """

    path = Path(path or CROP_CONFIG_PATH)

    camera_id = str(camera_id).strip()

    if not camera_id:
        raise ValueError("Camera id is empty.")

    disabled = {str(name).strip().lower() for name in disabled}
    unknown = disabled - set(PPE_CLASSES)

    if unknown:
        raise ValueError(
            f"Unknown PPE class {', '.join(sorted(unknown))}; "
            f"use one of {', '.join(PPE_CLASSES)}."
        )

    content: dict = {}

    if path.exists():
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    extra = {key: value for key, value in content.items() if key != "regions"}

    table = {
        str(camera): sorted(names)
        for camera, names in load_disabled_ppe(path).items()
        if names
    }

    chosen = sorted(name for name in PPE_CLASSES if name in disabled)

    if chosen:
        table[camera_id] = chosen
    else:
        table.pop(camera_id, None)

    if table:
        extra["disabled_ppe"] = table
    else:
        extra.pop("disabled_ppe", None)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_regions_yaml(load_regions(path), extra),
        encoding="utf-8",
        newline="\n",
    )


def fit_for_display(
    width: int,
    height: int,
    max_width: int = DISPLAY_WIDTH,
) -> tuple[int, int, float]:
    """
    (display width, display height, scale) where scale is full-image
    pixels per displayed pixel. Images narrower than max_width are
    shown as they are.
    """

    if width <= max_width:
        return width, height, 1.0

    scale = width / max_width

    return max_width, max(1, int(round(height / scale))), scale


def to_full_image(x: float, y: float, scale: float) -> Point:
    """A click on the displayed image -> full-image pixel position."""

    return int(round(x * scale)), int(round(y * scale))


def display_base(
    image: Image.Image,
    max_width: int = DISPLAY_WIDTH,
) -> tuple[Image.Image, float]:
    """The frame resized for display and its scale; cache this per frame."""

    disp_w, disp_h, scale = fit_for_display(*image.size, max_width)

    return image.convert("RGB").resize((disp_w, disp_h)), scale


def draw_overlay(
    image: Image.Image,
    crops,
    active: int | None = None,
    highlight: int | None = None,
    base: Image.Image | None = None,
    max_width: int = DISPLAY_WIDTH,
    shade: float = 0.0,
) -> tuple[Image.Image, float]:
    """
    The frame resized for display with every crop drawn in its label's
    colour (a two-point crop as the rectangle it will become). The
    crop at index `active` is drawn thicker with its points numbered,
    and `highlight` (an index into its points) rings the point being
    moved; the other crops carry their label as a tag. `shade` (0 to 1)
    darkens everything outside the drawn crops by that much, so the
    picture itself shows what the model will not see. Returns
    (image, scale). Pass `base` (from display_base) to skip resizing
    the full frame again.
    """

    if base is None:
        base, scale = display_base(image, max_width)
    else:
        scale = image.size[0] / base.size[0]

    canvas = base.copy()

    def shrink(polygon):
        return [(x / scale, y / scale) for x, y in polygon]

    if shade:
        drawn = [
            shrink(normalise_polygon(crop.points))
            for crop in drawn_crops(crops)
            if len(crop.points) >= 2
        ]

        if drawn:
            keep = Image.new("L", canvas.size, 0)
            keep_draw = ImageDraw.Draw(keep)

            for polygon in drawn:
                keep_draw.polygon(polygon, fill=255)

            black = Image.new("RGB", canvas.size, (0, 0, 0))
            canvas = Image.composite(canvas, Image.blend(canvas, black, shade), keep)

    draw = ImageDraw.Draw(canvas)

    for index, crop in enumerate(as_crops(crops)):
        colour = LABEL_COLOURS.get(crop.label, LABEL_COLOURS[DEFAULT_LABEL])
        is_active = index == active
        points = [_as_point(p) for p in crop.points]

        if len(points) >= 2:
            outline = shrink(normalise_polygon(points))
            draw.line(outline + [outline[0]], fill=colour, width=3 if is_active else 2)

        if not is_active:
            if points:
                x, y = shrink(points[:1])[0]
                draw.text((x + 8, y - 8), crop.label, fill=colour)

            continue

        for number, (x, y) in enumerate(shrink(points)):
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=colour)
            draw.text((x + 8, y - 8), str(number + 1), fill=colour)

            if number == highlight:
                draw.ellipse(
                    [x - 11, y - 11, x + 11, y + 11], outline=colour, width=3
                )

    return canvas, scale


def crop_preview(image: Image.Image, crops) -> Image.Image:
    """
    What the cropper will feed the model: the bounding box of the drawn
    crops, whatever their label (the whole frame when there is none),
    with everything outside them black. Crops with fewer than two
    points are ignored, and so are exclude crops: the black already
    covers them.
    """

    drawn: list[list[Point]] = [
        normalise_polygon(crop.points)
        for crop in drawn_crops(crops)
        if len(crop.points) >= 2
    ]

    mask = Image.new("L", image.size, 0 if drawn else 255)
    draw = ImageDraw.Draw(mask)

    for polygon in drawn:
        draw.polygon(polygon, fill=255)

    if drawn:
        x1, y1, x2, y2 = polygon_bbox([p for polygon in drawn for p in polygon])
    else:
        x1, y1, x2, y2 = 0, 0, image.size[0] - 1, image.size[1] - 1

    black = Image.new("RGB", image.size, (0, 0, 0))
    composed = Image.composite(image.convert("RGB"), black, mask)

    return composed.crop((x1, y1, x2 + 1, y2 + 1))
