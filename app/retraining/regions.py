"""
Crop-region polygons per camera: read and write the cropper's
image_crop_config.yaml, and the image helpers the Streamlit page uses
to mark a region by clicking. Pure functions, no Streamlit.

The yaml looks like:

    regions:
      "12846":
        polygon:
          - [7, 1649]
          - [2548, 1684]

Coordinates are full-image pixels. ImageCropper masks everything
outside the polygon (cv2.fillPoly closes the shape itself).
"""

from pathlib import Path

import yaml
from PIL import Image, ImageDraw

CROP_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "processing"
    / "Image_cropper"
    / "config"
    / "image_crop_config.yaml"
)

DISPLAY_WIDTH = 1200

SAVED_COLOUR = (31, 110, 235)   # blue: polygon already in the config
MARK_COLOUR = (255, 59, 48)     # red: points being marked now

Point = tuple[int, int]


def _as_point(point) -> Point:
    x, y = point
    return int(round(float(x))), int(round(float(y)))


def load_regions(path: Path | str | None = None) -> dict[str, list[Point]]:
    """Camera id -> polygon, from the yaml. Missing file -> {}."""

    path = Path(path or CROP_CONFIG_PATH)

    if not path.exists():
        return {}

    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    regions = content.get("regions") or {}

    return {
        str(camera): [_as_point(p) for p in ((config or {}).get("polygon") or [])]
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


def polygon_bbox(polygon) -> tuple[int, int, int, int]:
    """(x_min, y_min, x_max, y_max) of a polygon."""

    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]

    return min(xs), min(ys), max(xs), max(ys)


def render_regions_yaml(regions: dict[str, list[Point]], extra: dict | None = None) -> str:
    """
    The yaml text in the file's existing style: quoted camera keys and
    one "- [x, y]" row per point. Any other top-level keys are appended
    with the standard dumper.
    """

    lines = ["regions:"]

    for camera, polygon in regions.items():
        lines.append(f'  "{camera}":')
        lines.append("    polygon:")

        for x, y in polygon:
            lines.append(f"      - [{x}, {y}]")

        lines.append("")

    text = "\n".join(lines).rstrip("\n") + "\n"

    if extra:
        text += "\n" + yaml.safe_dump(extra, sort_keys=False)

    return text


def save_region(
    camera_id: str,
    points,
    path: Path | str | None = None,
    image_size: tuple[int, int] | None = None,
) -> list[Point]:
    """
    Replace (or add) one camera's polygon in the yaml, leaving the
    other cameras as they are. Points are normalised (two points =
    rectangle) and clamped to image_size when given. Returns the
    polygon that was written.
    """

    path = Path(path or CROP_CONFIG_PATH)

    camera_id = str(camera_id).strip()

    if not camera_id:
        raise ValueError("Camera id is empty.")

    polygon = normalise_polygon(points)

    if image_size:
        polygon = clamp_polygon(polygon, *image_size)

    content: dict = {}

    if path.exists():
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    regions = load_regions(path)
    regions[camera_id] = polygon

    extra = {key: value for key, value in content.items() if key != "regions"}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_regions_yaml(regions, extra), encoding="utf-8")

    return polygon


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
    saved_polygon=None,
    points=None,
    max_width: int = DISPLAY_WIDTH,
    base: Image.Image | None = None,
) -> tuple[Image.Image, float]:
    """
    The frame resized for display, with the saved polygon in blue and
    the points being marked in red (numbered, joined in order and
    closed from three points on). Returns (image, scale). Pass `base`
    (from display_base) to skip resizing the full frame again.
    """

    if base is None:
        base, scale = display_base(image, max_width)
    else:
        scale = image.size[0] / base.size[0]

    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)

    def shrink(polygon):
        return [(x / scale, y / scale) for x, y in polygon]

    if saved_polygon and len(saved_polygon) >= 2:
        draw.polygon(shrink(saved_polygon), outline=SAVED_COLOUR, width=3)

    if points:
        shown = shrink(points)

        if len(shown) >= 2:
            closed = shown + ([shown[0]] if len(shown) >= 3 else [])
            draw.line(closed, fill=MARK_COLOUR, width=3)

        for index, (x, y) in enumerate(shown, start=1):
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=MARK_COLOUR)
            draw.text((x + 8, y - 8), str(index), fill=MARK_COLOUR)

    return canvas, scale


def polygon_preview(image: Image.Image, points) -> Image.Image:
    """
    What the cropper will feed the model: the polygon's bounding box
    with everything outside the polygon black.
    """

    polygon = normalise_polygon(points)
    x1, y1, x2, y2 = polygon_bbox(polygon)

    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)

    black = Image.new("RGB", image.size, (0, 0, 0))
    composed = Image.composite(image.convert("RGB"), black, mask)

    return composed.crop((x1, y1, x2 + 1, y2 + 1))
