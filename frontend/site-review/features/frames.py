"""
The camera frames: finding and loading them (or a grey placeholder when
a frame is not on this computer), the person highlight, the zoom crop,
thumbnails and the heatmap overlay.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

import review_data as rd
from features import state
from features.state import APP_DIR, GREEN


# Folders tried in order for a frame, after the ones named in results.json.
IMAGE_DIRS = (APP_DIR / "processed", APP_DIR.parents[1] / "data" / "processed")
DISPLAY_WIDTH = 1800  # frames are shown and cropped at this width at most


def image_dirs() -> list[Path]:
    dirs = []
    for d in state.REVIEW.img_dirs:
        p = Path(d)
        dirs.append(p if p.is_absolute() else APP_DIR / p)
    dirs.extend(IMAGE_DIRS)
    return dirs


def find_image(file: str) -> Path | None:
    for d in image_dirs():
        p = d / file
        if p.is_file():
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_display(path: str, mtime: float) -> Image.Image:
    """The frame downscaled for display; cached per file version. Never drawn on directly."""

    with Image.open(path) as im:
        im.draft("RGB", (DISPLAY_WIDTH, DISPLAY_WIDTH))
        img = im.convert("RGB")
    img.thumbnail((DISPLAY_WIDTH, DISPLAY_WIDTH))
    return img


def placeholder(frame: rd.Frame) -> Image.Image:
    w = 1200
    h = max(1, round(w * frame.h / frame.w))
    img = Image.new("RGB", (w, h), (39, 39, 39))
    ImageDraw.Draw(img).text((16, h - 28), frame.file, fill=(179, 179, 179))
    return img


def frame_image(frame: rd.Frame) -> tuple[Image.Image, bool]:
    path = find_image(frame.file)
    if path is not None:
        try:
            return load_display(str(path), path.stat().st_mtime), True
        except OSError:
            pass
    return placeholder(frame), False


def box_px(img: Image.Image, frac: tuple[float, float, float, float], pad: float = 0.0):
    x, y, w, h = frac
    W, H = img.size
    x0, y0 = max(0, int((x - pad) * W)), max(0, int((y - pad) * H))
    x1, y1 = min(W, int((x + w + pad) * W)), min(H, int((y + h + pad) * H))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def draw_highlight(img: Image.Image, person: rd.Person) -> Image.Image:
    out = img.copy()
    width = max(2, out.width // 400)
    ImageDraw.Draw(out).rectangle(box_px(out, person.lbl, pad=0.004), outline=GREEN, width=width)
    return out


def zoomed(img: Image.Image, zoom: float, center: tuple[float, float] | None) -> Image.Image:
    """The part of the frame the viewer shows at this zoom, around the centre (fractions)."""

    if zoom <= 1.0:
        return img
    W, H = img.size
    w, h = W / zoom, H / zoom
    cx, cy = ((center or (0.5, 0.5))[0] * W, (center or (0.5, 0.5))[1] * H)
    x0 = min(max(0.0, cx - w / 2), W - w)
    y0 = min(max(0.0, cy - h / 2), H - h)
    return img.crop((int(x0), int(y0), int(x0 + w), int(y0 + h)))


def thumb(img: Image.Image, person: rd.Person) -> Image.Image:
    crop = img.crop(box_px(img, person.bbox))
    crop.thumbnail((96, 96))
    return crop


@st.cache_resource(show_spinner=False)
def heat_tile(size: int) -> Image.Image:
    """A red disc fading to transparent at the edge, the design's radial gradient."""

    alpha = Image.radial_gradient("L").resize((size, size))
    alpha = alpha.point(lambda v: int(0.55 * 255 * max(0, 1 - v / 255)))
    tile = Image.new("RGBA", (size, size), (232, 83, 79, 0))
    tile.putalpha(alpha)
    return tile


def heat_overlay(base: Image.Image, points: list[tuple[float, float]]) -> Image.Image:
    W, H = base.size
    r = max(4, round(W * 0.03))
    tile = heat_tile(2 * r)
    layer = Image.new("RGBA", (W + 2 * r, H + 2 * r), (0, 0, 0, 0))
    for cx, cy in points:
        layer.alpha_composite(tile, dest=(round(cx * W), round(cy * H)))
    layer = layer.crop((r, r, r + W, r + H))
    return Image.alpha_composite(base.convert("RGBA"), layer)


# ---- shared page parts
