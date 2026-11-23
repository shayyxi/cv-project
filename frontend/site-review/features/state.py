"""
What every page shares: the session defaults, the loaded results, the
date range in site time, time and number formatting, and the widget
callbacks (range, dates, camera, role, filter, strip, zoom, show all).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

import review_data as rd
from _shared import load_json, remember_choice
from features.i18n import register, t

APP_DIR = Path(__file__).resolve().parents[1]


FR = {
    'Today, {date}': 'Aujourd’hui, {date}',
    'last {n} days': '{n} derniers jours',
    'Missing helmet': 'Casque manquant',
    'Missing safety vest': 'Veste de sécurité manquante',
    'Missing safety boots': 'Bottes de sécurité manquantes',
    'Missing harness': 'Harnais manquant',
    'No helmet': 'Sans casque',
    'No safety vest': 'Sans veste',
    'No safety boots': 'Sans bottes',
    'No harness': 'Sans harnais',
}
register(FR)


PER_PAGE = 25  # sightings per table page
RANGES = ("1", "7", "30", "60", "c")
GREEN, RED, GREY = "#8ac646", "#ef6b64", "#9a9a9a"
NO_ITEM = {
    "helmet": "No helmet",
    "vest": "No safety vest",
    "boots": "No safety boots",
    "harness": "No harness",
}
MISSING = {
    "helmet": "Missing helmet",
    "vest": "Missing safety vest",
    "boots": "Missing safety boots",
    "harness": "Missing harness",
}


# "lang" is set by page_setup (browser language, ?lang=), not here.
# Dark only and Eastern time only: no theme or time-zone state.
DEFAULTS = {
    "role": "admin",
    "cam": None,
    "range": "60",
    "from": "",
    "to": "",
    "filter": "all",
    "heat": "bad",
    "sel": None,
    "person": None,
    "zoom": 1.0,
    "zoom_center": None,
    "log_page": 1,
    "pick": 0,
    "toast": "",
}
S = st.session_state

# Filled by init_state() on every script run: this module is imported
# once per process, but each browser session needs its defaults, the
# clock moves, and results.json may change while the server runs.
NOW = datetime.now(timezone.utc)
DATA: dict = {}
IS_SAMPLE = True
REVIEW = rd.ingest(None)
PAGES: dict[str, st.Page] = {}


def init_state() -> None:
    """Session defaults, the clock and the results file, once per script run."""

    global NOW, DATA, IS_SAMPLE, REVIEW
    for key, value in DEFAULTS.items():
        S.setdefault(key, value)
    NOW = datetime.now(timezone.utc)
    DATA, IS_SAMPLE = load_json(APP_DIR, "results.json", "sample_results.json")
    REVIEW = rd.ingest(DATA)
# The right part of the header's second row, where each page draws its controls.
CONTROLS = None


def lang() -> str:
    return S.get("lang", "en")


def tz():
    """Times are the site's (Eastern), as the reviewer asked; no UTC switch."""

    return rd.SITE_TZ


def tz_label() -> str:
    return t("Eastern")


def scope() -> list[str]:
    return rd.scope(REVIEW.cams, S.role)


def current_cam() -> str | None:
    cams = scope()
    if S.cam not in cams:
        S.cam = cams[0] if cams else None
    return S.cam


def cam_frames(cam: str | None) -> list[rd.Frame]:
    return REVIEW.frames.get(cam, []) if cam else []


def bounds() -> tuple[datetime, datetime]:
    return rd.range_bounds(S.range, S["from"], S["to"], NOW, tz())


def frames_in_range(cam: str | None, prior: bool = False) -> list[rd.Frame]:
    a, b = bounds()
    return rd.in_range(cam_frames(cam), a, b, prior)


def fd(d: datetime) -> str:
    return rd.fd(d, tz(), lang())


def fmt(d: datetime) -> str:
    return rd.fmt(d, tz(), lang())


def pct(value: float | None) -> str:
    return rd.pct(value, lang())


def range_text() -> str:
    a, b = bounds()
    return t("Today, {date}", date=fd(b)) if S.range == "1" else f"{fd(a)} – {fd(b)}"


def range_label() -> str:
    a, b = bounds()
    if S.range == "1":
        return t("today")
    if S.range == "c":
        return f"{fd(a)} – {fd(b)}"
    return t("last {n} days", n=S.range)


def item_label(key: str) -> str:
    return t(rd.CHECK_LABELS[key])


def no_item(key: str) -> str:
    return t(NO_ITEM[key]) if key in NO_ITEM else item_label(key)


def is_dark() -> bool:
    """The page is dark only; a print run is drawn light for paper."""

    return not S.get("print_request")


def reset_view() -> None:
    S.sel = None
    S.person = None
    S.zoom = 1.0
    S.zoom_center = None
    S.pop("strip", None)
    S.log_page = 1


# ---- widget callbacks (they run before the next script run)


def on_range() -> None:
    remember_choice("range_control", "range")
    if S.range == "c" and not (rd.valid_day(S["from"]) and rd.valid_day(S["to"])):
        today = rd.day_key(NOW, tz())
        S["from"], S["to"] = rd.shift_day(today, -6), today
    S.pop("from_input", None)
    S.pop("to_input", None)
    reset_view()


def on_dates() -> None:
    a, b = S.get("from_input"), S.get("to_input")
    if not (a and b):
        return
    if a > b:
        a, b = b, a
        S.from_input, S.to_input = a, b
    S["from"], S["to"] = a.isoformat(), b.isoformat()
    reset_view()


def on_cam() -> None:
    S.cam = S.cam_select
    reset_view()


def on_role() -> None:
    S.role = S.role_control
    cams = scope()
    S.cam = cams[0] if len(cams) == 1 else None
    S.pop("cam_select", None)
    reset_view()


def on_filter() -> None:
    remember_choice("filter_control", "filter")
    reset_view()


def on_strip() -> None:
    S.sel = S.strip
    S.person = None
    S.zoom = 1.0
    S.zoom_center = None


def step(delta: int, count: int) -> None:
    current = S.get("strip", count - 1)
    S.strip = max(0, min(count - 1, current + delta))
    S.sel = S.strip
    S.person = None
    S.zoom = 1.0
    S.zoom_center = None


def zoom_by(factor: float) -> None:
    """＋ and － scale the view; 0 fits the whole frame."""

    S.zoom = 1.0 if factor == 0 else max(1.0, min(8.0, S.zoom * factor))


def zoom_to(bbox: tuple[float, float, float, float], aspect: float) -> None:
    """Zoom the viewer onto a person's region (fractions), as the design page does."""

    x, y, w, h = bbox
    S.zoom = max(3.0, min(8.0, 0.21 / max(w, h / aspect, 0.01)))
    S.zoom_center = (x + w / 2, y + h / 2)


def toggle_person(index: int, bbox: tuple[float, float, float, float], aspect: float) -> None:
    if S.person == index:
        S.person = None
        S.zoom = 1.0
        S.zoom_center = None
    else:
        S.person = index
        zoom_to(bbox, aspect)


def go_images(key: str) -> None:
    """Open the Images page on one violation filter (from the score rows)."""

    S.filter = key
    reset_view()
    S.goto = "images"


def show_all() -> None:
    S.range = "c"
    S.pop("range_control", None)  # recreated with the new default, no warning
    if REVIEW.first:
        S["from"] = rd.day_key(REVIEW.first, tz())
    if REVIEW.last:
        S["to"] = rd.day_key(REVIEW.last, tz())
    S.pop("from_input", None)
    S.pop("to_input", None)
    reset_view()


def go(page_key: str) -> None:
    st.switch_page(PAGES[page_key])


# ---- frames: the folders are tried in order; a frame that is not here gets a placeholder
