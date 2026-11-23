"""Numbers, times, spans and intervals, by hand in both languages, and the status reason text."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import streamlit as st

import health_data as hd
from features.i18n import register, t


FR = {
    'no new frame for {age} against a {interval} interval': 'aucune nouvelle image depuis {age} pour un intervalle de {interval}',
    '{n} failed polls in a row': '{n} interrogations échouées de suite',
    '{n} min': '{n} min',
    '{n} h': '{n} h',
    '{n} s': '{n} s',
    'Mon': 'lun.',
    'Tue': 'mar.',
    'Wed': 'mer.',
    'Thu': 'jeu.',
    'Fri': 'ven.',
    'Sat': 'sam.',
    'Sun': 'dim.',
}
register(FR)


SITE_TZ = ZoneInfo("America/Toronto")


MONTHS = {
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "fr": [
        "janv.", "févr.", "mars", "avr.", "mai", "juin",
        "juil.", "août", "sept.", "oct.", "nov.", "déc.",
    ],
}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def lang() -> str:
    return st.session_state.get("lang", "en")


def num(value: float, digits: int) -> str:
    text = f"{value:,.{digits}f}"
    if lang() == "fr":
        text = text.replace(",", " ").replace(".", ",")
    return text


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "–"
    return num(value * 100, digits) + (" %" if lang() == "fr" else "%")


def int_(value: float | None) -> str:
    return "–" if value is None else num(round(value), 0)


def sec(ms: float | None) -> str:
    return "–" if ms is None else num(ms / 1000, 1) + " s"


def gb(value: float | None) -> str:
    return "–" if value is None else num(value / 1e9, 1) + " GB"


def plain(value: float) -> str:
    """7 -> "7", 2.5 -> "2.5" (2,5 in French), as JavaScript prints numbers."""

    text = str(int(value)) if float(value).is_integer() else f"{value:.1f}"
    return text.replace(".", ",") if lang() == "fr" else text


def fmt_time(d: datetime | None) -> str:
    return d.astimezone(SITE_TZ).strftime("%H:%M") if d else "–"


def fmt_datetime(d: datetime | None) -> str:
    if not d:
        return "–"
    d = d.astimezone(SITE_TZ)
    return f"{d.day} {MONTHS[lang()][d.month - 1]}, {d:%H:%M}"


def weekday(day_key: str) -> str:
    return t(WEEKDAYS[date.fromisoformat(day_key).weekday()])


def span(seconds: float | None) -> str:
    parts = hd.span_parts(seconds)
    if parts is None:
        return "–"
    unit, n = parts
    return t({"s": "{n} s", "min": "{n} min", "h": "{n} h", "d": "{n} d"}[unit], n=plain(n))


def ago(seconds: float | None) -> str:
    return t("never") if seconds is None else t("{age} ago", age=span(seconds))


def interval(seconds: int) -> str:
    unit, n = hd.interval_parts(seconds)
    return t("{n} h" if unit == "h" else "{n} min", n=plain(n))


def reason_for(cam: dict, status: str, age: float | None) -> str:
    parts = []
    if age is not None and status != "healthy":
        parts.append(t(
            "no new frame for {age} against a {interval} interval",
            age=span(age), interval=interval(cam.get("interval_seconds") or 3600),
        ))
    if cam.get("consecutive_failures"):
        parts.append(t("{n} failed polls in a row", n=cam["consecutive_failures"]))
    return ", ".join(parts)
