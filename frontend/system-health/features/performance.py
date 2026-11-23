"""Processing performance: P50 and P95 latency per day."""

from __future__ import annotations

from html import escape as esc

import altair as alt
import pandas as pd
import streamlit as st

import health_data as hd
from _shared import altair_chart
from features.format import int_, plain, weekday
from features.i18n import register, t
from features.widgets import section_head


FR = {
    'Processing performance': 'Performance du traitement',
    'P50 and P95 per-image processing latency, seconds': 'Latence de traitement par image P50 et P95, en secondes',
    'Latency per day': 'Latence par jour',
    'Hover a day for both values. {n} images in the last {d} days.': 'Survolez un jour pour voir les deux valeurs. {n} images dans les {d} derniers jours.',
    'No processed image in this window yet.': 'Aucune image traitée dans cette période pour l’instant.',
    'P50 processing time': 'Temps de traitement P50',
    'P95 processing time': 'Temps de traitement P95',
    'Time': 'Heure',
    'Size': 'Taille',
    'dark': 'sombre',
}
register(FR)


GREEN, GREY = "#8ac646", "#b3b3b3"


def performance_section(data: dict) -> None:
    section_head(t("Processing performance"), t("P50 and P95 per-image processing latency, seconds"))
    with st.container(key="card-performance"):
        performance_card(data)


def performance_card(data: dict) -> None:
    series = hd.latency_series(data)
    total = sum(count for *_, count in series)
    caption = t(
        "Hover a day for both values. {n} images in the last {d} days.",
        n=int_(total), d=data.get("window_days") or 7,
    ) if series else t("No processed image in this window yet.")
    st.markdown(f"<div class='h3'>{esc(t('Latency per day'))}</div><div class='cap'>{esc(caption)}</div>", unsafe_allow_html=True)
    if not series:
        return
    p50_label, p95_label = t("P50 processing time"), t("P95 processing time")
    long = pd.DataFrame([
        {"day": weekday(day), "series": label, "seconds": value}
        for day, p50, p95, _ in series
        for label, value in ((p50_label, p50), (p95_label, p95))
    ])
    base = alt.Chart(long).encode(
        # sort=None keeps the days in data order, so the translated weekday labels stay in sequence
        x=alt.X("day:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, grid=False)),
        y=alt.Y("seconds:Q", title="s", scale=alt.Scale(domainMin=0)),
        color=alt.Color(
            "series:N", title=None, legend=alt.Legend(orient="top"),
            # as on the design page: the usual time in grey, the slow cases in green
            scale=alt.Scale(domain=[p50_label, p95_label], range=[GREY, GREEN]),
        ),
        tooltip=[
            alt.Tooltip("day:N", title=t("Time")),
            alt.Tooltip("series:N", title=" "),
            alt.Tooltip("seconds:Q", title="s", format=".1f"),
        ],
    )
    # The last value of each line is written at its end, as on the design page.
    last_day, last_p50, last_p95, _ = series[-1]
    ends = pd.DataFrame([
        {"day": weekday(last_day), "series": label, "seconds": value, "text": f"{short} {plain(round(value, 1))} s"}
        for label, short, value in ((p50_label, "P50", last_p50), (p95_label, "P95", last_p95))
        if value is not None
    ])
    end_labels = alt.Chart(ends).mark_text(align="left", dx=10, fontSize=11, fontWeight="bold").encode(
        x=alt.X("day:N", sort=None),
        y="seconds:Q",
        text="text:N",
        color=alt.Color(
            "series:N", legend=None,
            scale=alt.Scale(domain=[p50_label, p95_label], range=[GREY, GREEN]),
        ),
    )
    chart = (base.mark_line(strokeWidth=2.5) + base.mark_circle(size=50) + end_labels).properties(
        height=280, width="container", padding={"left": 5, "top": 5, "right": 70, "bottom": 5}
    )
    altair_chart(chart, dark=True)
