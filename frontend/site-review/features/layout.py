"""
The pieces every page draws: the toolbar on the tab row, the page title
and the brand line, card headings, the design's clickable table, the
empty-range notice and the fact rows.
"""

from __future__ import annotations

from datetime import date

import streamlit as st
import streamlit.components.v1 as components

import review_data as rd
from features import state
from features.i18n import register, t
from features.reports import open_report_dialog, request_report
from features.state import (
    APP_DIR,
    RANGES,
    S,
    current_cam,
    fmt,
    frames_in_range,
    is_dark,
    on_cam,
    on_dates,
    on_range,
    range_text,
    scope,
    show_all,
    tz,
)


FR = {
    'Date range': 'Période',
    'Today': 'Aujourd’hui',
    '7 days': '7 jours',
    '30 days': '30 jours',
    '60 days': '60 jours',
    'Custom': 'Personnalisé',
    'From': 'Du',
    'Data as of {when}': 'Données au {when}',
    'latest frame {when}': 'dernière image {when}',
    'Sample data: results.json was not found next to the app.': 'Données d’exemple : results.json est introuvable à côté de l’application.',
    'No images in this range.': 'Aucune image dans cette période.',
    'Show all data on file': 'Afficher toutes les données',
}
register(FR)


# The design page's tables (uppercase headers, result pills, PPE tags,
# compliance bars) with rows that open on a click: table_component/index.html.
review_table = components.declare_component("review_table", path=str(APP_DIR / "table_component"))


def toolbar(show_cam: bool) -> None:
    """The page's controls, on the header's tab row, right of the tabs."""

    with state.CONTROLS:
        toolbar_row(show_cam)


def toolbar_row(show_cam: bool) -> None:
    cams = scope()
    multi = len(cams) > 1

    if show_cam and multi:
        with st.container(width=170):
            st.selectbox(
                t("Camera"),
                cams,
                index=cams.index(current_cam()),
                format_func=rd.camera_name,
                key="cam_select",
                on_change=on_cam,
                label_visibility="collapsed",
            )

    names = {
        "1": t("Today"), "7": t("7 days"), "30": t("30 days"), "60": t("60 days"), "c": t("Custom")
    }
    st.segmented_control(
        t("Date range"),
        list(RANGES),
        default=S.range,
        format_func=names.get,
        key="range_control",
        on_change=on_range,
        label_visibility="collapsed",
        width="content",
    )

    if S.range == "c":
        # Custom dates reach back to the oldest frame on file.
        today = rd.day_key(state.NOW, tz())
        oldest = rd.day_key(state.REVIEW.first, tz()) if state.REVIEW.first else today
        earliest = date.fromisoformat(min(oldest, rd.shift_day(today, -59)))
        latest = date.fromisoformat(today)
        for key, store, label in (("from_input", "from", "From"), ("to_input", "to", "To")):
            if key not in S:
                S[key] = date.fromisoformat(S[store])
            with st.container(width=140):
                st.date_input(
                    t(label),
                    key=key,
                    min_value=earliest,
                    max_value=latest,
                    on_change=on_dates,
                    label_visibility="collapsed",
                )

    label = t("Download reports") if multi else t("Download report")
    clicked = st.button(label, type="primary", key="download")
    if clicked:
        if multi:
            open_report_dialog("sel" if show_cam else "all")
        else:
            request_report(cams, True, False)


def page_header(title: str | None) -> None:
    cam = current_cam()
    heading = title or (f"{rd.camera_name(cam)} · {rd.SITE}" if cam else "")
    # Only on paper: the print stylesheet hides the controls and shows this line.
    st.markdown(
        f"<div class='print-only'>{t('Site Safety Review')} · {heading} · {range_text()}</div>",
        unsafe_allow_html=True,
    )
    if title:
        st.markdown(
            f"<div class='ptitle'>{title}</div><div class='psub'>{range_text()}</div>",
            unsafe_allow_html=True,
        )


def brand_html() -> str:
    """The header's left side: the title and the data line under it."""

    parts = []
    review = state.REVIEW
    if review.generated:
        parts.append(t("Data as of {when}", when=fmt(review.generated)))
    elif review.generated_raw:
        parts.append(t("Data as of {when}", when=review.generated_raw))
    if review.last:
        parts.append(t("latest frame {when}", when=fmt(review.last)))
    if state.IS_SAMPLE:
        parts.append(t("Sample data: results.json was not found next to the app."))
    return (
        f"<div class='brand'><div class='logo'>{t('Site Safety Review')}</div>"
        f"<div class='fresh'>{' · '.join(parts)}</div></div>"
    )


def card_title(text: str, right: str = "") -> None:
    """A card's small uppercase heading, with optional text at the right."""

    extra = f"<span class='r'>{right}</span>" if right else ""
    st.markdown(f"<div class='ch'><span>{text}</span>{extra}</div>", unsafe_allow_html=True)


def clickable_table(head: list[tuple[str, bool]], rows: list[str], key: str) -> int | None:
    """
    A design-styled table; `rows` are <tr data-i='n'> strings. Returns the
    row index clicked since the last call, else None.
    """

    ths = "".join(f"<th{' class=num' if num else ''}>{label}</th>" for label, num in head)
    html = f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    value = review_table(html=html, light=not is_dark(), key=key, default=None)
    if not value:
        return None
    token = (key, value.get("n"))
    if S.get("table_click") == token:
        return None
    S.table_click = token
    return value.get("row")


def empty_notice(cams: list[str]) -> bool:
    """When the range holds nothing: say so and offer the whole archive."""

    if state.REVIEW.last is None or any(frames_in_range(c) for c in cams):
        return False
    text_col, button_col = st.columns([4, 1], vertical_alignment="center")
    text_col.warning(
        f"{t('No images in this range.')} {t('latest frame {when}', when=fmt(state.REVIEW.last))}."
    )
    button_col.button(t("Show all data on file"), key="showall", on_click=show_all)
    return True


def meta_html(items: list[tuple[str, object]]) -> str:
    return "<div class='meta'>" + "".join(
        f"<div><span>{label}</span><b>{value}</b></div>" for label, value in items
    ) + "</div>"
