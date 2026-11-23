"""Recent failures: the last 20 failed downloads, processing jobs and deliveries."""

from __future__ import annotations

from html import escape as esc

import streamlit as st

import health_data as hd
from features.format import fmt_datetime
from features.i18n import register, t
from features.widgets import html_table, section_head


FR = {
    'Recent failures': 'Échecs récents',
    'The last 20 failed downloads, processing jobs and deliveries, newest first': 'Les 20 derniers échecs de téléchargement, de traitement et de livraison, du plus récent au plus ancien',
    'Kind': 'Type',
    'Message': 'Message',
    'download': 'téléchargement',
    'No failure recorded in the last 30 days.': 'Aucun échec enregistré dans les 30 derniers jours.',
}
register(FR)


def failures_section(data: dict) -> None:
    section_head(
        t("Recent failures"), t("The last 20 failed downloads, processing jobs and deliveries, newest first")
    )
    with st.container(key="card-failures"):
        failures_card(data)


def failures_card(data: dict) -> None:
    rows = []
    for failure in (data.get("failures") or [])[:20]:
        kind = failure.get("kind") or ""
        cls = "crit" if kind == "delivery" else "warn" if kind == "download" else "info"
        message = esc(failure.get("message") or "")
        status = failure.get("http_status")
        rows.append([
            esc(fmt_datetime(hd.parse_ts(failure.get("at")))),
            f"<span class='pill {cls}'>{esc(t(kind))}</span>",
            esc(hd.camera_label(data, failure.get("camera_id"))),
            esc("–" if status is None else str(status)),
            f"<span class='msg' title='{message}'>{message}</span>",
        ])
    if not rows:
        st.caption(t("No failure recorded in the last 30 days."))
        return
    st.markdown(html_table([
        (t("Time"), "time"), (t("Kind"), ""), (t("Camera"), ""), (t("HTTP"), "num"), (t("Message"), ""),
    ], rows), unsafe_allow_html=True)
