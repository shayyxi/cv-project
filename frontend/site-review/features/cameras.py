"""The All cameras page: every camera in the scope, worst compliance first."""

from __future__ import annotations

import streamlit as st

import review_data as rd
from features.i18n import register, t
from features.layout import clickable_table, empty_notice, page_header, toolbar
from features.state import S, frames_in_range, go, item_label, pct, reset_view, scope


FR = {
    'Camera': 'Caméra',
    'Search camera or site': 'Rechercher une caméra ou un chantier',
    '{n} cameras · worst first': '{n} caméras · pires en premier',
    'No cameras match.': 'Aucune caméra ne correspond.',
    'Compliance': 'Conformité',
    'People': 'Personnes',
    'Most missing': 'Le plus manquant',
}
register(FR)


def cams_page() -> None:
    toolbar(show_cam=False)
    page_header(t("All cameras"))
    cams = scope()
    empty_notice(cams)

    with st.container(key="card-cams"):
        cams_card(cams)


def cams_card(cams: list[str]) -> None:
    search_col, count_col = st.columns([2, 3], vertical_alignment="center")
    query = search_col.text_input(
        t("Search camera or site"),
        key="query",
        placeholder=t("Search camera or site"),
        label_visibility="collapsed",
    ).lower()
    rows = []
    for cam in cams:
        if query in f"{cam} {rd.SITE} {rd.camera_name(cam)}".lower():
            rows.append((cam, rd.stats(frames_in_range(cam))))
    rows.sort(key=lambda row: 2 if row[1].rate is None else row[1].rate)
    count_col.markdown(
        f"<div class='facts' style='text-align:right'>{t('{n} cameras · worst first', n=len(rows))}</div>",
        unsafe_allow_html=True,
    )
    if not rows:
        st.caption(t("No cameras match."))
        return

    table = []
    for i, (cam, s) in enumerate(rows):
        top = rd.top_missing(s)
        rate = s.rate
        cls = "" if rate is None else "bad" if rate < 0.7 else "ok" if rate >= 0.9 else ""
        colour = "var(--bad)" if (rate or 0) < 0.7 else "var(--ok)"
        bar = f"<span class='track'><i style='width:{(rate or 0) * 100:.0f}%;background:{colour}'></i></span>"
        most = f"{item_label(top)} ({s.miss[top]})" if s.n and top else "–"
        table.append(
            f"<tr data-i='{i}'><td><b>{rd.camera_name(cam)}</b> <span class='d'>{cam}</span></td>"
            f"<td>{rd.SITE}</td><td><span class='rate {cls}'>{pct(rate)}</span>{bar}</td>"
            f"<td>{s.n}</td><td>{most}</td></tr>"
        )
    picked = clickable_table(
        [(t("Camera"), False), (t("Site"), False), (t("Compliance"), False), (t("People"), False),
         (t("Most missing"), False)],
        table,
        key=f"cams_table_{S.pick}",
    )
    if picked is not None and picked < len(rows):
        S.pick += 1
        S.cam = rows[picked][0]
        S.pop("cam_select", None)
        reset_view()
        go("overview")
