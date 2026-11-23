"""
Download reports: the dialog (cameras, report types, summary) and the
request itself, which prints the page until the report service exists.
"""

from __future__ import annotations

import logging

import streamlit as st

import review_data as rd
from _shared import remember_choice
from features.i18n import register, t
from features.state import S, bounds, current_cam, range_text, scope

log = logging.getLogger("site_review")


FR = {
    'Download reports': 'Télécharger les rapports',
    'Download report': 'Télécharger le rapport',
    'today': 'aujourd’hui',
    'use “Save as PDF” in the print dialog': 'choisissez « Enregistrer en PDF » dans la fenêtre d’impression',
    'Select all': 'Tout sélectionner',
    'Clear': 'Effacer',
    '{n} selected': '{n} sélectionnée(s)',
    'Preparing {what} (prototype)': 'Préparation de {what} (prototype)',
    '{n} camera report': '{n} rapport de caméra',
    '{n} camera reports': '{n} rapports de caméra',
    '1 combined report': '1 rapport combiné',
    'Period: {range} (follows the date range on the page)': 'Période : {range} (suit la période choisie sur la page)',
    'All cameras ({n})': 'Toutes les caméras ({n})',
    'Choose cameras': 'Choisir les caméras',
    'Report type': 'Type de rapport',
    'One report per camera': 'Un rapport par caméra',
    'A separate PDF for each camera, with its full stats': 'Un PDF distinct pour chaque caméra, avec toutes ses statistiques',
    'One combined report': 'Un rapport combiné',
    'Overall stats across the chosen cameras, with a camera-by-camera comparison': 'Statistiques globales des caméras choisies, avec une comparaison caméra par caméra',
    'Choose at least one report type.': 'Choisissez au moins un type de rapport.',
    'Choose at least one camera.': 'Choisissez au moins une caméra.',
    '{n} individual report + 1 combined report': '{n} rapport individuel + 1 rapport combiné',
    '{n} individual reports + 1 combined report': '{n} rapports individuels + 1 rapport combiné',
    '1 report for this camera': '1 rapport pour cette caméra',
    '{n} separate reports, one per camera': '{n} rapports distincts, un par caméra',
    '1 combined report covering {n} camera': '1 rapport combiné couvrant {n} caméra',
    '1 combined report covering {n} cameras': '1 rapport combiné couvrant {n} caméras',
    'Cancel': 'Annuler',
}
register(FR)


def request_report(cams: list[str], per_camera: bool, combined: bool) -> None:
    """
    TEAM: wire this to the report service. cams = camera ids, per_camera
    = one PDF per camera, combined = one overall-stats PDF; the period is
    the date range currently selected on the page. Until then the browser
    prints what is on screen (the print stylesheet hides the controls)
    and "Save as PDF" makes the file.
    """

    a, b = bounds()
    n = len(cams)
    what = []
    if per_camera:
        what.append(t("{n} camera report" if n == 1 else "{n} camera reports", n=n))
    if combined:
        what.append(t("1 combined report"))
    log.info(
        "requestReport cameras=%s perCamera=%s combined=%s from=%s to=%s",
        cams, per_camera, combined, a.isoformat(), b.isoformat(),
    )
    S.toast = (
        t("Preparing {what} (prototype)", what=" + ".join(what))
        + " · " + t("use “Save as PDF” in the print dialog")
    )
    S.print_request = True


def open_report_dialog(default_mode: str) -> None:
    """A fresh dialog each time: the open camera preselected, like the design page."""

    cam = current_cam()
    for key in [k for k in S.keys() if str(k).startswith("report_")]:
        S.pop(key, None)
    S.report_sel = [cam] if cam else []
    S.report_mode = default_mode
    report_dialog(default_mode)


def set_report_camera(cam: str, site_members: list[str]) -> None:
    chosen = set(S.report_sel)
    (chosen.add if S.get(f"report_cam_{cam}") else chosen.discard)(cam)
    S.report_sel = sorted(chosen)
    S[f"report_site_{rd.SITE}"] = all(c in chosen for c in site_members)


def set_report_site(site: str, members: list[str]) -> None:
    on = bool(S.get(f"report_site_{site}"))
    chosen = set(S.report_sel)
    chosen = chosen | set(members) if on else chosen - set(members)
    S.report_sel = sorted(chosen)
    for cam in members:
        S[f"report_cam_{cam}"] = on


def set_report_all(cams: list[str], on: bool) -> None:
    S.report_sel = list(cams) if on else []
    for cam in cams:
        S[f"report_cam_{cam}"] = on
    S[f"report_site_{rd.SITE}"] = on


@st.dialog(t("Download reports"))
def report_dialog(default_mode: str) -> None:
    cams = scope()
    S.setdefault("report_sel", [])
    st.caption(t("Period: {range} (follows the date range on the page)", range=range_text()))

    st.markdown(f"**{t('Cameras')}**")
    # A click on the active segment would deselect it: the last choice is kept instead.
    mode = st.segmented_control(
        t("Cameras"),
        ["all", "sel"],
        default=S.get("report_mode", default_mode),
        format_func=lambda k: t("All cameras ({n})", n=len(cams)) if k == "all" else t("Choose cameras"),
        key="report_scope",
        on_change=remember_choice,
        args=("report_scope", "report_mode"),
        label_visibility="collapsed",
    ) or S.get("report_mode", default_mode)
    if mode == "all":
        selected = list(cams)
    else:
        tools = st.container(key="report_tools")
        query_col, all_col, clear_col, count_col = tools.columns(
            [2.6, 1.6, 1.2, 1.4], vertical_alignment="center"
        )
        query = query_col.text_input(
            t("Search camera or site"),
            key="report_q",
            placeholder=t("Search camera or site"),
            label_visibility="collapsed",
        ).lower()
        all_col.button(
            t("Select all"), key="report_all", on_click=set_report_all, args=(cams, True), use_container_width=True
        )
        clear_col.button(
            t("Clear"), key="report_clear", on_click=set_report_all, args=(cams, False), use_container_width=True
        )
        count_col.caption(t("{n} selected", n=len(S.report_sel)))

        # Grouped by site, one site today; the site box ticks all its cameras.
        members = [c for c in cams if query in f"{c} {rd.SITE} {rd.camera_name(c)}".lower()]
        if not members:
            st.caption(t("No cameras match."))
        else:
            S.setdefault(f"report_site_{rd.SITE}", all(c in S.report_sel for c in members))
            st.checkbox(
                f"{rd.SITE} ({len(members)})",
                key=f"report_site_{rd.SITE}",
                on_change=set_report_site,
                args=(rd.SITE, members),
            )
            for cam in members:
                S.setdefault(f"report_cam_{cam}", cam in S.report_sel)
                st.checkbox(
                    f"　{rd.camera_name(cam)}",
                    key=f"report_cam_{cam}",
                    on_change=set_report_camera,
                    args=(cam, members),
                )
        selected = [c for c in cams if c in S.report_sel]

    st.markdown(f"**{t('Report type')}**")
    per = st.checkbox(t("One report per camera"), value=True, key="report_per")
    st.caption(t("A separate PDF for each camera, with its full stats"))
    comb = st.checkbox(t("One combined report"), key="report_combined")
    st.caption(t("Overall stats across the chosen cameras, with a camera-by-camera comparison"))

    n = len(selected)
    ok = n > 0 and (per or comb)
    if not ok:
        st.error(t("Choose at least one report type.") if n else t("Choose at least one camera."))
    elif per and comb:
        key = "{n} individual report + 1 combined report" if n == 1 else "{n} individual reports + 1 combined report"
        st.info(t(key, n=n))
    elif per:
        st.info(t("1 report for this camera") if n == 1 else t("{n} separate reports, one per camera", n=n))
    else:
        key = "1 combined report covering {n} camera" if n == 1 else "1 combined report covering {n} cameras"
        st.info(t(key, n=n))

    cancel_col, go_col = st.columns(2)
    if cancel_col.button(t("Cancel"), key="report_cancel", use_container_width=True):
        st.rerun()
    if go_col.button(t("Download"), type="primary", disabled=not ok, key="report_go", use_container_width=True):
        request_report(selected, per, comb)
        st.rerun()


# ---- pages
