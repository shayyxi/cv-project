"""Camera fleet health: one row per camera with its status, and how the status is set."""

from __future__ import annotations

from datetime import datetime
from html import escape as esc

import streamlit as st

import health_data as hd
from features.format import ago, int_, interval, pct, plain, reason_for
from features.i18n import register, t
from features.widgets import html_table, section_head


FR = {
    'Camera fleet health': 'État du parc de caméras',
    'Freshness per camera against its own refresh interval': 'Fraîcheur par caméra par rapport à son propre intervalle de rafraîchissement',
    'Interval': 'Intervalle',
    'Last image': 'Dernière image',
    'Last processed': 'Dernier traitement',
    '24h success': 'Réussite 24 h',
    'Frames (7 d)': 'Images (7 j)',
    'Status': 'État',
    'Healthy': 'Saine',
    'Delayed': 'En retard',
    'Down': 'Hors service',
    'Unknown': 'Inconnue',
    'How the status is set': 'Comment l’état est établi',
    'Healthy = a new frame arrived within {df}× the camera’s own interval. Delayed = no new frame for more than {df}× the interval. Down = no new frame for more than {dn}× the interval, or {nf} failed polls in a row. Unknown = no frame on file yet.': 'Saine = une nouvelle image est arrivée dans {df}× l’intervalle propre à la caméra. En retard = aucune nouvelle image depuis plus de {df}× l’intervalle. Hors service = aucune nouvelle image depuis plus de {dn}× l’intervalle, ou {nf} interrogations échouées de suite. Inconnue = aucune image en mémoire.',
    'Intervals: {list}.': 'Intervalles : {list}.',
    '{name} is {status}: {reason}.': '{name} est {status} : {reason}.',
    'last error: {error}': 'dernière erreur : {error}',
}
register(FR)


def fleet_section(data: dict, now: datetime, th: dict, cams: list[dict]) -> None:
    section_head(t("Camera fleet health"), t("Freshness per camera against its own refresh interval"))
    with st.container(key="card-fleet"):
        fleet_card(data, now, th, cams)


def fleet_card(data: dict, now: datetime, th: dict, cams: list[dict]) -> None:
    rows = []
    notes = []
    for cam in cams:
        key, age = hd.camera_status(cam, th, now)
        cls, label = hd.PILL[key]
        reason = reason_for(cam, key, age)
        name = cam.get("name") or f"{t('Camera')} {cam['id']}"
        rows.append([
            f"<strong>{esc(name)}</strong><span class='id'>{esc(str(cam['id']))}</span>",
            f"<span class='muted'>{esc(interval(cam.get('interval_seconds') or 3600))}</span>",
            f"<span class='muted'>{esc(ago(age))}</span>",
            f"<span class='muted'>{esc(ago(hd.age_of(cam.get('last_processed_at'), now)))}</span>",
            esc(pct(cam.get("success_24h"))),
            f"{esc(int_(cam.get('frames_7d')))} / {esc(int_(cam.get('expected_7d')))}",
            f"<span class='pill {cls}' title='{esc(reason or t('Healthy'))}'>{esc(t(label))}</span>",
        ])
        if key in ("delayed", "down"):
            if cam.get("last_error"):
                reason += ("; " if reason else "") + t("last error: {error}", error=cam["last_error"])
            notes.append(t("{name} is {status}: {reason}.", name=name, status=t(key), reason=reason))
    st.markdown(html_table([
        (t("Camera"), ""), (t("Interval"), ""), (t("Last image"), ""), (t("Last processed"), ""),
        (t("24h success"), "num"), (t("Frames (7 d)"), "num"), (t("Status"), ""),
    ], rows), unsafe_allow_html=True)
    intervals = ", ".join(
        f"{cam.get('name') or cam['id']} {interval(cam.get('interval_seconds') or 3600)}" for cam in cams
    )
    rule = t(
        "Healthy = a new frame arrived within {df}× the camera’s own interval. Delayed = no new frame for "
        "more than {df}× the interval. Down = no new frame for more than {dn}× the interval, or {nf} failed "
        "polls in a row. Unknown = no frame on file yet.",
        df=plain(th["delayed_factor"]), dn=plain(th["down_factor"]), nf=th["down_failures"],
    ) + " " + t("Intervals: {list}.", list=intervals)
    # the design's "?" mark: the rule shown is the rule used
    st.markdown(f"**{t('How the status is set')}**", help=rule)
    if notes:
        st.caption(" ".join(notes))
