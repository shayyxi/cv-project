"""The attention line and the executive overview tiles."""

from __future__ import annotations

from datetime import datetime
from html import escape as esc

import streamlit as st

import health_data as hd
from features.format import fmt_time, int_, num, pct, reason_for, sec, span
from features.i18n import register, t
from features.widgets import section_head, tile


FR = {
    'Needs attention': 'À surveiller',
    'all cameras healthy': 'toutes les caméras saines',
    'WordPress API': 'API WordPress',
    'rejected {n} delivery in the last 24 h (last: HTTP {status} at {time})': 'a rejeté {n} livraison dans les dernières 24 h (dernière : HTTP {status} à {time})',
    'rejected {n} deliveries in the last 24 h (last: HTTP {status} at {time})': 'a rejeté {n} livraisons dans les dernières 24 h (dernière : HTTP {status} à {time})',
    'no failed delivery in the last 24 h': 'aucune livraison échouée dans les dernières 24 h',
    'Scheduler': 'Planificateur',
    'active, last successful cycle {age} ago': 'actif, dernier cycle réussi il y a {age}',
    'stopped, last successful cycle {age} ago': 'arrêté, dernier cycle réussi il y a {age}',
    'last cycle failed: {error}': 'dernier cycle en échec : {error}',
    'Executive overview': "Vue d'ensemble",
    'Rolling {n}-day figures unless stated': 'Chiffres glissants sur {n} jours sauf indication contraire',
    'Pipeline uptime': 'Disponibilité du pipeline',
    'Share of the window during which the scheduler kept cycling (a gap longer than two polling intervals counts as downtime)': 'Part de la période pendant laquelle le planificateur a tourné (un trou de plus de deux intervalles compte comme une panne)',
    'Last {n} days': '{n} derniers jours',
    'Active cameras': 'Caméras actives',
    '{h} healthy': '{h} saine(s)',
    '{d} delayed': '{d} en retard',
    '{x} down': '{x} hors service',
    '{u} unknown': '{u} inconnue(s)',
    'Processing success': 'Réussite du traitement',
    'WordPress delivery': 'Livraison WordPress',
    'Eligible attempts': 'Tentatives admissibles',
    'Delivered / (delivered + failed); images with nobody in the frame are never sent and not counted': 'Livrées / (livrées + échouées) ; les images sans personne ne sont jamais envoyées et ne comptent pas',
    'P95 processing': 'Traitement P95',
    'P50: {v}': 'P50 : {v}',
    'Pending jobs': 'Tâches en attente',
    'Oldest: {age}': 'La plus ancienne : {age}',
    'none': 'aucune',
    'Last successful poll': 'Dernière interrogation réussie',
    'Scheduler active': 'Planificateur actif',
    'Scheduler stopped': 'Planificateur arrêté',
    'Images processed': 'Images traitées',
    '{p} of expected frames received': '{p} des images attendues reçues',
    'Camera': 'Caméra',
    'delayed': 'en retard',
    'Delivered': 'Livrées',
    'HTTP': 'HTTP',
    'processing': 'traitement',
    'delivery': 'livraison',
    '{n} d': '{n} j',
    '{age} ago': 'il y a {age}',
}
register(FR)


def attention_section(data: dict, now: datetime, th: dict, cams: list[dict]) -> bool:
    items = []
    unhealthy = [(cam, *hd.camera_status(cam, th, now)) for cam in cams]
    unhealthy = [(cam, key, age) for cam, key, age in unhealthy if key in ("delayed", "down")]
    if unhealthy:
        for cam, key, age in unhealthy:
            dot = "crit" if key == "down" else "warn"
            name = esc(f"{cam.get('name') or cam['id']} ({cam['id']})")
            items.append(f"<span><i class='sdot {dot}'></i><strong>{name}</strong> {esc(reason_for(cam, key, age))}</span>")
    else:
        items.append(f"<span><i class='sdot'></i><strong>{esc(t('Camera'))}</strong> {esc(t('all cameras healthy'))}</span>")

    rejected = hd.recent_delivery_failures(data, now)
    if rejected:
        last = rejected[0]
        key = (
            "rejected {n} delivery in the last 24 h (last: HTTP {status} at {time})" if len(rejected) == 1
            else "rejected {n} deliveries in the last 24 h (last: HTTP {status} at {time})"
        )
        text = t(
            key, n=len(rejected),
            status="–" if last.get("http_status") is None else last["http_status"],
            time=fmt_time(hd.parse_ts(last.get("at"))),
        )
        items.append(f"<span><i class='sdot crit'></i><strong>{esc(t('WordPress API'))}</strong> {esc(text)}</span>")
    else:
        items.append(
            f"<span><i class='sdot'></i><strong>{esc(t('WordPress API'))}</strong> "
            f"{esc(t('no failed delivery in the last 24 h'))}</span>"
        )

    running, last_ok = hd.scheduler_running(data, now)
    scheduler = data.get("scheduler") or {}
    if running:
        text = t("active, last successful cycle {age} ago", age=span(last_ok))
    else:
        text = t("stopped, last successful cycle {age} ago", age="–" if last_ok is None else span(last_ok))
    if scheduler.get("last_cycle_ok") is False and scheduler.get("last_error"):
        text += " · " + t("last cycle failed: {error}", error=scheduler["last_error"])
    dot = "" if running else "crit"
    items.append(f"<span><i class='sdot {dot}'></i><strong>{esc(t('Scheduler'))}</strong> {esc(text)}</span>")

    st.markdown(
        f"<div class='attention' aria-label='{esc(t('Needs attention'))}'>{''.join(items)}</div>",
        unsafe_allow_html=True,
    )
    return running


def overview_section(data: dict, now: datetime, th: dict, cams: list[dict], running: bool) -> None:
    days = data.get("window_days") or 7
    p = data.get("processing") or {}
    d = data.get("delivery") or {}
    counts = hd.status_counts(cams, th, now)
    cam_sub = [t("{h} healthy", h=counts["healthy"])]
    if counts["delayed"]:
        cam_sub.append(t("{d} delayed", d=counts["delayed"]))
    if counts["down"]:
        cam_sub.append(t("{x} down", x=counts["down"]))
    if counts["unknown"]:
        cam_sub.append(t("{u} unknown", u=counts["unknown"]))
    cam_dot = "crit" if counts["down"] else "warn" if counts["delayed"] else "info" if counts["unknown"] else ""
    frames = sum(cam.get("frames_7d") or 0 for cam in cams)
    expected = sum(cam.get("expected_7d") or 0 for cam in cams)
    _, last_ok = hd.scheduler_running(data, now)
    poll_value, _, poll_unit = span(last_ok).partition(" ") if last_ok is not None else ("–", "", "")
    last_days = t("Last {n} days", n=days)

    section_head(t("Executive overview"), t("Rolling {n}-day figures unless stated", n=days))
    tiles = [
        tile(
            t("Pipeline uptime"), "–" if data.get("uptime") is None else pct(data["uptime"]), "", last_days, "",
            t("Share of the window during which the scheduler kept cycling (a gap longer than two polling "
              "intervals counts as downtime)"),
        ),
        tile(t("Active cameras"), f"{counts['healthy'] + counts['delayed']} / {len(cams)}", "", ", ".join(cam_sub), cam_dot),
        tile(t("Processing success"), pct(p.get("success_rate")), "", last_days),
        tile(
            t("WordPress delivery"), pct(d.get("success_rate")), "", t("Eligible attempts"), "",
            t("Delivered / (delivered + failed); images with nobody in the frame are never sent and not counted"),
        ),
        tile(
            t("P95 processing"), "–" if p.get("p95_ms") is None else num(p["p95_ms"] / 1000, 1), "s",
            t("P50: {v}", v=sec(p.get("p50_ms"))),
        ),
        tile(
            t("Pending jobs"), int_(p.get("pending")), "",
            t("Oldest: {age}", age=span(p.get("oldest_pending_seconds")) if p.get("pending") else t("none")),
        ),
        tile(
            t("Last successful poll"), poll_value, poll_unit,
            t("Scheduler active") if running else t("Scheduler stopped"), "" if running else "crit",
        ),
        tile(
            t("Images processed"), int_(p.get("completed")), "",
            t("{p} of expected frames received", p=pct(frames / expected, 0)) if expected else last_days,
        ),
    ]
    st.markdown(f"<div class='tiles'>{''.join(tiles)}</div>", unsafe_allow_html=True)
