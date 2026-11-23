"""
The page's frame: the file it reads, the top bar with the data line and
the stale badge (re-read every minute), and the footer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape as esc
from pathlib import Path

import streamlit as st

import health_data as hd
from _shared import load_json
from features.format import fmt_datetime, span
from features.i18n import register, t

APP_DIR = Path(__file__).resolve().parents[1]


FR = {
    'Operational dashboard': 'Tableau de bord opérationnel',
    'design sample, mock data only': 'maquette, données fictives seulement',
    'live data': 'données en direct',
    'data as of': 'données au',
    'Stale: data is {age} old': 'Périmé : données vieilles de {age}',
    'Design sample: the figures are illustrative until the pipeline host writes the file.': 'Maquette : les chiffres sont illustratifs tant que l’hôte du pipeline n’écrit pas le fichier.',
    'This page reads {file} next to the app every minute; until the pipeline host writes it, the sample shipped with the app is shown. A file older than {s} is flagged as stale. All times are shown in Eastern time.': 'Cette page lit {file} à côté de l’application chaque minute ; tant que l’hôte du pipeline ne l’écrit pas, l’exemple livré avec l’application est affiché. Un fichier plus vieux que {s} est signalé comme périmé. Toutes les heures sont affichées en heure de l’Est.',
}
register(FR)


DATA_FILE = "health.json"
REFRESH_SECONDS = 60


@st.fragment(run_every=timedelta(seconds=REFRESH_SECONDS))
def header_fragment() -> None:
    """The top bar: the title and the data line, re-read every minute like the page."""

    data, is_sample = load_json(APP_DIR, DATA_FILE, "sample_health.json")
    data["mock"] = is_sample
    gen = hd.parse_ts(data.get("generated_at"))
    label = t("AI Vision System Health").replace("AI Vision – ", "AI Vision|").replace("AI Vision ", "AI Vision|")
    head, _, tail = label.partition("|")
    logo = f"{esc(head)}<span class='sep'>|</span>{esc(tail)}" if tail else esc(label)
    source = t("design sample, mock data only") if data["mock"] else t("live data")
    text = esc(f"{t('Operational dashboard')} · {t('data as of')} {fmt_datetime(gen)} · {source}")
    stale = hd.stale_age(data, datetime.now(timezone.utc))
    if stale is not None:
        text += f"<span class='badge'>{esc(t('Stale: data is {age} old', age=span(stale)))}</span>"
    st.markdown(
        f"<div class='brand'><div class='logo'>{logo}</div><div class='sub'>{text}</div></div>",
        unsafe_allow_html=True,
    )


def footer_section(data: dict) -> None:
    stale_after = data.get("stale_after_seconds") or (data.get("poll_interval_seconds") or 60) * 2
    st.caption(t(
        "This page reads {file} next to the app every minute; until the pipeline host writes it, the sample "
        "shipped with the app is shown. A file older than {s} is flagged as stale. All times are shown in "
        "Eastern time.", file=DATA_FILE, s=span(stale_after),
    ))
    if data["mock"]:
        st.caption(t("Design sample: the figures are illustrative until the pipeline host writes the file."))
