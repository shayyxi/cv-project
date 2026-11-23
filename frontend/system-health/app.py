"""
Streamlit page for the AI Vision system health: cameras, processing,
WordPress delivery, recent failures and the host, read from one JSON
document. A replica of docs/design/system_health_dashboard.html.

Start it with:
    streamlit run frontend/system-health/app.py --server.port 8503
or double-click system_health.bat in the project folder.

This file only draws the sections in the design's order; each section
lives in features/ with the formatting, the HTML widgets and the French
strings next to the code that uses them. Frontend only: the page reads
health.json next to this file every minute (the pipeline host will
write it; same contract as the design page, see health_data.py) and
falls back to sample_health.json. All times are shown in Eastern time.
"""

import sys
from datetime import timedelta
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
# `streamlit run` puts this folder on sys.path; frontend/ holds _shared. This
# folder goes first so `features` is always this one.
sys.path[:0] = [str(APP_DIR), str(APP_DIR.parent)]

import streamlit as st  # noqa: E402

import health_data as hd  # noqa: E402
from _shared import load_json, page_setup  # noqa: E402
from features.cameras import fleet_section  # noqa: E402
from features.failures import failures_section  # noqa: E402
from features.i18n import register, t  # noqa: E402
from features.infrastructure import infrastructure_section  # noqa: E402
from features.overview import attention_section, overview_section  # noqa: E402
from features.page import DATA_FILE, REFRESH_SECONDS, footer_section, header_fragment  # noqa: E402
from features.performance import performance_section  # noqa: E402
from features.pipeline import pipeline_section  # noqa: E402
from theme import PAGE_CSS  # noqa: E402


FR = {
    'AI Vision System Health': 'AI Vision – État du système',
    'never': 'jamais',
    'down': 'hors service',
    'Language': 'Langue',
    'Line chart of P50 and P95 processing latency per day': 'Graphique linéaire de la latence de traitement P50 et P95 par jour',
    'received / expected at this interval': 'reçues / attendues à cet intervalle',
    'Show how the status is set': 'Afficher comment l’état est établi',
    'This page renders the JSON in its health-data block. Served over HTTP it reads {file} next to itself every minute instead; a file older than {s} is flagged as stale. All times are shown in Eastern time.': 'Cette page affiche le JSON de son bloc health-data. Servie en HTTP, elle lit plutôt {file} à côté d’elle chaque minute ; un fichier plus vieux que {s} est signalé comme périmé. Toutes les heures sont affichées en heure de l’Est.',
    'light': 'clair',
}
register(FR)


@st.fragment(run_every=timedelta(seconds=REFRESH_SECONDS))
def dashboard() -> None:
    """The whole page; re-read from disk every minute without a full rerun."""

    data, is_sample = load_json(APP_DIR, DATA_FILE, "sample_health.json")
    data["mock"] = is_sample  # a live file is never treated as the design sample
    now = hd.clock(data)
    th = hd.thresholds(data)
    cams = data.get("cameras") or []

    running = attention_section(data, now, th, cams)
    overview_section(data, now, th, cams, running)
    performance_section(data)
    fleet_section(data, now, th, cams)
    pipeline_section(data)
    failures_section(data)
    infrastructure_section(data, th)
    footer_section(data)


def main() -> None:
    page_setup(t("AI Vision System Health"))
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    # The top bar as on the design page: the title and the data line.
    # Dark only, language from the browser, no switches, no sidebar.
    with st.container(key="header"):
        header_fragment()

    dashboard()


main()
