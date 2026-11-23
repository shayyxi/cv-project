"""
Streamlit page for the site safety review: compliance per camera over
a date range, the images behind the figures and a heatmap of where the
violations happen. A replica of docs/design/ppe-review-dashboard.html.

Start it with:
    streamlit run frontend/site-review/app.py --server.port 8502
or double-click site_review.bat in the project folder.

This file only wires the header and the tabs; each page lives in
features/ (cameras, overview, images) with the shared state, frames,
layout and the report dialog beside them, and the French strings next
to the code that uses them. Frontend only: the page reads results.json
next to this file (the pipeline will write it; same contract as the
design page, see review_data.py) and falls back to sample_results.json.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
# `streamlit run` puts this folder on sys.path; frontend/ holds _shared. This
# folder goes first so `features` is always this one.
sys.path[:0] = [str(APP_DIR), str(APP_DIR.parent)]

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

import review_data as rd  # noqa: E402
from _shared import page_setup  # noqa: E402
from features import state  # noqa: E402
from features.cameras import cams_page  # noqa: E402
from features.i18n import register, t  # noqa: E402
from features.images import images_page  # noqa: E402
from features.layout import brand_html  # noqa: E402
from features.overview import overview_page  # noqa: E402
from features.state import PAGES, S, on_role, scope  # noqa: E402
from theme import LIGHT_PAGE_CSS, PAGE_CSS  # noqa: E402


FR = {
    'Site Safety Review': 'Revue de sécurité du chantier',
    'Eastern': 'Est',
    'Prototype – view as': 'Prototype – rôle',
    'Admin ({n} cameras)': 'Admin ({n} caméras)',
    'Group client ({n} cameras)': 'Client groupe ({n} caméras)',
    'Single-camera client': 'Client à une caméra',
    'All cameras': 'Toutes les caméras',
    'Overview': 'Aperçu',
    'Images': 'Images',
    'Site': 'Chantier',
    'Image': 'Image',
    'Download': 'Télécharger',
    'Helmet': 'Casque',
    'Safety vest': 'Veste de sécurité',
    'Safety boots': 'Bottes de sécurité',
    'Harness': 'Harnais',
    'Guard rails': 'Garde-corps',
    'Rebar protection': 'Protection des armatures',
}
register(FR)


def role_name(role: str) -> str:
    if role == "admin":
        return t("Admin ({n} cameras)", n=len(state.REVIEW.cams))
    if role == "group":
        return t("Group client ({n} cameras)", n=len(rd.scope(state.REVIEW.cams, "group")))
    return t("Single-camera client")


def main() -> None:
    state.init_state()  # session defaults, the clock, results.json: every run
    # Dark only, language from the browser, Eastern time: no switches.
    # A print run (Download) is drawn in the light palette, once.
    _, dark = page_setup(t("Site Safety Review"), light=bool(S.get("print_request")))
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    if not dark:
        st.markdown(LIGHT_PAGE_CSS, unsafe_allow_html=True)
    if S.toast:
        st.toast(S.toast)
        S.toast = ""

    multi = len(scope()) > 1
    pages = []
    if multi:
        pages.append(st.Page(cams_page, title=t("All cameras"), url_path="cams", default=True))
    pages.append(st.Page(overview_page, title=t("Overview"), url_path="overview", default=not multi))
    pages.append(st.Page(images_page, title=t("Images"), url_path="images"))
    PAGES.update({p.url_path: p for p in pages})
    page = st.navigation(pages, position="hidden")

    target = S.pop("goto", None)
    if target in PAGES and PAGES[target] is not page:
        st.switch_page(PAGES[target])

    # The header as on the design page: brand and data line, the "view
    # as" selector (standing in for the login), then the tabs with the
    # page's controls to their right. No sidebar.
    with st.container(key="header"):
        brand_col, role_col = st.columns([8.8, 2.6], vertical_alignment="bottom")
        brand_col.markdown(brand_html(), unsafe_allow_html=True)
        role_col.selectbox(
            t("Prototype – view as"),
            list(rd.ROLES),
            index=rd.ROLES.index(S.role),
            format_func=role_name,
            key="role_control",
            on_change=on_role,
        )
        # Tabs left, the page's controls right; a long French row wraps
        # inside the controls group instead of overlapping.
        with st.container(
            horizontal=True, gap="medium", horizontal_alignment="distribute",
            vertical_alignment="center", key="navrow",
        ):
            with st.container(horizontal=True, gap="small", key="tabs"):
                for item in pages:
                    with st.container(key=f"tab-{item.url_path}"):
                        st.page_link(item, use_container_width=False)
            state.CONTROLS = st.container(
                horizontal=True, gap="small", horizontal_alignment="right",
                vertical_alignment="center", key="toolbar",
            )
        # the open page's tab gets the green underline
        st.markdown(
            f"<style>.st-key-tab-{page.url_path} [data-testid='stPageLink-NavLink']"
            "{border-bottom-color:#8ac646 !important;color:inherit !important}</style>",
            unsafe_allow_html=True,
        )

    page.run()

    if S.pop("print_request", False):
        # After the page is drawn: the browser's print dialog, where
        # "Save as PDF" makes the file.
        components.html(
            "<script>setTimeout(function () { window.parent.print(); }, 800);</script>",
            height=0,
        )


main()
