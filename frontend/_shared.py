"""
Shared pieces of the frontend Streamlit apps: the FINGO theme (dark
only), the browser-language rule, a translator built from an
English-to-French table, the phone layout fix, and the "live JSON next
to the app, else the sample" loader. The retraining page keeps its own
copy of the theme for now.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import streamlit as st

LANGUAGES = {"en": "English", "fr": "Français"}

# Palette and fonts follow the FINGO login page: dark grey surfaces, lime
# green accent, Montserrat. Colours live in .streamlit/config.toml; fonts
# and the finer details are set here.
THEME_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap");

html, body, [class*="st-"], button, input, textarea, select {
    font-family: "Montserrat", system-ui, -apple-system, "Segoe UI", sans-serif;
}
/* Streamlit draws its arrows and other icons with an icon font; keep it. */
[data-testid="stIconMaterial"], [class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
}
h1, h2, h3 {
    font-family: "Montserrat", system-ui, -apple-system, "Segoe UI", sans-serif !important;
}
h1 { font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }
h2, h3 { font-weight: 600; }
code, pre {
    font-family: ui-monospace, "Cascadia Mono", Consolas, monospace !important;
}
[data-testid="stWidgetLabel"] p { font-weight: 600; }
[data-testid="stHeader"] { background: #3b3b3b; }
[data-testid="stCaptionContainer"] { color: #b3b3b3; }
[data-testid="stExpander"], [data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 4px;
}
/* Buttons: outlined in the accent green with uppercase labels, as on the
   login page; the primary button is filled instead. */
.stButton > button, .stDownloadButton > button {
    border-radius: 4px;
    border: 1px solid #8ac646;
    background: transparent;
    color: #ffffff;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #8ac646;
    background: rgba(138, 198, 70, 0.14);
    color: #ffffff;
}
.stButton > button[kind="primary"] { background: #8ac646; color: #1e1e1e; }
.stButton > button[kind="primary"] p { color: #1e1e1e; }
.stButton > button[kind="primary"]:hover { background: #7eb145; border-color: #7eb145; }
.stButton > button:disabled, .stDownloadButton > button:disabled { opacity: 0.45; }
hr { border-color: rgba(255, 255, 255, 0.14); }
/* Hide the hover link icon next to headings. */
[data-testid="stHeaderActionElements"] { display: none; }
/* Segmented controls (DARK | LIGHT, EN | FR): a bordered pill, the
   chosen segment filled in the accent green with dark text. */
[data-testid="stButtonGroup"] {
    gap: 2px;
    padding: 2px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 6px;
    background: #2a2a2a;
}
[data-testid="stButtonGroup"] button[kind="segmented_control"],
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] {
    min-height: 0;
    border: 0 !important;
    border-radius: 4px !important;
    background: transparent;
    padding: 2px 10px;
    color: #ffffff;
}
/* The label is a markdown paragraph with its own 14 px type. */
[data-testid="stButtonGroup"] button p {
    margin: 0;
    font-size: 11px;
    line-height: 1.5;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="stButtonGroup"] button[kind="segmented_control"]:hover {
    background: rgba(138, 198, 70, 0.14);
}
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"]:hover {
    background: #8ac646;
    color: #1e1e1e;
}
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] p,
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] span { color: #1e1e1e; }
/* The sidebar is the menu: darker panel (colours in config.toml), the
   open page highlighted in the accent green. */
[data-testid="stSidebar"] { border-right: 1px solid rgba(255, 255, 255, 0.10); }
/* Collapsed, Streamlit slides the sidebar off-screen and squeezes it to
   nothing. Keep it instead as a narrow rail with the menu icons: text,
   language and dark-mode controls hidden, icons centred. */
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important;
    min-width: 64px !important;
    max-width: 64px !important;
    width: 64px !important;
    overflow: hidden;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] {
    width: 64px !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    /* Streamlit reserves a 10 px scrollbar gutter on both edges. */
    scrollbar-gutter: auto !important;
    overflow: hidden !important;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"],
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavItems"],
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLinkContainer"] {
    width: 64px !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLink"] {
    justify-content: center;
    width: 40px !important;
    height: 40px !important;
    padding: 0 !important;
    margin: 4px auto !important;
    border-radius: 6px;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {
    font-size: 24px !important;
    width: 24px !important;
    height: 24px !important;
}
/* The link is [icon wrapper span][text span]: hide the one without an icon. */
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLink"] > span:not(:has([data-testid="stIconMaterial"])) {
    display: none;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarUserContent"],
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavSeparator"],
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavViewButton"] {
    display: none;
}
[data-testid="stSidebarNavLink"] { border-radius: 4px; }
[data-testid="stSidebarNavLink"] span { font-weight: 600; }
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: rgba(138, 198, 70, 0.18);
}
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: #8ac646; }
</style>
"""


# Light palette of the same dashboard, laid over the dark theme.
LIGHT_CSS = """
<style>
.stApp, [data-testid="stHeader"] { background: #eceef2; }
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3,
.stApp span:not([data-testid="stIconMaterial"]), .stApp summary { color: #141920; }
[data-testid="stCaptionContainer"] { color: #7c8493; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div,
input:not([type="checkbox"]), textarea, [data-testid="stFileUploaderDropzone"] {
    background: #fbfbfc !important; color: #141920 !important;
}
/* Checkboxes: grey when off, accent green when on, white tick. */
[data-baseweb="checkbox"] > div:first-child { background: #c4c9d2 !important; }
[data-baseweb="checkbox"]:has(input:checked) > div:first-child {
    background: #7eb145 !important;
}
[data-baseweb="checkbox"] > div:first-child > div { background: #ffffff !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li { background: #fbfbfc; color: #141920; }
[data-testid="stCode"] pre, [data-testid="stCode"] code { background: #f2f3f6 !important; color: #141920; }
[data-testid="stExpander"], [data-testid="stDataFrame"], hr {
    border-color: rgba(20, 25, 32, 0.11);
}
.stButton > button[kind="primary"] { color: #141920; }
.stButton > button[kind="primary"] span, .stButton > button[kind="primary"] p { color: #141920; }
[data-testid="stSidebar"] {
    background: #ffffff; border-right: 1px solid rgba(20, 25, 32, 0.11);
}
[data-testid="stButtonGroup"] { background: #fbfbfc; border-color: rgba(20, 25, 32, 0.2); }
[data-testid="stButtonGroup"] button[kind="segmented_control"] { color: #141920; }
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"]:hover {
    background: #8ac646; color: #141920;
}
/* Icons are font glyphs and white in the base theme: give them the text
   colour too, except inside the alert boxes, which colour their own. */
.stApp [data-testid="stIconMaterial"] { color: #141920; }
[data-testid="stAlertContainer"] [data-testid="stIconMaterial"] { color: inherit; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: #5f8f2f; }
</style>
"""


def translator(french: dict[str, str]) -> Callable[..., str]:
    """t(): the page's text in the chosen language; the English is the key."""

    def t(text: str, **values) -> str:
        if st.session_state.get("lang") == "fr":
            text = french.get(text, text)
        return text.format(**values) if values else text

    return t


def remember_choice(widget_key: str, store: str) -> None:
    """
    Keeps a segmented control's choice under its own key. Clicking the
    active segment again deselects it (the widget reports None): the
    previous choice is put back instead.
    """

    choice = st.session_state.get(widget_key)

    if choice is None:
        st.session_state[widget_key] = st.session_state[store]
        return

    st.session_state[store] = choice


def init_language() -> str:
    """
    The language follows the browser; ?lang=fr or ?lang=en in the
    address forces one. There is no switch on the pages (reviewer
    decision, October 2026).
    """

    if "lang" not in st.session_state:
        forced = st.query_params.get("lang")
        browser = (st.context.locale or "").lower()
        if forced in LANGUAGES:
            st.session_state["lang"] = forced
        else:
            st.session_state["lang"] = "fr" if browser.startswith("fr") else "en"

    return st.session_state["lang"]


# On a phone Streamlit lays the main section (absolute, full width)
# under the sidebar; the collapsed icon rail of THEME_CSS then covers
# the first 64 px of the content, so the main section starts right of
# the rail instead. The rail's sibling wraps the main section.
PHONE_CSS = """
<style>
@media (max-width: 767px) {
    [data-testid="stSidebar"][aria-expanded="false"] ~ * [data-testid="stMain"] {
        left: 64px !important;
        width: calc(100% - 64px) !important;
    }
}
</style>
"""


def page_setup(title: str, light: bool = False) -> tuple[str, bool]:
    """
    Page config and theme, read before anything is drawn so the first
    paint has the right language. The pages are dark only (the Fingo
    colours); `light` is for a print run. Returns (lang, dark).
    """

    lang = init_language()
    dark = not light

    st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="expanded")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(PHONE_CSS, unsafe_allow_html=True)

    if not dark:
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)

    return lang, dark


def altair_chart(chart, dark: bool) -> None:
    """
    Draws an Altair chart in the page's palette. Streamlit's own chart
    theme follows config.toml (dark); the light palette is only CSS laid
    over it, so a light page gets a transparent chart with dark labels.
    """

    # Transparent, so a chart sits on its card without a lighter box
    # behind it (Streamlit's own chart theme paints the page colour).
    ink = "#d9d9d9" if dark else "#333a46"
    line = "rgba(255, 255, 255, 0.13)" if dark else "rgba(20, 25, 32, 0.11)"
    st.altair_chart(
        chart.configure(background="transparent", font="Montserrat, sans-serif")
        .configure_axis(
            labelColor=ink,
            titleColor=ink,
            gridColor=line,
            domainColor=line,
            tickColor=line,
        )
        .configure_legend(labelColor=ink, titleColor=ink)
        .configure_view(strokeWidth=0),
        theme=None,
    )


def load_json(folder: Path, live: str, sample: str) -> tuple[dict, bool]:
    """
    The live file next to the app when it exists and parses, else the
    sample shipped with it. Returns (data, is_sample).
    """

    for name, is_sample in ((live, False), (sample, True)):
        try:
            return json.loads((folder / name).read_text(encoding="utf-8")), is_sample
        except (OSError, ValueError):
            continue

    return {}, True
