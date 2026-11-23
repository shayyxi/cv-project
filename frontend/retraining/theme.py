"""The FINGO look of the retraining page: one CSS block laid over the dark theme of config.toml."""


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
