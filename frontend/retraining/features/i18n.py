"""
The page's text in English or French. The English text is the key, so a
string without an entry simply stays English; {placeholders} are filled
by t(). Each feature registers its own French strings at import. The
language itself follows the browser (see _shared.init_language).
"""

from __future__ import annotations

import streamlit as st

FR: dict[str, str] = {}


def register(french: dict[str, str]) -> None:
    """Add a feature's English -> French strings."""

    FR.update(french)


def t(text: str, **values) -> str:
    """The page's text in the chosen language; the English is the key."""

    if st.session_state.get("lang") == "fr":
        text = FR.get(text, text)

    return text.format(**values) if values else text
