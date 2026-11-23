"""
The page's text in English or French. The English text is the key, so a
string without an entry simply stays English; {placeholders} are filled
by t(). Each feature registers its own French strings at import. The
language itself follows the browser (see _shared.init_language).
"""

from __future__ import annotations

from _shared import translator

FR: dict[str, str] = {}


def register(french: dict[str, str]) -> None:
    """Add a feature's English -> French strings."""

    FR.update(french)


t = translator(FR)
