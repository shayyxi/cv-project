"""
The page's text in English or French. The English text is the key, so a
string without an entry simply stays English; {placeholders} are filled
by t(). Each section registers its own French strings at import; the
table is the design page's, lifted verbatim. The language follows the
browser (see _shared.init_language).
"""

from __future__ import annotations

from _shared import translator

FR: dict[str, str] = {}


def register(french: dict[str, str]) -> None:
    """Add a section's English -> French strings."""

    FR.update(french)


t = translator(FR)
