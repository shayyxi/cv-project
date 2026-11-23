"""The design's building blocks as HTML: section bands, tiles, meters and tables."""

from __future__ import annotations

from html import escape as esc

import streamlit as st

import health_data as hd


def html_table(head: list[tuple[str, str]], rows: list[list[str]]) -> str:
    """head = (label, cell class) per column; the cells are HTML the caller escaped."""

    ths = "".join(f"<th class='{cls}'>{esc(label)}</th>" for label, cls in head)
    trs = "".join(
        "<tr>" + "".join(f"<td class='{cls}'>{cell}</td>" for cell, (_, cls) in zip(row, head)) + "</tr>"
        for row in rows
    )
    return (
        f"<div class='table-wrap'><table class='htable'><thead><tr>{ths}</tr></thead>"
        f"<tbody>{trs}</tbody></table></div>"
    )


def tile(label: str, value: str, unit: str = "", sub: str = "", dot: str = "", help_text: str = "") -> str:
    title = f" title='{esc(help_text)}'" if help_text else ""
    unit_html = f"<small>{esc(unit)}</small>" if unit else ""
    dot_html = f"<i class='sdot {dot}'></i>" if dot else ""
    return (
        f"<div class='tile'><span class='label'{title}>{esc(label)}</span>"
        f"<span class='value'>{esc(value)}{unit_html}</span>"
        f"<span class='sub'>{dot_html}{esc(sub)}</span></div>"
    )


def meter(label: str, value: float | None, warn: float, crit: float) -> str:
    cls = hd.level(value, warn, crit)
    width = max(0, min(100, value or 0))
    shown = "–" if value is None else f"{round(value)}%"
    return (
        f"<div class='meter'><span>{esc(label)}</span><div class='track'>"
        f"<span class='fill {cls}' style='width:{width}%'></span>"
        f"<i class='mark' style='left:{warn}%'></i><i class='mark crit' style='left:{crit}%'></i></div>"
        f"<span class='val'>{shown}</span></div>"
    )


# ---- sections, in the design's order


def section_head(title: str, caption: str) -> None:
    st.markdown(f"<div class='sh'><h2>{esc(title)}</h2><span>{esc(caption)}</span></div>", unsafe_allow_html=True)
