"""
The Overview page: the camera's summary card with the safety score, the
compliance trend, violations by hour, worth a look, and the sightings
table whose rows open the image.
"""

from __future__ import annotations

import math
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import review_data as rd
from _shared import altair_chart
from features.i18n import register, t
from features.layout import card_title, clickable_table, empty_notice, meta_html, page_header, toolbar
from features.state import (
    GREEN,
    GREY,
    MISSING,
    PER_PAGE,
    RED,
    S,
    bounds,
    cam_frames,
    current_cam,
    fd,
    frames_in_range,
    go,
    go_images,
    is_dark,
    item_label,
    lang,
    no_item,
    pct,
    range_label,
    reset_view,
    scope,
    tz,
    tz_label,
    zoom_to,
)


FR = {
    'camera id': 'id caméra',
    'To': 'Au',
    '{n} images in range': '{n} images dans la période',
    '{n} on file': '{n} au total',
    'Safety score': 'Score de sécurité',
    'No people seen': 'Aucune personne vue',
    '✓ Good': '✓ Bon',
    '! Watch': '! À surveiller',
    '✕ Critical': '✕ Critique',
    'No previous period to compare': 'Aucune période précédente à comparer',
    '{arrow} {diff} pts vs prior day': '{arrow} {diff} pts vs jour précédent',
    '{arrow} {diff} pts vs prior {days} days': '{arrow} {diff} pts vs {days} jours précédents',
    '{ok} of {n} people fully compliant ({pct})': '{ok} personnes sur {n} entièrement conformes ({pct})',
    '{n} label unreadable, not counted': '{n} étiquette illisible, non comptée',
    '{n} labels unreadable, not counted': '{n} étiquettes illisibles, non comptées',
    'weight': 'poids',
    'Images:': 'Images :',
    'Nearly every worker is flagged for {item} on this camera ({own}), far above the other cameras ({other}). Check the detector or the camera view before acting on it.': 'Presque tous les travailleurs sont signalés pour « {item} » sur cette caméra ({own}), bien plus que sur les autres ({other}). Vérifiez le détecteur ou le cadrage avant d’agir.',
    'See the images': 'Voir les images',
    'Compliance trend': 'Tendance de conformité',
    'Pick a longer range to see a trend.': 'Choisissez une période plus longue pour voir une tendance.',
    'Violations by hour of day ({tz})': 'Infractions par heure de la journée ({tz})',
    'Share of people non-compliant in each hour.': 'Part des personnes non conformes à chaque heure.',
    'Hour': 'Heure',
    'Worth a look': 'À examiner',
    'Most missed': 'Le plus manquant',
    'Worst hour': 'Pire heure',
    'Worst day': 'Pire jour',
    '{h}:00 – {pct} non-compliant ({bad} of {all})': '{h} h – {pct} non conformes ({bad} sur {all})',
    '{day} – {pct} compliant': '{day} – {pct} conformes',
    'Pick a longer range': 'Choisissez une période plus longue',
    'Worker sightings': 'Observations de travailleurs',
    '{n} in range': '{n} dans la période',
    'Time ({tz})': 'Heure ({tz})',
    'Person': 'Personne',
    'Result': 'Résultat',
    'Missing PPE': 'EPI manquant',
    'Detector conf.': 'Conf. détecteur',
    'In frame': 'Dans l’image',
    'Compliant': 'Conforme',
    'Non-compliant': 'Non conforme',
    'Unknown': 'Inconnu',
    'label unreadable': 'étiquette illisible',
    '{a}–{b} of {n}': '{a}–{b} sur {n}',
    'Previous': 'Précédent',
    'Next': 'Suivant',
    'No people detected in this range.': 'Aucune personne détectée pour cette période.',
    '{n} images': '{n} images',
    'Captured': 'Capturée',
    'conf': 'conf.',
    'fully compliant': 'entièrement conforme',
}
register(FR)


def overview_page() -> None:
    toolbar(show_cam=True)
    cam = current_cam()
    page_header(None)
    empty_notice([cam])

    a, b = bounds()
    z = tz()
    frames = frames_in_range(cam)
    s = rd.stats(frames)
    prior = rd.stats(frames_in_range(cam, prior=True))

    with st.container(key="card-summary"):
        summary_card(cam, frames, s, prior, a, b)

    dark = is_dark()
    keys, values, hourly = rd.trend(s, a, b, z)
    bars = rd.hour_bars(s, cam_frames(cam), z)
    trend_col, hour_col = st.columns(2, gap="medium")
    with trend_col, st.container(key="card-trend"):
        card_title(t("Compliance trend"))
        if sum(v is not None for v in values) > 1:
            altair_chart(trend_chart(keys, values, hourly, z), dark)
        else:
            st.caption(t("Pick a longer range to see a trend."))
    with hour_col, st.container(key="card-hours"):
        card_title(t("Violations by hour of day ({tz})", tz=tz_label()))
        df = pd.DataFrame([
            {"hour": h, "share": bad / total if total else 0, "bad": bad, "all": total}
            for h, bad, total in bars
        ])
        chart = alt.Chart(df).mark_bar(cornerRadius=2).encode(
            x=alt.X("hour:O", title=None, axis=alt.Axis(labelAngle=0, grid=False)),
            y=alt.Y("share:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(alt.datum.share > 0.35, alt.value(RED), alt.value(GREY)),
            tooltip=[
                alt.Tooltip("hour:O", title=t("Hour")),
                alt.Tooltip("bad:Q", title=t("Non-compliant")),
                alt.Tooltip("all:Q", title=t("People")),
            ],
        ).properties(height=220, width="container")
        altair_chart(chart, dark)
        st.caption(t("Share of people non-compliant in each hour."))

    with st.container(key="card-worth"):
        card_title(t("Worth a look"))
        top = rd.top_missing(s)
        wh = rd.worst_hour(bars)
        wd = None if hourly else rd.worst_day(s, keys, values, z)
        st.markdown(meta_html([
            (t("Most missed"), f"{item_label(top)} ({s.miss[top]})" if top else "–"),
            (t("Worst hour"), t(
                "{h}:00 – {pct} non-compliant ({bad} of {all})", h=wh[0], pct=pct(wh[1]), bad=wh[2], all=wh[3]
            ) if wh else "–"),
            (t("Worst day"), t(
                "{day} – {pct} compliant", day=fd(rd.day_start(wd[0], z)), pct=pct(wd[1])
            ) if wd else (t("Pick a longer range") if hourly else "–")),
        ]), unsafe_allow_html=True)

    with st.container(key="card-sightings"):
        sightings_section(frames)


def summary_card(cam: str, frames: list[rd.Frame], s: rd.Stats, prior: rd.Stats, a: datetime, b: datetime) -> None:
    left, right = st.columns([5, 6], gap="large")
    with left:
        facts = [rd.SITE, t("{n} images in range", n=len(frames)), t("{n} on file", n=len(cam_frames(cam)))]
        if frames:
            first, last = frames[0], frames[-1]
            facts.append(f"{first.w}×{first.h} · {fd(first.at)} – {fd(last.at)}")
        facts.append(f"{t('camera id')} {cam}")
        st.markdown(
            f"<div class='eyebrow'>{t('Camera')}</div><div class='camname'>{rd.camera_name(cam)}</div>"
            f"<div class='facts'>{' · '.join(facts)}</div>",
            unsafe_allow_html=True,
        )
        note = rd.detector_note(s, [rd.stats(frames_in_range(o)) for o in scope() if o != cam])
        if note:
            key, own, other = note
            st.warning(t(
                "Nearly every worker is flagged for {item} on this camera ({own}), far above the "
                "other cameras ({other}). Check the detector or the camera view before acting on it.",
                item=item_label(key).lower(), own=pct(own), other=pct(other),
            ))
            st.button(t("See the images"), key="detector_images", on_click=go_images, args=(key,))

    with right:
        sc, ps = rd.score(s), rd.score(prior)
        cls = rd.score_class(sc)
        status = {"good": t("✓ Good"), "warn": t("! Watch"), "crit": t("✕ Critical")}.get(cls, t("No people seen"))
        sub = []
        if sc is not None:
            if ps is None:
                sub.append(f"<span>{t('No previous period to compare')}</span>")
            else:
                diff = sc - ps
                days = max(1, round((b - a).total_seconds() / 86400))
                klass = "up" if diff >= 0.05 else "down" if diff <= -0.05 else ""
                key = "{arrow} {diff} pts vs prior day" if days == 1 else "{arrow} {diff} pts vs prior {days} days"
                text = t(key, arrow="▲" if diff >= 0 else "▼", diff=rd.nf1(abs(diff), lang()), days=days)
                sub.append(f"<span class='{klass}'>{text}</span>")
            sub.append("<span>" + t(
                "{ok} of {n} people fully compliant ({pct})", ok=s.ok, n=s.known, pct=pct(s.ok / s.known)
            ) + "</span>")
        if s.unk:
            key = "{n} label unreadable, not counted" if s.unk == 1 else "{n} labels unreadable, not counted"
            sub.append(f"<span>{t(key, n=s.unk)}</span>")
        rows = []
        for key in rd.enabled("person"):
            share = s.miss[key] / s.known if s.known else 0
            rows.append(
                f"<div class='row'><span><b>{t(MISSING[key])}</b> · {t('weight')} {rd.WEIGHTS.get(key, 1)}</span>"
                f"<span class='track'><i style='width:{round(share * 100)}%'></i></span>"
                f"<span class='n'>{s.miss[key]} ({pct(share)})</span></div>"
            )
        hero = "—" if sc is None else f"{rd.nf1(sc, lang())}<span class='unit'>/ 100</span>"
        st.markdown(
            f"<div class='eyebrow'>{t('Safety score')} · {range_label()}</div>"
            f"<div><span class='hero'>{hero}</span><span class='status {cls}'>{status}</span></div>"
            f"<div class='meter {cls}'><span class='fill' style='width:{sc or 0}%'></span></div>"
            f"<div class='score-sub'>{''.join(sub)}</div><div class='items'>{''.join(rows)}</div>",
            unsafe_allow_html=True,
        )
        with st.container(horizontal=True, gap="small", key="see_images"):
            st.caption(t("Images:"))
            for key in rd.enabled("person"):
                st.button(no_item(key), key=f"gomiss_{key}", on_click=go_images, args=(key,))


def trend_chart(keys: list[str], values: list[float | None], hourly: bool, z) -> alt.LayerChart:
    """Compliance per day (or per hour for a one-day range), 0 to 100 % like the design."""

    # The axis spans the whole range, so it is proportional to time; the line joins the
    # buckets where somebody was seen, as on the design page.
    if hourly:
        when = [int(k) for k in keys]
        x = alt.X(
            "when:O", title=None, scale=alt.Scale(domain=when),
            axis=alt.Axis(labelAngle=0, labelExpr="datum.value + ':00'", grid=False),
        )
        when_tip = alt.Tooltip("when:O", title=t("Hour"))
    else:
        when = [rd.day_start(k, z).replace(tzinfo=None) for k in keys]
        x = alt.X(
            "when:T", title=None, scale=alt.Scale(domain=[when[0], when[-1]]),
            axis=alt.Axis(labelAngle=0, format="%d %b", grid=False),
        )
        when_tip = alt.Tooltip("when:T", title=t("Captured"), format="%d %b")
    df = pd.DataFrame([
        {"when": w, "compliance": round(v * 100, 1)} for w, v in zip(when, values) if v is not None
    ])
    base = alt.Chart(df).encode(
        x=x,
        y=alt.Y(
            "compliance:Q", title=None, scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(labelExpr="datum.value + '%'", values=[0, 50, 100]),
        ),
        tooltip=[when_tip, alt.Tooltip("compliance:Q", title=t("Compliance"), format=".1f")],
    )
    line = base.mark_line(color=GREEN, strokeWidth=2.5)
    points = base.mark_circle(color=GREEN, size=40)
    return (line + points).properties(height=220, width="container")


def sightings_section(frames: list[rd.Frame]) -> None:
    """Every person seen in the range, newest first, 25 to a page; a row opens the image."""

    sightings = [(f, p, k) for f in reversed(frames) for k, p in enumerate(f.people)]
    card_title(t("Worker sightings"), t("{n} in range", n=len(sightings)))
    if not sightings:
        st.caption(t("No people detected in this range."))
        return

    pages = max(1, math.ceil(len(sightings) / PER_PAGE))
    page = min(S.log_page, pages)
    start = (page - 1) * PER_PAGE
    chunk = sightings[start:start + PER_PAGE]
    pills = {
        "bad": f"<span class='pill viol'><span class='i'>✕</span>{t('Non-compliant')}</span>",
        "unk": f"<span class='pill unk'><span class='i'>?</span>{t('Unknown')}</span>",
        "ok": f"<span class='pill ok'><span class='i'>✓</span>{t('Compliant')}</span>",
    }
    rows = []
    for i, (f, p, _) in enumerate(chunk):
        tags = "".join(f"<span class='tag'>{item_label(x)}</span>" for x in p.missing_enabled)
        if not tags:
            tags = f"<span class='tag'>{t('label unreadable')}</span>" if p.unread else "<span class='d'>—</span>"
        conf = "—" if p.conf is None else rd.nf2(p.conf, lang())
        rows.append(
            f"<tr data-i='{i}'><td><span class='d'>{fd(f.at)}</span><b>{rd.tm(f.at, tz())}</b></td>"
            f"<td><b>P{p.no}</b></td><td>{pills[p.state]}</td><td>{tags}</td>"
            f"<td class='num'>{conf}</td><td class='num'><b>{len(f.people)}</b></td></tr>"
        )
    picked = clickable_table(
        [(t("Time ({tz})", tz=tz_label()), False), (t("Person"), False), (t("Result"), False),
         (t("Missing PPE"), False), (t("Detector conf."), True), (t("In frame"), True)],
        rows,
        key=f"log_table_{page}_{S.pick}",
    )
    if picked is not None and picked < len(chunk):
        f, p, k = chunk[picked]
        S.pick += 1
        reset_view()
        S.filter = "all"
        S.sel = frames.index(f)
        S.person = k
        zoom_to(p.bbox, f.w / f.h)
        go("images")

    text_col, prev_col, next_col = st.columns([4, 1, 1])
    text_col.caption(t("{a}–{b} of {n}", a=start + 1, b=start + len(chunk), n=len(sightings)))
    prev_col.button(
        t("Previous"), key="log_prev", disabled=page <= 1, use_container_width=True,
        on_click=lambda: S.__setitem__("log_page", page - 1),
    )
    next_col.button(
        t("Next"), key="log_next", disabled=page >= pages, use_container_width=True,
        on_click=lambda: S.__setitem__("log_page", page + 1),
    )
