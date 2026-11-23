"""
The Images page: the filters and the strip of frames, the viewer with its
zoom, the image's data with the people detected, and the heatmap of
where violations happen.
"""

from __future__ import annotations

import streamlit as st
from PIL import Image

import review_data as rd
from _shared import remember_choice
from features.frames import draw_highlight, frame_image, heat_overlay, thumb, zoomed
from features.i18n import register, t
from features.layout import card_title, empty_notice, meta_html, page_header, toolbar
from features.state import (
    S,
    current_cam,
    fmt,
    frames_in_range,
    lang,
    no_item,
    on_filter,
    on_strip,
    step,
    toggle_person,
    zoom_by,
)


FR = {
    'Filter': 'Filtre',
    'All images': 'Toutes les images',
    'Has violations': 'Avec infractions',
    'Heatmap': 'Carte de chaleur',
    'No images match this filter.': 'Aucune image ne correspond à ce filtre.',
    'Where violations happen': 'Où surviennent les infractions',
    'All violations': 'Toutes les infractions',
    '{n} non-compliant people in this range, more at the red spots. Cameras are fixed, so hot spots are real locations. Pick a filter to go back to the images.': '{n} personnes non conformes dans la période, plus nombreuses aux points rouges. Les caméras sont fixes, les points chauds sont donc de vrais emplacements. Choisissez un filtre pour revenir aux images.',
    'Background: latest frame in the range, {when}.': 'Fond : dernière image de la période, {when}.',
    'viewing {k}': 'image {k}',
    'Slide to move through the images in the range; ◀ ▶ step one at a time.': 'Faites glisser pour parcourir les images de la période ; ◀ ▶ avancent d’une image.',
    'Each bar is one image: red has a violation, green is all compliant, grey is empty.': 'Chaque barre est une image : rouge = infraction, vert = tout conforme, gris = vide.',
    'Annotated image': 'Image annotée',
    'Fit': 'Ajuster',
    'This frame is not on this computer: {file}': 'Cette image n’est pas sur cet ordinateur : {file}',
    'Use ＋ / － to zoom, Fit to see the whole frame. Select a person below to zoom straight to them.': '＋ / － pour zoomer, Ajuster pour voir toute l’image. Sélectionnez une personne ci-dessous pour zoomer directement sur elle.',
    'Data for this image': 'Données de cette image',
    'Resolution': 'Résolution',
    'People detected': 'Personnes détectées',
    'Person {n}': 'Personne {n}',
    'Select': 'Sélectionner',
    'Selected': 'Sélectionnée',
    'No people detected in this image.': 'Aucune personne détectée dans cette image.',
    'Cameras': 'Caméras',
}
register(FR)


def images_page() -> None:
    toolbar(show_cam=True)
    cam = current_cam()
    page_header(f"{rd.camera_name(cam)} · {rd.SITE}")
    empty_notice([cam])
    frames = frames_in_range(cam)

    options = ["all", "bad", *rd.enabled(), "heat"]
    names = {"all": t("All images"), "bad": t("Has violations"), "heat": t("Heatmap")}
    shown = rd.filter_frames(frames, S.filter) if S.filter != "heat" else []
    n = len(shown)
    if n and (S.get("strip") is None or not 0 <= S.strip < n):
        S.strip = n - 1 if S.sel is None else max(0, min(S.sel, n - 1))
    index = S.strip if n else 0

    # The filter card: the filters, the count with ◀ ▶, the strip of bars.
    with st.container(key="card-filters"):
        filter_col, right_col = st.columns([7, 3.4], vertical_alignment="center")
        filter_col.segmented_control(
            t("Filter"),
            options,
            default=S.filter if S.filter in options else "all",
            format_func=lambda k: names.get(k) or no_item(k),
            key="filter_control",
            on_change=on_filter,
            label_visibility="collapsed",
        )
        with right_col, st.container(
            horizontal=True, gap="small", horizontal_alignment="right", vertical_alignment="center",
            key="stepper",
        ):
            if S.filter == "heat":
                st.markdown(f"<span class='facts'>{t('{n} images in range', n=len(frames))}</span>", unsafe_allow_html=True)
            elif n:
                st.markdown(
                    f"<span class='facts'>{t('{n} images', n=n)} · {t('viewing {k}', k=index + 1)}</span>",
                    unsafe_allow_html=True,
                )
                st.button("◀", key="prev_image", disabled=index <= 0, on_click=step, args=(-1, n))
                st.button("▶", key="next_image", disabled=index >= n - 1, on_click=step, args=(1, n))
        if n:
            bars = []
            for k, f in enumerate(shown):
                cls = "r" if f.bad else "g" if any(p.state == "ok" for p in f.people) else ""
                bars.append(f"<i class='{cls}{' sel' if k == index else ''}' title='{fmt(f.at)}'></i>")
            st.markdown(f"<div class='strip'>{''.join(bars)}</div>", unsafe_allow_html=True)
            if n > 1:
                st.slider(t("Image"), 0, n - 1, key="strip", on_change=on_strip, label_visibility="collapsed")
            st.caption(
                t("Each bar is one image: red has a violation, green is all compliant, grey is empty.")
                + " " + t("Slide to move through the images in the range; ◀ ▶ step one at a time.")
            )

    if S.filter == "heat":
        heatmap_section(frames)
        return
    if not n:
        with st.container(key="card-empty"):
            st.caption(t("No images match this filter."))
        return
    S.sel = index
    cur = shown[index]

    # The viewer card: time and camera left, ＋ － Fit right, the frame under.
    with st.container(key="card-viewer"):
        head_col, zoom_col = st.columns([7, 3], vertical_alignment="center")
        head_col.markdown(
            f"<b style='font-size:16px'>{fmt(cur.at)}</b> <span class='facts'>{rd.camera_name(cam)}</span>",
            unsafe_allow_html=True,
        )
        with zoom_col, st.container(horizontal=True, gap="small", horizontal_alignment="right", key="zoom"):
            st.button("＋", key="zoom_in", disabled=S.zoom >= 8.0, on_click=zoom_by, args=(1.5,))
            st.button("－", key="zoom_out", disabled=S.zoom <= 1.0, on_click=zoom_by, args=(1 / 1.5,))
            st.button(t("Fit"), key="zoom_fit", disabled=S.zoom <= 1.0, on_click=zoom_by, args=(0,))

        img, found = frame_image(cur)
        person = S.person if S.person is not None and S.person < len(cur.people) else None
        shown_img = draw_highlight(img, cur.people[person]) if person is not None else img
        caption = t("Annotated image") + (f" · {S.zoom:.1f}×" if S.zoom > 1.0 else "")
        st.image(zoomed(shown_img, S.zoom, S.zoom_center), caption=caption, use_container_width=True)
        if not found:
            st.caption(t("This frame is not on this computer: {file}", file=cur.file))
        st.caption(t(
            "Use ＋ / － to zoom, Fit to see the whole frame. Select a person below to zoom "
            "straight to them."
        ))

    counts = {"ok": 0, "bad": 0, "unk": 0}
    for p in cur.people:
        counts[p.state] += 1
    meta = [
        (t("Captured"), fmt(cur.at)),
        (t("Resolution"), f"{cur.w} × {cur.h}"),
        (t("People"), len(cur.people)),
        (t("Compliant"), counts["ok"]),
        (t("Non-compliant"), counts["bad"]),
    ]
    if counts["unk"]:
        meta.append((t("Unknown"), counts["unk"]))
    with st.container(key="card-data"):
        card_title(t("Data for this image"))
        st.markdown(meta_html(meta), unsafe_allow_html=True)
        card_title(t("People detected"))
        people_grid(cur, img, person)


def people_grid(cur: rd.Frame, img: Image.Image, person: int | None) -> None:
    if not cur.people:
        st.caption(t("No people detected in this image."))
        return
    cols = st.columns(min(4, len(cur.people)))
    for k, p in enumerate(cur.people):
        with cols[k % len(cols)]:
            st.image(thumb(img, p), width=96)
            if p.missing_enabled:
                chips = "".join(f"<span class='chip'>{no_item(x).lower()}</span>" for x in p.missing_enabled)
            elif p.state == "bad":
                chips = f"<span class='chip'>{t('Non-compliant').lower()}</span>"
                if p.unread:
                    chips += f"<span class='chip u'>{t('label unreadable')}</span>"
            elif p.state == "unk":
                chips = f"<span class='chip u'>{t('label unreadable')}</span>"
            else:
                chips = f"<span class='chip o'>{t('fully compliant')}</span>"
            conf = "—" if p.conf is None else rd.nf2(p.conf, lang())
            st.markdown(
                f"**{t('Person {n}', n=p.no)}** <span class='facts'>{t('conf')} {conf}</span><br>{chips}",
                unsafe_allow_html=True,
            )
            st.button(
                t("Selected") if person == k else t("Select"),
                key=f"person_{k}",
                type="primary" if person == k else "secondary",
                on_click=toggle_person,
                args=(k, p.bbox, cur.w / cur.h),
            )


def heatmap_section(frames: list[rd.Frame]) -> None:
    with st.container(key="card-heat"):
        heatmap_card(frames)


def heatmap_card(frames: list[rd.Frame]) -> None:
    options = ["bad", *rd.enabled("person")]
    title_col, filter_col = st.columns([4, 6], vertical_alignment="center")
    with title_col:
        card_title(t("Where violations happen"))
    filter_col.segmented_control(
        t("Heatmap"),
        options,
        default=S.heat if S.heat in options else "bad",
        format_func=lambda k: t("All violations") if k == "bad" else no_item(k),
        key="heat_control",
        on_change=remember_choice,
        args=("heat_control", "heat"),
        label_visibility="collapsed",
    )
    if not frames:
        st.caption(t("No images match this filter."))
        return
    last = frames[-1]
    points = rd.heat_points(frames, S.heat)
    base, found = frame_image(last)
    st.image(heat_overlay(base, points), use_container_width=True)
    if not found:
        st.caption(t("This frame is not on this computer: {file}", file=last.file))
    st.markdown(
        t(
            "{n} non-compliant people in this range, more at the red spots. Cameras are fixed, so "
            "hot spots are real locations. Pick a filter to go back to the images.", n=len(points)
        ) + " <span class='ramp'></span> " + t("Background: latest frame in the range, {when}.", when=fmt(last.at)),
        unsafe_allow_html=True,
    )


# ---- sidebar and navigation
