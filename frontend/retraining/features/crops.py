"""
The Crop regions page: choose or add a camera, load a frame, draw the
crops by clicking, undo, edit the points, save them to the crop config.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from app.config import settings
from app.retraining.history import load_sahi_settings
from app.retraining.regions import (
    CROP_CONFIG_PATH,
    DEFAULT_LABEL,
    LABEL_COLOURS,
    LABELS,
    Crop,
    crop_preview,
    display_base,
    draw_overlay,
    drawn_crops,
    load_regions,
    nearest_point_index,
    normalise_crops,
    save_region,
    to_full_image,
)
from features.camera_settings import camera_settings_section, ppe_checks_section
from features.i18n import register, t

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # the section explains how to install it
    streamlit_image_coordinates = None


FR = {
    # Crop regions page
    "Marking by clicking needs the streamlit-image-coordinates package: pip "
    "install streamlit-image-coordinates":
        "Le marquage par clic nécessite le paquet streamlit-image-coordinates : "
        "pip install streamlit-image-coordinates",
    "Camera": "Caméra",
    "add a camera...": "ajouter une caméra...",
    "Panomax id of the camera to add": "Identifiant Panomax de la caméra à ajouter",
    "The id is the camera's Panomax id, the same one as in CAMERA_IDS; it cannot "
    "be changed here. The camera must also be added to CAMERA_IDS before its "
    "pictures are downloaded.":
        "L’identifiant est celui de la caméra chez Panomax, le même que dans "
        "CAMERA_IDS ; il ne se modifie pas ici. La caméra doit aussi être ajoutée à "
        "CAMERA_IDS avant le téléchargement de ses images.",
    "Type the camera's Panomax id first.":
        "Saisissez d’abord l’identifiant Panomax de la caméra.",
    "This camera already exists: pick it in the list.":
        "Cette caméra existe déjà : choisissez-la dans la liste.",
    "Frame to mark (jpg or png)": "Image à marquer (jpg ou png)",
    "Use the latest raw frame of this camera instead of an upload":
        "Utiliser la dernière image brute de cette caméra au lieu d'un "
        "téléversement",
    "No raw frames under {folder}.": "Aucune image brute sous {folder}.",
    "Upload a frame, or tick the raw-frame option, to start marking.":
        "Téléversez une image, ou cochez l'option image brute, pour commencer "
        "le marquage.",
    "Crop config could not be read: {error}":
        "La configuration des zones n'a pas pu être lue : {error}",
    "Crops": "Zones",
    "Mark a crop with two or more points to see the model view.":
        "Marquez une zone avec deux points ou plus pour voir la vue du modèle.",
    "Label": "Étiquette",
    "person detection": "détection de personnes",
    "safety": "sécurité",
    "priority": "priorité",
    "New crop": "Nouvelle zone",
    "Delete this crop": "Supprimer cette zone",
    "A click on the picture will": "Un clic sur l'image va",
    "add a point": "ajouter un point",
    "move a point": "déplacer un point",
    "remove a point": "supprimer un point",
    "Point {number} is picked up: click where it should go.":
        "Le point {number} est saisi : cliquez où il doit aller.",
    "No point near that click. Click on one of the numbered points.":
        "Aucun point près de ce clic. Cliquez sur l'un des points numérotés.",
    "Remove last point": "Supprimer le dernier point",
    "Clear points": "Effacer les points",
    "Undo last change": "Annuler le dernier changement",
    "Reset to saved crops": "Revenir aux zones enregistrées",
    "Click on the picture to add the first point of this crop.":
        "Cliquez sur l'image pour ajouter le premier point de cette zone.",
    "What the model will see (black = outside the drawn crops)":
        "Ce que le modèle verra (noir = hors des zones dessinées)",
    "Save crops to config": "Enregistrer les zones dans la configuration",
    "Saved {count} crops for camera {camera} to {path}. The pipeline uses "
    "them from its next frame.":
        "{count} zones enregistrées pour la caméra {camera} dans {path}. Le "
        "pipeline les utilise dès sa prochaine image.",
    "These crops are the saved ones; nothing to save.":
        "Ces zones sont celles enregistrées ; rien à enregistrer.",
    "Crop {number} has a single point: add another or remove the crop.":
        "La zone {number} n'a qu'un seul point : ajoutez-en un autre ou "
        "supprimez la zone.",
}
register(FR)


NEW_CAMERA = "add a camera..."

FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")

# What a click on the picture does.
ADD_POINT = "add a point"
MOVE_POINT = "move a point"
REMOVE_POINT = "remove a point"

# How close (in on-screen pixels) a click must be to pick up a point.
GRAB_RADIUS = 20

# How much the picture is darkened outside the drawn crops.
SHADE = 0.55

# A coloured dot in front of each crop in the list (Streamlit's named
# colours, close to the overlay's).
LABEL_DOTS = {
    "person detection": ":green[●]",
    "safety": ":orange[●]",
    "priority": ":blue[●]",
}


def latest_raw_frame(camera: str) -> Path | None:
    folder = settings.local_raw_dir / camera

    if not folder.is_dir():
        return None

    frames = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES
    ]

    return max(frames, key=lambda path: path.stat().st_mtime) if frames else None


def load_frame(source: str, read_bytes) -> tuple[Image.Image, Image.Image, float]:
    """
    (full frame, display-sized copy, scale) for the chosen source,
    kept in the session so clicks do not re-decode a 5 MB frame.
    """

    cache = st.session_state.get("region_frame_cache")

    if cache and cache["source"] == source:
        return cache["image"], cache["base"], cache["scale"]

    import io

    image = Image.open(io.BytesIO(read_bytes()))
    image.load()

    base, scale = display_base(image)

    st.session_state["region_frame_cache"] = {
        "source": source, "image": image, "base": base, "scale": scale,
    }

    return image, base, scale


def remember(camera: str, crops: list[Crop], active: int) -> None:
    """A snapshot of the working copy before a change, for Undo last change."""

    stack = st.session_state.setdefault("region_undo", {}).setdefault(camera, [])
    stack.append((clone_crops(crops), active))
    del stack[:-50]


def region_section() -> None:
    st.title(t("Crop regions"))

    flash = st.session_state.pop("region_flash", None)

    if flash:
        st.success(flash)

    if streamlit_image_coordinates is None:
        st.warning(
            t(
                "Marking by clicking needs the streamlit-image-coordinates "
                "package: pip install streamlit-image-coordinates"
            )
        )
        return

    try:
        regions = load_regions()
    except ValueError as error:  # a hand edit with an unknown label
        st.error(t("Crop config could not be read: {error}", error=error))
        return

    try:
        default_sahi, sahi_cameras = load_sahi_settings()
    except (OSError, ValueError, KeyError) as error:
        st.error(t("The vision config could not be read: {error}", error=error))
        return

    # A camera is known from its crops or from its settings.
    cameras = sorted(set(regions) | set(sahi_cameras))

    left, right = st.columns(2)

    with left:
        # The values stay English (NEW_CAMERA is compared below); only
        # the shown text follows the language.
        choice = st.selectbox(
            t("Camera"),
            cameras + [NEW_CAMERA],
            format_func=t,
            key="region_camera",
        )

    with right:
        typed = st.text_input(
            t("Panomax id of the camera to add"),
            key="region_new_id",
            disabled=choice != NEW_CAMERA,
        )

    camera = typed.strip() if choice == NEW_CAMERA else choice

    if choice == NEW_CAMERA:
        st.caption(t(
            "The id is the camera's Panomax id, the same one as in CAMERA_IDS; it cannot "
            "be changed here. The camera must also be added to CAMERA_IDS before its "
            "pictures are downloaded."
        ))

        if not camera:
            st.info(t("Type the camera's Panomax id first."))
            return

        if camera in cameras:
            st.warning(t("This camera already exists: pick it in the list."))
            return

    camera_settings_section(camera, default_sahi, sahi_cameras.get(camera))
    ppe_checks_section(camera)

    uploaded = st.file_uploader(
        t("Frame to mark (jpg or png)"),
        type=["jpg", "jpeg", "png"],
        key="region_upload",
    )

    use_raw = st.checkbox(
        t("Use the latest raw frame of this camera instead of an upload"),
        key="region_use_raw",
    )

    frame = base = None
    scale = 1.0
    source_name = ""

    if use_raw and camera:
        raw = latest_raw_frame(camera)

        if raw is None:
            st.warning(
                t("No raw frames under {folder}.", folder=settings.local_raw_dir / camera)
            )
        else:
            frame, base, scale = load_frame(str(raw), raw.read_bytes)
            source_name = raw.name

    elif uploaded is not None:
        source = f"upload:{uploaded.name}:{uploaded.size}"
        frame, base, scale = load_frame(source, uploaded.getvalue)
        source_name = uploaded.name

    if frame is None:
        st.caption(t("Upload a frame, or tick the raw-frame option, to start marking."))
        return

    marks = st.session_state.setdefault("region_marks", {})
    actives = st.session_state.setdefault("region_active", {})
    # The file also holds the derived exclude crops; only the drawn ones
    # are edited here (Save computes the exclude crops again).
    saved_all = regions.get(camera, [])
    saved = drawn_crops(saved_all)

    if camera not in marks:
        # The saved crops start as the working copy (copies, never the
        # loaded objects), so they can be adjusted instead of drawn
        # again. A camera without any starts with one empty person crop.
        marks[camera] = clone_crops(saved) or [Crop()]
        actives[camera] = 0

    crops = marks[camera]

    if not crops:
        crops.append(Crop())

    active = actives[camera] = min(actives.get(camera, 0), len(crops) - 1)
    crop = crops[active]
    points = crop.points

    # A workspace: the picture on the left, the inspector with the crop
    # list and the controls on the right. The inspector is written first
    # (its choices decide what the picture shows); Streamlit lets a
    # column be written to more than once, so the rest follows below.
    canvas_col, panel = st.columns([1, 1], gap="large")

    with panel:
        st.markdown(legend_html(), unsafe_allow_html=True)

        # Keyless, with their index from the working copy, so a changed
        # crop or label makes a fresh widget: nothing can leak from one
        # crop to another after New, Delete or Reset.
        chosen = st.radio(
            t("Crops"),
            list(range(len(crops))),
            index=active,
            format_func=lambda index: t(
                "{dot} {number} · {label} · {count} pts",
                dot=LABEL_DOTS.get(crops[index].label, ""),
                number=index + 1,
                label=t(crops[index].label),
                count=len(crops[index].points),
            ),
        )

        if chosen != active:
            actives[camera] = chosen
            st.session_state["region_grab"] = None
            st.rerun()

        new, delete = st.columns(2)

        if new.button(t("New crop"), width="stretch"):
            # The first crop is nearly always the person area, then a
            # safety and a priority zone: the first label without a crop
            # yet, or person again; the label can be changed below.
            remember(camera, crops, active)
            used = {c.label for c in crops}
            crops.append(
                Crop(next((name for name in LABELS if name not in used), DEFAULT_LABEL))
            )
            actives[camera] = len(crops) - 1
            st.session_state["region_grab"] = None
            st.rerun()

        if delete.button(
            t("Delete this crop"),
            disabled=len(crops) == 1 and not points,
            width="stretch",
        ):
            remember(camera, crops, active)
            del crops[active]

            if not crops:
                crops.append(Crop())

            actives[camera] = min(active, len(crops) - 1)
            st.session_state["region_grab"] = None
            st.rerun()

        label = st.selectbox(
            t("Label"), list(LABELS), index=LABELS.index(crop.label), format_func=t
        )

        if label != crop.label:
            remember(camera, crops, active)
            crop.label = label
            st.rerun()

        mode = st.radio(
            t("A click on the picture will"),
            [ADD_POINT, MOVE_POINT, REMOVE_POINT],
            format_func=t,
            horizontal=True,
            key="region_click_mode",
        )

        # The point picked up in move mode: (camera, crop index, point
        # index), or None.
        grab = st.session_state.get("region_grab")

        if grab and (
            grab[0] != camera
            or grab[1] != active
            or mode != MOVE_POINT
            or grab[2] >= len(points)
        ):
            grab = st.session_state["region_grab"] = None

        picked = grab[2] if grab else None

        if picked is not None:
            st.info(
                t("Point {number} is picked up: click where it should go.", number=picked + 1)
            )

        note = st.session_state.pop("region_note", None)

        if note:
            st.warning(note)

    with canvas_col:
        st.caption(
            t(
                "{name}, {width}x{height} px.",
                name=source_name,
                width=frame.size[0],
                height=frame.size[1],
            )
        )

        # Outside the drawn crops the picture is darkened: the part the
        # model will not see.
        overlay, scale = draw_overlay(
            frame, crops, active=active, highlight=picked, base=base, shade=SHADE
        )

        # The picture scales to the column (it is never clipped); the
        # click handler below maps on-screen pixels back through the
        # reported width.
        click = streamlit_image_coordinates(
            overlay, key=f"region-canvas-{camera}", image_format="JPEG", width="stretch"
        )

        # The component's value carries unix_time, so a new click on the
        # same pixel (for another crop, say) differs from the stale one;
        # comparing the whole dict is what makes this dedupe work.
        if click and click != st.session_state.get("region_last_click"):
            st.session_state["region_last_click"] = click

            # The component reports on-screen pixels and the on-screen
            # size of the picture; the picture itself is the display-sized
            # copy.
            shown_width = click.get("width") or overlay.size[0]
            css_scale = overlay.size[0] / shown_width

            clicked = to_full_image(click["x"] * css_scale, click["y"] * css_scale, scale)
            before = clone_crops(crops)
            apply_click(mode, clicked, points, picked, camera, active, GRAB_RADIUS * scale)

            if crops != before:  # picking up a point to move is not a change yet
                remember(camera, before, active)

            st.rerun()

    with panel:
        remove_last, clear = st.columns(2)

        if remove_last.button(t("Remove last point"), disabled=not points, width="stretch"):
            remember(camera, crops, active)
            points.pop()
            st.session_state["region_grab"] = None
            st.rerun()

        if clear.button(t("Clear points"), disabled=not points, width="stretch"):
            remember(camera, crops, active)
            points.clear()
            st.session_state["region_grab"] = None
            st.rerun()

        # Every change above (and a click, a typed coordinate) is pushed
        # by remember(); this pops the last one.
        history = st.session_state.get("region_undo", {}).get(camera, [])

        if st.button(t("Undo last change"), disabled=not history, width="stretch"):
            previous, previous_active = history.pop()
            marks[camera] = previous or [Crop()]
            actives[camera] = min(previous_active, len(marks[camera]) - 1)
            st.session_state["region_grab"] = None
            st.rerun()

        if st.button(
            t("Reset to saved crops"),
            disabled=not saved or crops == saved,
            width="stretch",
        ):
            remember(camera, crops, active)
            marks[camera] = clone_crops(saved)
            actives[camera] = 0
            st.session_state["region_grab"] = None
            st.rerun()

        if points:
            typed_points = edit_points_table(camera, active, points, frame.size)

            if typed_points != points:
                remember(camera, crops, active)
                crop.points = typed_points
                st.rerun()
        else:
            st.caption(t("Click on the picture to add the first point of this crop."))

    # What Save would write: nothing while a crop has a single point, or
    # while no crop has two points yet.
    single = next((i for i, c in enumerate(crops) if len(c.points) == 1), None)
    pending = None

    if single is None:
        try:
            pending = normalise_crops(crops, frame.size)
        except ValueError:
            pending = None

    with canvas_col:
        if any(len(c.points) >= 2 for c in crops):
            # Preview on the display-sized copy: same shape, far cheaper.
            small = [
                Crop(c.label, [(x / scale, y / scale) for x, y in c.points])
                for c in crops
            ]
            st.image(
                crop_preview(base, small),
                caption=t("What the model will see (black = outside the drawn crops)"),
                use_container_width=True,
            )
        else:
            st.caption(t("Mark a crop with two or more points to see the model view."))

    unchanged = pending is not None and pending == saved_all

    with panel:
        if st.button(
            t("Save crops to config"),
            type="primary",
            disabled=pending is None or unchanged,
            width="stretch",
        ):
            written = save_region(camera, crops, image_size=frame.size)
            # Forget the working copy so the saved crops are reloaded.
            marks.pop(camera, None)
            actives.pop(camera, None)
            st.session_state.get("region_undo", {}).pop(camera, None)
            st.session_state["region_grab"] = None
            st.session_state["region_flash"] = t(
                "Saved {count} crops for camera {camera} to {path}. The pipeline "
                "uses them from its next frame.",
                count=len(drawn_crops(written)),
                camera=camera,
                path=CROP_CONFIG_PATH,
            )
            st.rerun()

        if single is not None:
            st.caption(
                t(
                    "Crop {number} has a single point: add another or remove the crop.",
                    number=single + 1,
                )
            )
        elif unchanged:
            st.caption(t("These crops are the saved ones; nothing to save."))


def clone_crops(crops: list[Crop]) -> list[Crop]:
    """Working copies of saved crops, so edits never touch the loaded list."""

    return [Crop(crop.label, list(crop.points)) for crop in crops]


def legend_html() -> str:
    """The drawable labels as colour chips, in the overlay's colours."""

    chips = "".join(
        '<span style="display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0">'
        '<span style="display:inline-block;width:12px;height:12px;border-radius:3px;'
        f'background:rgb{LABEL_COLOURS[label]}"></span>'
        f'<span style="font-size:0.85rem">{t(label)}</span></span>'
        for label in LABELS
    )

    return f'<div style="margin:0 0 6px">{chips}</div>'


def apply_click(
    mode: str,
    clicked: tuple[int, int],
    points: list,
    picked: int | None,
    camera: str,
    crop_index: int,
    grab_distance: float,
) -> None:
    """
    Add, move or remove a point of the crop being edited for one click,
    in full-image pixels.
    """

    if mode == ADD_POINT:
        points.append(clicked)
        return

    if mode == MOVE_POINT and picked is not None:
        points[picked] = clicked
        st.session_state["region_grab"] = None
        return

    index = nearest_point_index(points, *clicked, max_distance=grab_distance)

    if index is None:
        st.session_state["region_note"] = t(
            "No point near that click. Click on one of the numbered points."
        )
        return

    if mode == MOVE_POINT:
        st.session_state["region_grab"] = (camera, crop_index, index)
    else:
        points.pop(index)


def edit_points_table(
    camera: str,
    crop_index: int,
    points: list,
    image_size: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    The points of one crop as an editable table (type coordinates, add
    or delete rows). Returns the points as typed; rows missing x or y
    are left out until they are complete.
    """

    frame = pd.DataFrame(points, columns=["x", "y"])
    frame.insert(0, "#", range(1, len(points) + 1))

    width, height = image_size

    # The key changes with the crop and its points, so the editor starts
    # from the current list after every click, undo or reset instead of
    # applying its old edits on top of them (or on another crop's).
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        hide_index=True,
        disabled=["#"],
        key=f"region-table-{camera}-{crop_index}-{hash(tuple(points))}",
        column_config={
            "x": st.column_config.NumberColumn(
                min_value=0, max_value=width - 1, step=1, format="%d"
            ),
            "y": st.column_config.NumberColumn(
                min_value=0, max_value=height - 1, step=1, format="%d"
            ),
        },
    )

    rows = edited[["x", "y"]].dropna()

    return [
        (int(round(float(x))), int(round(float(y))))
        for x, y in rows.itertuples(index=False)
    ]


# Palette and fonts follow the FINGO login page: dark grey surfaces, lime
# green accent, Montserrat. Colours live in .streamlit/config.toml; fonts
