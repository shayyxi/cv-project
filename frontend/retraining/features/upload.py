"""The Upload to Roboflow page: the queue count, the upload button and the crop gallery."""

from __future__ import annotations

import re

import streamlit as st

from app.retraining.config import HOW_TO_CONFIGURE, LABEL_QUEUE_DIR, RoboflowSettings
from app.retraining.upload import RoboflowUploadService
from features.i18n import register, t
from features.jobs import runner, running_notice, start_job, status_and_log


FR = {
    # Upload page
    "Crops waiting for upload: ": "Recadrages en attente de téléversement : ",
    "unknown": "inconnu",
    "Roboflow is not configured; {how}.": "Roboflow n'est pas configuré; {how}.",
    "Dry run (only count, send nothing)":
        "Essai à blanc (compter seulement, ne rien envoyer)",
    "Upload": "Téléverser",
    "Sends each crop once with its pre-labels. Review the batch in Roboflow "
    "and add it to the dataset before training.":
        "Envoie chaque recadrage une fois avec ses pré-étiquettes. Vérifiez le "
        "lot dans Roboflow et ajoutez-le au jeu de données avant l'entraînement.",
    # Crop gallery
    "Crops in the queue": "Recadrages dans la file",
    "The label queue could not be read.":
        "La file d'étiquetage n'a pas pu être lue.",
    "No crops in {folder} yet. The pipeline's daily job fills it.":
        "Aucun recadrage dans {folder} pour l'instant. La tâche quotidienne du "
        "pipeline la remplit.",
    "Show": "Afficher",
    "pending": "en attente",
    "uploaded": "téléversés",
    "all": "tous",
    "all cameras": "toutes les caméras",
    "{count} crops match, page {page} of {pages}.":
        "{count} recadrages correspondent, page {page} sur {pages}.",
    "person {number}": "personne {number}",
    "helmet": "casque",
    "vest": "gilet",
    "boots": "bottes",
    "no boxes": "aucune boîte",
    "uploaded {date}": "téléversé le {date}",
    "duplicate in Roboflow": "doublon dans Roboflow",
}
register(FR)


def queue_crops(config: RoboflowSettings) -> list[dict] | None:
    """
    Every crop in the label queue with its pre-labels and upload
    state, or None when the queue cannot be read.
    """

    try:
        return RoboflowUploadService(config=config).list_crops()
    except Exception:
        return None


def upload_page() -> None:
    st.title(t("Upload to Roboflow"))

    config = RoboflowSettings.from_env()
    busy = runner().busy()

    form, _ = st.columns(2)

    with form:
        crops = queue_crops(config)
        pending = (
            sum(1 for crop in crops if crop["uploaded_at"] is None)
            if crops is not None
            else None
        )
        st.write(
            t("Crops waiting for upload: ")
            + (f"**{pending}**" if pending is not None else t("unknown"))
        )

        if not config.configured:
            st.warning(t("Roboflow is not configured; {how}.", how=HOW_TO_CONFIGURE))

        dry_run = st.checkbox(t("Dry run (only count, send nothing)"), value=False)

        if st.button(
            t("Upload"),
            type="primary",
            disabled=busy is not None or (not config.configured and not dry_run),
        ):
            start_job("upload", {"dry_run": dry_run})

        st.caption(
            t(
                "Sends each crop once with its pre-labels. Review the batch "
                "in Roboflow and add it to the dataset before training."
            )
        )

    running_notice(busy)

    st.divider()

    status_and_log("upload")

    st.divider()

    crop_gallery(crops)


# The gallery of queued crops. The filter values stay English (they are


PENDING = "pending"
UPLOADED = "uploaded"
EVERY_CROP = "all"
EVERY_CAMERA = "all cameras"
GALLERY_COLUMNS = 6
GALLERY_PAGE = 24

# After the camera id: "<frame timestamp>_p<person>_<crop timestamp>".
CROP_NAME = re.compile(r"^(?P<frame>[^_]+)_p(?P<person>[^_]+)_")


def crop_gallery(crops: list[dict] | None) -> None:
    """Thumbnails of the queue's crops, by state and camera, a page at a time."""

    st.subheader(t("Crops in the queue"))

    if crops is None:
        st.caption(t("The label queue could not be read."))
        return

    if not crops:
        st.caption(
            t(
                "No crops in {folder} yet. The pipeline's daily job fills it.",
                folder=LABEL_QUEUE_DIR,
            )
        )
        return

    state_column, camera_column, page_column = st.columns([2, 2, 1])

    state = state_column.radio(
        t("Show"),
        [PENDING, UPLOADED, EVERY_CROP],
        format_func=t,
        horizontal=True,
        key="gallery_state",
    )
    cameras = sorted({crop["camera_id"] for crop in crops if crop["camera_id"]})
    camera = camera_column.selectbox(
        t("Camera"), [EVERY_CAMERA] + cameras, format_func=t, key="gallery_camera"
    )

    shown = [
        crop
        for crop in crops
        if (state == EVERY_CROP or (state == PENDING) == (crop["uploaded_at"] is None))
        and (camera == EVERY_CAMERA or crop["camera_id"] == camera)
    ]

    pages = max(1, -(-len(shown) // GALLERY_PAGE))
    # A changed filter is a new page control, so it starts again at 1.
    page = page_column.number_input(
        t("Page"),
        min_value=1,
        max_value=pages,
        value=1,
        key=f"gallery_page_{state}_{camera}",
    )

    st.caption(
        t(
            "{count} crops match, page {page} of {pages}.",
            count=len(shown),
            page=page,
            pages=pages,
        )
    )

    start = (page - 1) * GALLERY_PAGE

    for row_start in range(start, min(start + GALLERY_PAGE, len(shown)), GALLERY_COLUMNS):
        row = shown[row_start:row_start + GALLERY_COLUMNS]

        for column, crop in zip(st.columns(GALLERY_COLUMNS), row):
            with column:
                st.image(str(crop["path"]), use_container_width=True)
                st.caption(crop_caption(crop))


def crop_caption(crop: dict) -> str:
    """Three short lines: camera, frame time and person; pre-labels; state."""

    rest = crop["name"]

    if crop["camera_id"]:
        rest = rest[len(crop["camera_id"]) + 1:]

    match = CROP_NAME.match(rest)

    if match:
        frame = match.group("frame").replace("T", " ").rstrip("Z")
        date, _, clock = frame.partition(" ")
        where = f"{date} {clock.replace('-', ':')}".strip()
        where += " · " + t("person {number}", number=match.group("person"))
    else:
        where = rest.rsplit(".", 1)[0]

    labels = ", ".join(t(label) for label in crop["labels"]) or t("no boxes")

    if crop["duplicate"]:
        state = t("duplicate in Roboflow")
    elif crop["uploaded_at"]:
        state = t("uploaded {date}", date=crop["uploaded_at"][:10])
    else:
        state = t("pending")

    return f"{crop['camera_id'] or '?'} · {where}  \n{labels}  \n{state}"
