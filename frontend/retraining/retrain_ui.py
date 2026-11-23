"""
Streamlit page for the retraining loop:

    1. Upload the label queue to Roboflow
    2. Train the PPE model, compare it, replace the deployed one by button
    3. Mark or adjust a camera's crop region by clicking on a frame

Start it with:
    streamlit run frontend/retraining/retrain_ui.py
or double-click retrain_ui.bat in the project folder.

This file only wires the menu; each page lives in features/ (upload,
training, crops, camera_settings) with the job helpers in features/jobs
and the French strings next to the code that uses them. Each button
runs the matching CLI script (scripts.roboflow_upload, scripts.train_ppe)
in the background, shows its progress as a bar, and can stop it. Run
this on the machine that should do the training.
"""

import sys
from pathlib import Path

# `streamlit run frontend/retraining/retrain_ui.py` puts frontend/retraining/ on sys.path, not
# the project root; add the root so `app.*` imports work, and frontend/ for _shared. The
# page's own folder goes first so `features` is always this one.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(1, str(HERE.parent))
sys.path.insert(0, str(HERE))

import streamlit as st  # noqa: E402

from _shared import PHONE_CSS, init_language  # noqa: E402
from features.crops import region_section  # noqa: E402
from features.i18n import register, t  # noqa: E402
from features.training import training_page  # noqa: E402
from features.upload import upload_page  # noqa: E402
from theme import THEME_CSS  # noqa: E402


FR = {
    "PPE retraining": "Réentraînement EPI",
    "The language follows the browser (English or French). Add ?lang=fr or "
    "?lang=en to the address to force one.":
        "La langue suit celle du navigateur (anglais ou français). Ajoutez "
        "?lang=fr ou ?lang=en à l’adresse pour en imposer une.",
    # Sidebar: menu entries
    "Upload to Roboflow": "Téléverser vers Roboflow",
    "Retraining": "Réentraînement",
    "Crop regions": "Zones de recadrage",
}
register(FR)


def main() -> None:
    # The language is read before anything is drawn, so the first paint
    # is already right. It follows the browser (?lang=fr / ?lang=en
    # forces one); the page is dark only. No switches (reviewer
    # decision, October 2026).
    init_language()

    st.set_page_config(
        page_title=t("PPE retraining"),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(PHONE_CSS, unsafe_allow_html=True)

    # The sidebar is the dashboard's menu, one entry per page. The url
    # paths never change, so the open page survives a language switch.
    page = st.navigation(
        [
            st.Page(
                upload_page,
                title=t("Upload to Roboflow"),
                icon=":material/cloud_upload:",
                url_path="upload",
                default=True,
            ),
            st.Page(
                training_page,
                title=t("Retraining"),
                icon=":material/model_training:",
                url_path="retraining",
            ),
            st.Page(
                region_section,
                title=t("Crop regions"),
                icon=":material/crop:",
                url_path="crops",
            ),
        ]
    )

    with st.sidebar:
        st.caption(t(
            "The language follows the browser (English or French). Add ?lang=fr or "
            "?lang=en to the address to force one."
        ))

    page.run()


main()
