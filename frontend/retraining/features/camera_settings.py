"""
The per-camera panels of the Crop regions page: the SAHI settings saved
to vision_config.yaml and the PPE classes switched off per camera.
"""

from __future__ import annotations

import streamlit as st

from app.retraining.history import SAHI_FIELDS, SAHI_FLAG, SAHI_LABELS, save_sahi_settings
from app.retraining.regions import CROP_CONFIG_PATH, PPE_CLASSES, load_disabled_ppe, save_disabled_ppe
from features.i18n import register, t


FR = {
    # Camera settings (vision_config.yaml, sahi.cameras)
    "The vision config could not be read: {error}":
        "La configuration vision n'a pas pu être lue : {error}",
    "Camera settings": "Réglages de la caméra",
    "New camera: filled with the default settings.":
        "Nouvelle caméra : remplie avec les réglages par défaut.",
    "Tile size (pixels)": "Taille des tuiles (pixels)",
    "Tile overlap, width": "Chevauchement, largeur",
    "Tile overlap, height": "Chevauchement, hauteur",
    "Tile overlap": "Chevauchement",
    "Merge threshold": "Seuil de fusion",
    "Second merge threshold": "Deuxième seuil de fusion",
    "Person confidence": "Confiance personne",
    "Second merge pass": "Deuxième passe de fusion",
    "Picture size is read from the picture on screen and is not entered here.":
        "La taille de l’image est lue sur l’image affichée et ne se saisit pas ici.",
    "Save camera settings": "Enregistrer les réglages de la caméra",
    "These settings are the saved ones; nothing to save.":
        "Ces réglages sont ceux enregistrés ; rien à enregistrer.",
    "Saved the settings of camera {camera} to {path} (sahi.cameras). The pipeline "
    "uses them from its next frame.":
        "Réglages de la caméra {camera} enregistrés dans {path} (sahi.cameras). Le "
        "pipeline les utilise dès sa prochaine image.",
    "PPE checked on camera {camera}. An unticked class counts as present "
    "there, so it is never a violation.":
        "EPI vérifiés sur la caméra {camera}. Une classe décochée est "
        "considérée présente, donc jamais une infraction.",
    "Save PPE checks": "Enregistrer les EPI vérifiés",
    "Saved the PPE checks for camera {camera} to {path}. The pipeline uses "
    "them from its next frame.":
        "EPI vérifiés enregistrés pour la caméra {camera} dans {path}. Le "
        "pipeline les utilise dès sa prochaine image.",
}
register(FR)


def ppe_checks_section(camera: str) -> None:
    """
    One checkbox per PPE class for this camera. An unticked class is
    saved under disabled_ppe in the crop config; the engine then counts
    it as present on that camera.
    """

    disabled = load_disabled_ppe().get(camera, set())

    st.caption(
        t(
            "PPE checked on camera {camera}. An unticked class counts as present "
            "there, so it is never a violation.",
            camera=camera,
        )
    )

    columns = st.columns(len(PPE_CLASSES) + 1)

    ticked = {
        name
        for column, name in zip(columns, PPE_CLASSES)
        if column.checkbox(
            t(name).capitalize(),
            value=name not in disabled,
            key=f"ppe_check_{camera}_{name}",
        )
    }

    unticked = set(PPE_CLASSES) - ticked

    with columns[-1]:
        if st.button(
            t("Save PPE checks"),
            disabled=unticked == disabled,
            key="ppe_checks_save",
            width="stretch",
        ):
            save_disabled_ppe(camera, unticked)
            st.session_state["region_flash"] = t(
                "Saved the PPE checks for camera {camera} to {path}. The pipeline "
                "uses them from its next frame.",
                camera=camera,
                path=CROP_CONFIG_PATH,
            )
            st.rerun()


def same_setting(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return a == b


def camera_settings_section(camera: str, defaults: dict, saved: dict | None) -> None:
    """
    The camera's SAHI settings from vision_config.yaml (sahi.cameras):
    tile size, overlaps, merge thresholds, person confidence and the
    second merge pass. A camera without an entry starts from the
    defaults; Save writes its entry.
    """

    flash = st.session_state.pop("settings_flash", None)

    with st.expander(t("Camera settings"), expanded=True):
        if flash:
            st.success(flash)

        if saved is None:
            st.caption(t("New camera: filled with the default settings."))

        current = {**defaults, **(saved or {})}
        values: dict = {}
        columns = st.columns(2)

        for index, (key, low, high, step) in enumerate(SAHI_FIELDS):
            value = min(max(current.get(key, low), low), high)

            with columns[index % 2]:
                if isinstance(step, int):
                    values[key] = int(st.number_input(
                        t(SAHI_LABELS[key]),
                        min_value=int(low),
                        max_value=int(high),
                        value=int(value),
                        step=step,
                        key=f"sahi_{camera}_{key}",
                    ))
                else:
                    values[key] = float(st.number_input(
                        t(SAHI_LABELS[key]),
                        min_value=float(low),
                        max_value=float(high),
                        value=float(value),
                        step=step,
                        format="%.2f",
                        key=f"sahi_{camera}_{key}",
                    ))

        values[SAHI_FLAG] = st.checkbox(
            t("Second merge pass"),
            value=bool(current.get(SAHI_FLAG, False)),
            key=f"sahi_{camera}_{SAHI_FLAG}",
        )

        st.caption(t("Picture size is read from the picture on screen and is not entered here."))

        unchanged = saved is not None and all(
            same_setting(values[key], saved.get(key, defaults.get(key))) for key in values
        )

        if st.button(
            t("Save camera settings"),
            type="primary",
            disabled=unchanged,
            key="sahi_save",
        ):
            try:
                path = save_sahi_settings(camera, values)
            except (OSError, ValueError) as error:
                st.error(str(error))
                return

            st.session_state["settings_flash"] = t(
                "Saved the settings of camera {camera} to {path} (sahi.cameras). The pipeline "
                "uses them from its next frame.",
                camera=camera,
                path=path,
            )
            st.rerun()

        if unchanged:
            st.caption(t("These settings are the saved ones; nothing to save."))
