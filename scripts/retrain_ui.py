"""
Streamlit page for the retraining loop:

    1. Upload the label queue to Roboflow
    2. Train the PPE model, compare it, replace the deployed one by button
    3. Mark or adjust a camera's crop region by clicking on a frame

Start it with:
    streamlit run scripts/retrain_ui.py
or double-click retrain_ui.bat in the project folder.

Each button runs the matching CLI script (scripts.roboflow_upload,
scripts.train_ppe) in the background, shows its progress as a bar,
and can stop it. The last model comparison (deployed vs new, overall
and per class) is shown as tables. Run this on the machine that
should do the training.
"""

import json
import re
import sys
from pathlib import Path

# `streamlit run scripts/retrain_ui.py` puts scripts/ on sys.path, not
# the project root; add the root so `app.*` imports work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from app.config import settings  # noqa: E402
from app.retraining.config import (  # noqa: E402
    HOW_TO_CONFIGURE,
    LABEL_QUEUE_DIR,
    RoboflowSettings,
)
from app.retraining.regions import (  # noqa: E402
    CROP_CONFIG_PATH,
    DEFAULT_LABEL,
    LABEL_COLOURS,
    LABELS,
    PPE_CLASSES,
    Crop,
    crop_preview,
    display_base,
    draw_overlay,
    drawn_crops,
    load_disabled_ppe,
    load_regions,
    nearest_point_index,
    normalise_crops,
    save_disabled_ppe,
    save_region,
    to_full_image,
)

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # the section explains how to install it
    streamlit_image_coordinates = None
from app.retraining.jobs import JobBusy, JobRunner, JobState  # noqa: E402
from app.retraining.progress import parse_progress  # noqa: E402
from app.retraining.train import (  # noqa: E402
    LATEST_COMPARISON_FILE,
    REGRESSION_TOLERANCE,
    install_weights,
    mark_installed,
    ppe_weights_path,
)
from app.retraining.upload import RoboflowUploadService  # noqa: E402

LOG_LINES = 200

# The comparison tiles, headline metric first.
METRIC_ORDER = ("mAP50-95", "mAP50", "precision", "recall")
# deployed = grey, new = FINGO green.
CHART_COLOURS = ["#8a8a8a", "#8ac646"]

# The page in English or French. The English text is the key, so a string
# without an entry simply stays English; {placeholders} are filled by t().
LANGUAGES = {"en": "English", "fr": "Français"}
THEMES = ("dark", "light")

FR = {
    "PPE retraining": "Réentraînement EPI",
    "dark": "sombre",
    "light": "clair",
    # Sidebar: menu entries
    "Upload to Roboflow": "Téléverser vers Roboflow",
    "Retraining": "Réentraînement",
    "Crop regions": "Zones de recadrage",
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
    # Retraining page
    "Epochs": "Époques",
    "Device": "Appareil",
    "Where training runs. gpu: the first NVIDIA GPU (fast, needs one). cpu: "
    "the processor (slow, works anywhere).":
        "Où l'entraînement s'exécute. gpu : le premier GPU NVIDIA (rapide, il "
        "en faut un). cpu : le processeur (lent, fonctionne partout).",
    "Generate a new dataset version from the current labels":
        "Générer une nouvelle version du jeu de données à partir des "
        "étiquettes actuelles",
    "Train": "Entraîner",
    "Fine-tunes from the deployed best.pt and compares both on the test "
    "split. Nothing is replaced automatically: the numbers and the replace "
    "button appear at the bottom of the page.":
        "Affine à partir du best.pt déployé et compare les deux sur le jeu de "
        "test. Rien n'est remplacé automatiquement : les chiffres et le bouton "
        "de remplacement apparaissent au bas de la page.",
    "'{job}' is running. Use Stop below to abort it.":
        "« {job} » est en cours. Utilisez Arrêter ci-dessous pour l'interrompre.",
    # Job names, as the runner calls them.
    "upload": "téléversement",
    "train": "entraînement",
    # Status and log
    "Started: {command}": "Démarré : {command}",
    "running since {since}": "en cours depuis {since}",
    "done": "terminé",
    "failed (exit code {code})": "échec (code de sortie {code})",
    "stopped": "arrêté",
    "idle": "inactif",
    "Stopped at: {step}": "Arrêté à : {step}",
    "Failed at: {step}": "Échec à : {step}",
    "Current step: {step} ({percent}%)": "Étape en cours : {step} ({percent} %)",
    "Overall: {percent}%  |  epochs completed {done} of {total}":
        "Global : {percent} %  |  époques terminées {done} sur {total}",
    "Overall: {percent}%": "Global : {percent} %",
    "Stop": "Arrêter",
    "took {minutes} min {seconds} s": "a pris {minutes} min {seconds} s",
    "Show log": "Afficher le journal",
    "(no run yet)": "(aucune exécution pour l'instant)",
    # Comparison and replacement
    "Last model comparison": "Dernière comparaison de modèles",
    "No comparison yet. It appears after the first Train run with 'install "
    "if not worse' enabled.":
        "Aucune comparaison pour l'instant. Elle apparaît après le premier "
        "entraînement avec « installer si pas moins bon » activé.",
    "New model installed": "Nouveau modèle installé",
    "New model is at least as good, not installed yet":
        "Le nouveau modèle est au moins aussi bon, pas encore installé",
    "New model scores lower, not installed":
        "Le nouveau modèle obtient un score inférieur, non installé",
    "**{outcome}** on the {split} split. Run `{run}`, {timestamp}.":
        "**{outcome}** sur le jeu {split}. Exécution `{run}`, {timestamp}.",
    "mAP50-95 dropped by more than {tolerance} on: {classes}":
        "mAP50-95 a baissé de plus de {tolerance} sur : {classes}",
    "New model; the arrow is the change versus the deployed one.":
        "Nouveau modèle ; la flèche est l'écart par rapport au modèle déployé.",
    "Deployed: {value}": "Déployé : {value}",
    "**Overall**": "**Global**",
    "**mAP50-95 per class**": "**mAP50-95 par classe**",
    "deployed": "déployé",
    "new": "nouveau",
    "delta": "écart",
    "This model was installed on {when}. The pipeline uses it from the next "
    "frame.":
        "Ce modèle a été installé le {when}. Le pipeline l'utilise dès la "
        "prochaine image.",
    "This comparison has no weights path recorded; run the training again to "
    "get the replace button.":
        "Cette comparaison n'a pas de chemin de poids enregistré; relancez "
        "l'entraînement pour obtenir le bouton de remplacement.",
    "The new model file no longer exists: {path}":
        "Le fichier du nouveau modèle n'existe plus : {path}",
    "This model scores lower than the deployed one on mAP50-95. Replacing it "
    "would make the pipeline worse.":
        "Ce modèle obtient un mAP50-95 inférieur à celui du modèle déployé. Le "
        "remplacer dégraderait le pipeline.",
    "I understand it scores lower and want to replace the deployed model "
    "anyway":
        "Je comprends que son score est inférieur et je veux quand même "
        "remplacer le modèle déployé",
    "Replace deployed model with this one":
        "Remplacer le modèle déployé par celui-ci",
    "Install failed: {error}": "Échec de l'installation : {error}",
    "Installed -> {target}.": "Installé -> {target}.",
    " Previous weights kept at {backup}.":
        " Poids précédents conservés dans {backup}.",
    " The pipeline picks it up on its next frame, no restart needed.":
        " Le pipeline le prend en compte à sa prochaine image, sans redémarrage.",
    "Copies the new best.pt over {path} and keeps the previous file as a "
    "backup next to it. The running pipeline reloads it on the next frame.":
        "Copie le nouveau best.pt sur {path} et garde le fichier précédent en "
        "sauvegarde à côté. Le pipeline en cours le recharge à la prochaine "
        "image.",
    # Crop regions page
    "Marking by clicking needs the streamlit-image-coordinates package: pip "
    "install streamlit-image-coordinates":
        "Le marquage par clic nécessite le paquet streamlit-image-coordinates : "
        "pip install streamlit-image-coordinates",
    "Camera": "Caméra",
    "new camera id...": "nouvel identifiant de caméra...",
    "New camera id": "Nouvel identifiant de caméra",
    "Frame to mark (jpg or png)": "Image à marquer (jpg ou png)",
    "Use the latest raw frame of this camera instead of an upload":
        "Utiliser la dernière image brute de cette caméra au lieu d'un "
        "téléversement",
    "No raw frames under {folder}.": "Aucune image brute sous {folder}.",
    "Upload a frame, or tick the raw-frame option, to start marking.":
        "Téléversez une image, ou cochez l'option image brute, pour commencer "
        "le marquage.",
    "Type the new camera id first.":
        "Saisissez d'abord le nouvel identifiant de caméra.",
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
    "PPE checked on camera {camera}. An unticked class counts as present "
    "there, so it is never a violation.":
        "EPI vérifiés sur la caméra {camera}. Une classe décochée est "
        "considérée présente, donc jamais une infraction.",
    "Save PPE checks": "Enregistrer les EPI vérifiés",
    "Saved the PPE checks for camera {camera} to {path}. The pipeline uses "
    "them from its next frame.":
        "EPI vérifiés enregistrés pour la caméra {camera} dans {path}. Le "
        "pipeline les utilise dès sa prochaine image.",
    "Crop {number} has a single point: add another or remove the crop.":
        "La zone {number} n'a qu'un seul point : ajoutez-en un autre ou "
        "supprimez la zone.",
}

# The progress texts come from app.retraining.progress with numbers inside,
# so they are matched by pattern rather than looked up.
PROGRESS_FR = [
    (re.compile(r"^Starting$"), "Démarrage"),
    (re.compile(r"^Generating a dataset version in Roboflow$"),
     "Génération d'une version du jeu de données dans Roboflow"),
    (re.compile(r"^Downloading the dataset from Roboflow$"),
     "Téléchargement du jeu de données depuis Roboflow"),
    (re.compile(r"^Downloading the dataset (\d+)%$"),
     r"Téléchargement du jeu de données \1 %"),
    (re.compile(r"^Extracting the dataset (\d+)%$"),
     r"Extraction du jeu de données \1 %"),
    (re.compile(r"^Aligning class order to the deployed model$"),
     "Alignement de l'ordre des classes sur le modèle déployé"),
    (re.compile(r"^Preparing (\w+) images (\d+)/(\d+)$"),
     r"Préparation des images \1 \2/\3"),
    (re.compile(r"^Loading the model$"), "Chargement du modèle"),
    (re.compile(r"^Starting training \((\d+) epochs\)$"),
     r"Début de l'entraînement (\1 époques)"),
    (re.compile(r"^Epoch (\d+)/(\d+), batch (\d+)/(\d+)$"),
     r"Époque \1/\2, lot \3/\4"),
    (re.compile(r"^Validating epoch (\d+)/(\d+)$"),
     r"Validation de l'époque \1/\2"),
    (re.compile(r"^Comparing with the deployed model \((\d)/2\)$"),
     r"Comparaison avec le modèle déployé (\1/2)"),
    (re.compile(r"^Finished: new weights installed$"),
     "Terminé : nouveaux poids installés"),
    (re.compile(r"^Finished: new model not better, deployed weights kept$"),
     "Terminé : nouveau modèle pas meilleur, poids déployés conservés"),
    (re.compile(r"^Finished: training done, weights not installed$"),
     "Terminé : entraînement fait, poids non installés"),
    (re.compile(r"^Uploading crop (\d+) of (\d+)$"),
     r"Téléversement du recadrage \1 sur \2"),
    (re.compile(r"^Failed(: .*)?$"), r"Échec\1"),
]


def t(text: str, **values) -> str:
    """The page's text in the chosen language; the English is the key."""

    if st.session_state.get("lang") == "fr":
        text = FR.get(text, text)

    return text.format(**values) if values else text


def translate_progress(text: str) -> str:
    """A progress line from the log parser, in the chosen language."""

    if st.session_state.get("lang") != "fr":
        return text

    for pattern, french in PROGRESS_FR:
        if pattern.match(text):
            return pattern.sub(french, text)

    return text


def remember_choice(widget_key: str, store: str) -> None:
    """
    Keeps a segmented control's choice under its own key. Clicking the
    active segment again deselects it (the widget reports None): the
    previous choice is put back instead. The theme control's labels
    follow the language, which makes it a new widget to Streamlit each
    time the language changes; the stored choice survives the switch.
    """

    choice = st.session_state.get(widget_key)

    if choice is None:
        st.session_state[widget_key] = st.session_state[store]
        return

    st.session_state[store] = choice


@st.cache_resource
def runner() -> JobRunner:
    """One runner for the whole server, so jobs survive page reloads."""

    return JobRunner()


def queue_crops(config: RoboflowSettings) -> list[dict] | None:
    """
    Every crop in the label queue with its pre-labels and upload
    state, or None when the queue cannot be read.
    """

    try:
        return RoboflowUploadService(config=config).list_crops()
    except Exception:
        return None


def start_job(name: str, options: dict) -> None:
    try:
        state = runner().start(name, options)
    except JobBusy as error:
        st.error(str(error))
        return

    st.success(t("Started: {command}", command=" ".join(state.command)))
    st.rerun()


def badge(state: JobState) -> str:
    if state.state == "running":
        since = state.started_at.strftime("%H:%M:%S") if state.started_at else ""
        return f":orange[{t('running since {since}', since=since)}]"

    if state.state == "done":
        return f":green[{t('done')}]"

    if state.state == "failed":
        return f":red[{t('failed (exit code {code})', code=state.exit_code)}]"

    if state.state == "stopped":
        return f":orange[{t('stopped')}]"

    return f":grey[{t('idle')}]"


def load_comparison() -> dict | None:
    try:
        return json.loads(LATEST_COMPARISON_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _clamp(value: float | None) -> float:
    return min(max(value if value is not None else 0.0, 0.0), 1.0)


def show_progress(job: str, state: JobState, progress: dict) -> None:
    """
    Two bars: the current step with its own percentage (download,
    extraction, batches of this epoch, ...) and the overall run.
    """

    label = translate_progress(progress["text"])

    if state.state == "stopped":
        label = t("Stopped at: {step}", step=label)
    elif state.state == "failed" and not progress["text"].startswith("Failed"):
        label = t("Failed at: {step}", step=label)

    step = _clamp(progress["phase_fraction"])
    st.progress(
        step,
        text=t("Current step: {step} ({percent}%)", step=label, percent=f"{step * 100:.0f}"),
    )

    overall = _clamp(progress["fraction"])

    if job == "train" and progress["epochs"]:
        epochs_done = max(progress["epoch"] - 1, 0)

        if progress["phase"] in ("compare", "done"):
            epochs_done = progress["epochs"]

        st.progress(
            overall,
            text=t(
                "Overall: {percent}%  |  epochs completed {done} of {total}",
                percent=f"{overall * 100:.0f}",
                done=epochs_done,
                total=progress["epochs"],
            ),
        )
    else:
        st.progress(
            overall, text=t("Overall: {percent}%", percent=f"{overall * 100:.0f}")
        )


@st.fragment(run_every="2s")
def status_and_log(job: str) -> None:
    """
    One job's state, progress and log, under that job's page. Refreshes
    on its own while a job runs; full rerun when one ends.
    """

    jobs = runner()
    status = jobs.status()

    busy_now = any(state.running for state in status.values())
    was_busy = st.session_state.get("was_busy", False)
    st.session_state["was_busy"] = busy_now

    if was_busy and not busy_now:
        # A job just finished: re-enable the buttons above.
        st.rerun()

    state = status[job]

    st.markdown(f"**{t(job)}**: {badge(state)}")

    if state.running:
        if st.button(t("Stop"), key=f"stop-{job}"):
            jobs.stop(job)
            st.rerun()

    elif state.finished_at and state.started_at:
        seconds = int((state.finished_at - state.started_at).total_seconds())
        st.caption(
            t("took {minutes} min {seconds} s", minutes=seconds // 60, seconds=seconds % 60)
        )

    log_text = jobs.log_text(job)

    if state.state != "idle":
        show_progress(job, state, parse_progress(job, log_text))

    # The bar is the view; the raw log is there when something needs
    # a closer look.
    with st.expander(t("Show log"), expanded=False):
        tail = "\n".join(log_text.splitlines()[-LOG_LINES:])
        st.code(tail or t("(no run yet)"), language="text")


def comparison_section() -> None:
    st.subheader(t("Last model comparison"))

    record = load_comparison()

    if not record:
        st.caption(
            t(
                "No comparison yet. It appears after the first Train run "
                "with 'install if not worse' enabled."
            )
        )
        return

    if record.get("installed"):
        outcome = t("New model installed")
    elif record.get("better"):
        outcome = t("New model is at least as good, not installed yet")
    else:
        outcome = t("New model scores lower, not installed")

    st.markdown(
        t(
            "**{outcome}** on the {split} split. Run `{run}`, {timestamp}.",
            outcome=outcome,
            split=record.get("split"),
            run=record.get("run"),
            timestamp=record.get("timestamp"),
        )
    )

    regressed = record.get("regressed_classes") or []

    if regressed:
        st.warning(
            t(
                "mAP50-95 dropped by more than {tolerance} on: {classes}",
                tolerance=REGRESSION_TOLERANCE,
                classes=", ".join(regressed),
            )
        )

    overall = pd.DataFrame(
        {"deployed": record["deployed"], "new": record["new"]}
    )
    overall["delta"] = overall["new"] - overall["deployed"]

    # The headline first: one tile per metric with the new value and the
    # change against the deployed model as a coloured arrow.
    order = [m for m in METRIC_ORDER if m in overall.index]
    order += [m for m in overall.index if m not in order]

    st.caption(t("New model; the arrow is the change versus the deployed one."))

    for column, metric in zip(st.columns(len(order)), order):
        column.metric(
            metric,
            f"{overall.loc[metric, 'new']:.4f}",
            delta=f"{overall.loc[metric, 'delta']:+.4f}",
            help=t("Deployed: {value}", value=f"{overall.loc[metric, 'deployed']:.4f}"),
        )

    per_class = pd.DataFrame(record["per_class"]).T

    if not per_class.empty:
        per_class = per_class[["deployed", "new", "delta"]]

    # Column names in the chosen language, for display only.
    columns = {name: t(name) for name in ("deployed", "new", "delta")}

    left, right = st.columns(2)

    with left:
        st.markdown(t("**Overall**"))
        st.dataframe(overall.round(4).rename(columns=columns))

    with right:
        st.markdown(t("**mAP50-95 per class**"))
        st.dataframe(per_class.round(4).rename(columns=columns))

    if not per_class.empty:
        chart = per_class[["deployed", "new"]].rename(columns=columns)

        try:
            st.bar_chart(chart, stack=False, color=CHART_COLOURS)
        except TypeError:
            st.bar_chart(chart)

    replace_section(record)


def replace_section(record: dict) -> None:
    """The decision: replace the deployed model with the compared one."""

    flash = st.session_state.pop("install_flash", None)

    if flash:
        st.success(flash)

    if record.get("installed"):
        st.info(
            t(
                "This model was installed on {when}. The pipeline uses it "
                "from the next frame.",
                when=record.get("installed_at") or record.get("timestamp"),
            )
        )
        return

    new_weights = record.get("new_weights")

    if not new_weights:
        st.caption(
            t(
                "This comparison has no weights path recorded; run the "
                "training again to get the replace button."
            )
        )
        return

    if not Path(new_weights).is_file():
        st.error(t("The new model file no longer exists: {path}", path=new_weights))
        return

    confirmed = True

    if not record.get("better"):
        st.error(
            t(
                "This model scores lower than the deployed one on mAP50-95. "
                "Replacing it would make the pipeline worse."
            )
        )
        confirmed = st.checkbox(
            t(
                "I understand it scores lower and want to replace the "
                "deployed model anyway"
            ),
            key="confirm_worse",
        )

    busy = runner().busy()

    if st.button(
        t("Replace deployed model with this one"),
        type="primary",
        disabled=not confirmed or busy is not None,
    ):
        try:
            target, backup = install_weights(Path(new_weights))
        except Exception as error:
            st.error(t("Install failed: {error}", error=error))
            return

        mark_installed()

        message = t("Installed -> {target}.", target=target)

        if backup:
            message += t(" Previous weights kept at {backup}.", backup=backup)

        message += t(
            " The pipeline picks it up on its next frame, no restart needed."
        )

        st.session_state["install_flash"] = message
        st.rerun()

    st.caption(
        t(
            "Copies the new best.pt over {path} and keeps the previous file "
            "as a backup next to it. The running pipeline reloads it on the "
            "next frame.",
            path=ppe_weights_path(),
        )
    )


NEW_CAMERA = "new camera id..."

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

    left, right = st.columns(2)

    with left:
        # The values stay English (NEW_CAMERA is compared below); only
        # the shown text follows the language.
        choice = st.selectbox(
            t("Camera"),
            sorted(regions) + [NEW_CAMERA],
            format_func=t,
            key="region_camera",
        )

    with right:
        typed = st.text_input(
            t("New camera id"),
            key="region_new_id",
            disabled=choice != NEW_CAMERA,
        )

    camera = typed.strip() if choice == NEW_CAMERA else choice

    if camera:
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

    if not camera:
        st.warning(t("Type the new camera id first."))
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
            apply_click(mode, clicked, points, picked, camera, active, GRAB_RADIUS * scale)
            st.rerun()

    with panel:
        undo, clear = st.columns(2)

        if undo.button(t("Remove last point"), disabled=not points, width="stretch"):
            points.pop()
            st.session_state["region_grab"] = None
            st.rerun()

        if clear.button(t("Clear points"), disabled=not points, width="stretch"):
            points.clear()
            st.session_state["region_grab"] = None
            st.rerun()

        if st.button(
            t("Reset to saved crops"),
            disabled=not saved or crops == saved,
            width="stretch",
        ):
            marks[camera] = clone_crops(saved)
            actives[camera] = 0
            st.session_state["region_grab"] = None
            st.rerun()

        if points:
            typed_points = edit_points_table(camera, active, points, frame.size)

            if typed_points != points:
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
# and the finer details are set here.
THEME_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap");

html, body, [class*="st-"], button, input, textarea, select {
    font-family: "Montserrat", system-ui, -apple-system, "Segoe UI", sans-serif;
}
/* Streamlit draws its arrows and other icons with an icon font; keep it. */
[data-testid="stIconMaterial"], [class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
}
h1, h2, h3 {
    font-family: "Montserrat", system-ui, -apple-system, "Segoe UI", sans-serif !important;
}
h1 { font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }
h2, h3 { font-weight: 600; }
code, pre {
    font-family: ui-monospace, "Cascadia Mono", Consolas, monospace !important;
}
[data-testid="stWidgetLabel"] p { font-weight: 600; }
[data-testid="stHeader"] { background: #3b3b3b; }
[data-testid="stCaptionContainer"] { color: #b3b3b3; }
[data-testid="stExpander"], [data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 4px;
}
/* Buttons: outlined in the accent green with uppercase labels, as on the
   login page; the primary button is filled instead. */
.stButton > button, .stDownloadButton > button {
    border-radius: 4px;
    border: 1px solid #8ac646;
    background: transparent;
    color: #ffffff;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #8ac646;
    background: rgba(138, 198, 70, 0.14);
    color: #ffffff;
}
.stButton > button[kind="primary"] { background: #8ac646; color: #1e1e1e; }
.stButton > button[kind="primary"] p { color: #1e1e1e; }
.stButton > button[kind="primary"]:hover { background: #7eb145; border-color: #7eb145; }
.stButton > button:disabled, .stDownloadButton > button:disabled { opacity: 0.45; }
hr { border-color: rgba(255, 255, 255, 0.14); }
/* Hide the hover link icon next to headings. */
[data-testid="stHeaderActionElements"] { display: none; }
/* Segmented controls (DARK | LIGHT, EN | FR): a bordered pill, the
   chosen segment filled in the accent green with dark text. */
[data-testid="stButtonGroup"] {
    gap: 2px;
    padding: 2px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 6px;
    background: #2a2a2a;
}
[data-testid="stButtonGroup"] button[kind="segmented_control"],
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] {
    min-height: 0;
    border: 0 !important;
    border-radius: 4px !important;
    background: transparent;
    padding: 2px 10px;
    color: #ffffff;
}
/* The label is a markdown paragraph with its own 14 px type. */
[data-testid="stButtonGroup"] button p {
    margin: 0;
    font-size: 11px;
    line-height: 1.5;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="stButtonGroup"] button[kind="segmented_control"]:hover {
    background: rgba(138, 198, 70, 0.14);
}
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"]:hover {
    background: #8ac646;
    color: #1e1e1e;
}
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] p,
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] span { color: #1e1e1e; }
/* The sidebar is the menu: darker panel (colours in config.toml), the
   open page highlighted in the accent green. */
[data-testid="stSidebar"] { border-right: 1px solid rgba(255, 255, 255, 0.10); }
/* Collapsed, Streamlit slides the sidebar off-screen and squeezes it to
   nothing. Keep it instead as a narrow rail with the menu icons: text,
   language and dark-mode controls hidden, icons centred. */
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important;
    min-width: 64px !important;
    max-width: 64px !important;
    width: 64px !important;
    overflow: hidden;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] {
    width: 64px !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    /* Streamlit reserves a 10 px scrollbar gutter on both edges. */
    scrollbar-gutter: auto !important;
    overflow: hidden !important;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"],
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavItems"],
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLinkContainer"] {
    width: 64px !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLink"] {
    justify-content: center;
    width: 40px !important;
    height: 40px !important;
    padding: 0 !important;
    margin: 4px auto !important;
    border-radius: 6px;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {
    font-size: 24px !important;
    width: 24px !important;
    height: 24px !important;
}
/* The link is [icon wrapper span][text span]: hide the one without an icon. */
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavLink"] > span:not(:has([data-testid="stIconMaterial"])) {
    display: none;
}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarUserContent"],
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavSeparator"],
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNavViewButton"] {
    display: none;
}
[data-testid="stSidebarNavLink"] { border-radius: 4px; }
[data-testid="stSidebarNavLink"] span { font-weight: 600; }
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: rgba(138, 198, 70, 0.18);
}
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: #8ac646; }
</style>
"""


# Light palette of the same dashboard, laid over the dark theme.
LIGHT_CSS = """
<style>
.stApp, [data-testid="stHeader"] { background: #eceef2; }
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3,
.stApp span:not([data-testid="stIconMaterial"]), .stApp summary { color: #141920; }
[data-testid="stCaptionContainer"] { color: #7c8493; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div,
input:not([type="checkbox"]), textarea, [data-testid="stFileUploaderDropzone"] {
    background: #fbfbfc !important; color: #141920 !important;
}
/* Checkboxes: grey when off, accent green when on, white tick. */
[data-baseweb="checkbox"] > div:first-child { background: #c4c9d2 !important; }
[data-baseweb="checkbox"]:has(input:checked) > div:first-child {
    background: #7eb145 !important;
}
[data-baseweb="checkbox"] > div:first-child > div { background: #ffffff !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li { background: #fbfbfc; color: #141920; }
[data-testid="stCode"] pre, [data-testid="stCode"] code { background: #f2f3f6 !important; color: #141920; }
[data-testid="stExpander"], [data-testid="stDataFrame"], hr {
    border-color: rgba(20, 25, 32, 0.11);
}
.stButton > button[kind="primary"] { color: #141920; }
.stButton > button[kind="primary"] span, .stButton > button[kind="primary"] p { color: #141920; }
[data-testid="stSidebar"] {
    background: #ffffff; border-right: 1px solid rgba(20, 25, 32, 0.11);
}
[data-testid="stButtonGroup"] { background: #fbfbfc; border-color: rgba(20, 25, 32, 0.2); }
[data-testid="stButtonGroup"] button[kind="segmented_control"] { color: #141920; }
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"]:hover {
    background: #8ac646; color: #141920;
}
/* Icons are font glyphs and white in the base theme: give them the text
   colour too, except inside the alert boxes, which colour their own. */
.stApp [data-testid="stIconMaterial"] { color: #141920; }
[data-testid="stAlertContainer"] [data-testid="stIconMaterial"] { color: inherit; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: #5f8f2f; }
</style>
"""


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
# compared in code); only the shown text follows the language.
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


def training_page() -> None:
    st.title(t("Retraining"))

    config = RoboflowSettings.from_env()
    busy = runner().busy()

    form, _ = st.columns(2)

    with form:
        epochs = st.number_input(t("Epochs"), min_value=1, max_value=1000, value=50)
        device_choice = st.selectbox(
            t("Device"),
            ["gpu", "cpu"],
            index=0,
            help=t(
                "Where training runs. gpu: the first NVIDIA GPU (fast, needs "
                "one). cpu: the processor (slow, works anywhere)."
            ),
        )
        # The training script calls the first GPU "0".
        device = "0" if device_choice == "gpu" else "cpu"
        generate = st.checkbox(
            t("Generate a new dataset version from the current labels"),
            value=True,
        )

        if st.button(
            t("Train"),
            type="primary",
            disabled=busy is not None or not config.configured,
        ):
            start_job(
                "train",
                {
                    "epochs": int(epochs),
                    "device": device,
                    "generate": generate,
                },
            )

        st.caption(
            t(
                "Fine-tunes from the deployed best.pt and compares both on "
                "the test split. Nothing is replaced automatically: the "
                "numbers and the replace button appear at the bottom of the "
                "page."
            )
        )

    running_notice(busy)

    st.divider()

    status_and_log("train")

    st.divider()

    comparison_section()


def running_notice(busy: str | None) -> None:
    if busy is not None:
        st.info(t("'{job}' is running. Use Stop below to abort it.", job=t(busy)))


def sidebar_controls(lang: str, dark: bool) -> None:
    """
    Under the menu: DARK | LIGHT and EN | FR as small segmented
    controls in one row (they wrap when the sidebar is too narrow).
    """

    with st.container(horizontal=True, gap="small"):
        st.segmented_control(
            "Theme",
            list(THEMES),
            default="dark" if dark else "light",
            format_func=t,
            key=f"theme_{lang}",
            on_change=remember_choice,
            args=(f"theme_{lang}", "theme"),
            label_visibility="collapsed",
            width="content",
        )
        # Codes shown as they are, so this widget keeps its identity
        # (and value) across languages.
        st.segmented_control(
            "Language",
            list(LANGUAGES),
            default=lang,
            format_func=str.upper,
            key="lang_control",
            on_change=remember_choice,
            args=("lang_control", "lang"),
            label_visibility="collapsed",
            width="content",
        )


def main() -> None:
    # Both choices are read before anything is drawn, so the first paint
    # already has the right language and palette.
    lang = st.session_state.get("lang", "en")
    dark = st.session_state.get("theme", "dark") != "light"

    st.set_page_config(
        page_title=t("PPE retraining"),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    if not dark:
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)

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
        sidebar_controls(lang, dark)

    page.run()


main()
