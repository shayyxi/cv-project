"""
What the Upload and Train buttons share: the background job runner,
starting a job, its status badge, progress bars and the live log, and
the French progress lines.
"""

from __future__ import annotations

import re

import streamlit as st

from app.retraining.jobs import JobBusy, JobRunner, JobState
from app.retraining.progress import parse_progress
from features.i18n import register, t


FR = {
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
}
register(FR)


LOG_LINES = 200


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


def translate_progress(text: str) -> str:
    """A progress line from the log parser, in the chosen language."""

    if st.session_state.get("lang") != "fr":
        return text

    for pattern, french in PROGRESS_FR:
        if pattern.match(text):
            return pattern.sub(french, text)

    return text


@st.cache_resource
def runner() -> JobRunner:
    """One runner for the whole server, so jobs survive page reloads."""

    return JobRunner()


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


def running_notice(busy: str | None) -> None:
    if busy is not None:
        st.info(t("'{job}' is running. Use Stop below to abort it.", job=t(busy)))
