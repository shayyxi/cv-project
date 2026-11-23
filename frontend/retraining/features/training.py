"""
The Retraining page: the training form, the last model comparison and
the replace button, the run history, the deployed model and its
previous versions to restore.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.retraining.config import RoboflowSettings
from app.retraining.history import deployed_weights, file_sha256, list_backups, list_runs
from app.retraining.train import (
    LATEST_COMPARISON_FILE,
    REGRESSION_TOLERANCE,
    install_weights,
    mark_installed,
    ppe_weights_path,
)
from features.i18n import register, t
from features.jobs import runner, running_notice, start_job, status_and_log


FR = {
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
    "Fine-tunes from the deployed best.pt, so every retraining builds on the "
    "model in production, and the dataset version holds every label so far, so "
    "earlier corrections are kept. Better is not guaranteed: both models are "
    "compared on the test split and nothing is replaced automatically. The "
    "numbers and the replace button appear at the bottom of the page.":
        "Affine à partir du best.pt déployé : chaque réentraînement part du "
        "modèle en production et la version du jeu de données contient toutes "
        "les étiquettes à ce jour. Un meilleur résultat n’est pas garanti : les "
        "deux modèles sont comparés sur le jeu de test et rien n’est remplacé "
        "automatiquement. Les chiffres et le bouton de remplacement "
        "apparaissent au bas de la page.",
    "'{job}' is running. Use Stop below to abort it.":
        "« {job} » est en cours. Utilisez Arrêter ci-dessous pour l'interrompre.",
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
    # Run history and deployed model
    "Run history": "Historique des exécutions",
    "The training folder could not be read: {error}":
        "Le dossier d'entraînement n'a pas pu être lu : {error}",
    "No training run on disk yet.": "Aucun entraînement sur le disque pour l'instant.",
    "Run": "Exécution",
    "Date": "Date",
    "mAP50-95 new": "mAP50-95 nouveau",
    "mAP50-95 deployed": "mAP50-95 déployé",
    "Delta": "Écart",
    "Outcome": "Résultat",
    "Weights": "Poids",
    "yes": "oui",
    "no": "non",
    "installed": "installé",
    "not worse, not installed": "pas moins bon, non installé",
    "scores lower": "score inférieur",
    "not compared": "non comparé",
    "Deployed model": "Modèle déployé",
    "No deployed weights at {path} yet.":
        "Aucun poids déployé dans {path} pour l'instant.",
    "**{path}** · {size} MB · installed {when} · sha256 {digest}…":
        "**{path}** · {size} Mo · installé le {when} · sha256 {digest}…",
    "No previous version yet: one is kept each time the replace button installs "
    "a model.":
        "Aucune version précédente pour l'instant : une est conservée chaque fois "
        "que le bouton de remplacement installe un modèle.",
    "Previous versions": "Versions précédentes",
    "Restore this version over the deployed model. The current file is kept as a "
    "new backup.":
        "Rétablir cette version à la place du modèle déployé. Le fichier actuel "
        "est conservé comme nouvelle sauvegarde.",
    "Restore previous weights": "Rétablir les poids précédents",
    "Restore failed: {error}": "Échec du rétablissement : {error}",
    "Restored {source} -> {target}.": "Rétabli {source} -> {target}.",
}
register(FR)


# The comparison tiles, headline metric first.
METRIC_ORDER = ("mAP50-95", "mAP50", "precision", "recall")
# deployed = grey, new = FINGO green.
CHART_COLOURS = ["#8a8a8a", "#8ac646"]


def load_comparison() -> dict | None:
    try:
        return json.loads(LATEST_COMPARISON_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


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


def run_outcome(comparison: dict | None) -> str:
    if not comparison:
        return t("not compared")
    if comparison.get("installed"):
        return t("installed")
    return t("not worse, not installed") if comparison.get("better") else t("scores lower")


def history_section() -> None:
    """Earlier training rounds on disk, newest first, with their comparison."""

    st.subheader(t("Run history"))

    try:
        runs = list_runs()
    except OSError as error:
        st.error(t("The training folder could not be read: {error}", error=error))
        return

    if not runs:
        st.caption(t("No training run on disk yet."))
        return

    rows = []

    for run in runs:
        new, deployed = run.map5095
        rows.append({
            t("Run"): run.name,
            t("Date"): run.started.strftime("%Y-%m-%d %H:%M"),
            t("Epochs"): run.epochs,
            t("mAP50-95 new"): new,
            t("mAP50-95 deployed"): deployed,
            t("Delta"): None if new is None or deployed is None else round(new - deployed, 4),
            t("Outcome"): run_outcome(run.comparison),
            t("Weights"): t("yes") if run.has_weights else t("no"),
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True)


@st.cache_data(show_spinner=False)
def sha256_prefix(path: str, mtime: float) -> str:
    """The deployed file's digest, computed once per file version."""

    return file_sha256(path)


def deployed_section() -> None:
    """The weights in use, and the backups the replace button kept, to go back to."""

    st.subheader(t("Deployed model"))

    flash = st.session_state.pop("restore_flash", None)

    if flash:
        st.success(flash)

    current = deployed_weights()

    if current is None:
        st.caption(t("No deployed weights at {path} yet.", path=ppe_weights_path()))
        return

    st.markdown(t(
        "**{path}** · {size} MB · installed {when} · sha256 {digest}…",
        path=current.path,
        size=f"{current.size_mb:.1f}",
        when=current.modified.strftime("%Y-%m-%d %H:%M"),
        digest=sha256_prefix(str(current.path), current.modified.timestamp()),
    ))

    backups = list_backups()

    if not backups:
        st.caption(t(
            "No previous version yet: one is kept each time the replace button installs "
            "a model."
        ))
        return

    half, _ = st.columns(2)

    with half:
        index = st.selectbox(
            t("Previous versions"),
            list(range(len(backups))),
            format_func=lambda i: f"{backups[i].modified:%Y-%m-%d %H:%M:%S} · {backups[i].size_mb:.1f} MB",
            key="restore_choice",
        )

    chosen = backups[index]
    confirmed = st.checkbox(
        t(
            "Restore this version over the deployed model. The current file is kept as a "
            "new backup."
        ),
        key="restore_confirm",
    )
    busy = runner().busy()

    if st.button(
        t("Restore previous weights"),
        disabled=not confirmed or busy is not None,
        key="restore_button",
    ):
        try:
            target, backup = install_weights(chosen.path)
        except Exception as error:
            st.error(t("Restore failed: {error}", error=error))
            return

        message = t("Restored {source} -> {target}.", source=chosen.path.name, target=target)

        if backup:
            message += t(" Previous weights kept at {backup}.", backup=backup)

        message += t(" The pipeline picks it up on its next frame, no restart needed.")
        st.session_state["restore_flash"] = message
        st.rerun()


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
                "Fine-tunes from the deployed best.pt, so every retraining builds on the "
                "model in production, and the dataset version holds every label so far, so "
                "earlier corrections are kept. Better is not guaranteed: both models are "
                "compared on the test split and nothing is replaced automatically. The "
                "numbers and the replace button appear at the bottom of the page."
            )
        )

    running_notice(busy)

    st.divider()

    status_and_log("train")

    st.divider()

    comparison_section()

    st.divider()

    history_section()

    st.divider()

    deployed_section()
