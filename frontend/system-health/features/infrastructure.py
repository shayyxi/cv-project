"""Infrastructure: host utilization meters and the storage footprint."""

from __future__ import annotations

from html import escape as esc

import streamlit as st

import health_data as hd
from features.format import gb
from features.i18n import register, t
from features.widgets import html_table, meter, section_head


FR = {
    'Infrastructure': 'Infrastructure',
    'Host utilization and storage footprint': "Utilisation de l'hôte et empreinte de stockage",
    'Host utilization': "Utilisation de l'hôte",
    'Sampled on the pipeline host': "Mesuré sur l'hôte du pipeline",
    'CPU': 'Processeur',
    'Memory': 'Mémoire',
    'Disk': 'Disque',
    'Amber from {w}% (warning), red from {c}% (critical); the marks on each bar show the thresholds.': 'Ambre à partir de {w} % (avertissement), rouge à partir de {c} % (critique) ; les repères sur chaque barre montrent les seuils.',
    'Disk is above the {w}% warning threshold.': 'Le disque dépasse le seuil d’avertissement de {w} %.',
    'Disk is above the {c}% critical threshold.': 'Le disque dépasse le seuil critique de {c} %.',
    'Storage footprint': 'Empreinte de stockage',
    'Resource': 'Ressource',
    'Health implication': "Incidence sur l'état",
    'Raw images': 'Images brutes',
    'Retention and storage growth': 'Rétention et croissance du stockage',
    'Processed images': 'Images traitées',
    'Annotated image retention': 'Rétention des images annotées',
    'PostgreSQL': 'PostgreSQL',
    'Database capacity': 'Capacité de la base de données',
    'Disk utilization': 'Utilisation du disque',
    'Within limits': 'Dans les limites',
    'Above warning threshold': 'Au-dessus du seuil d’avertissement',
    'Above critical threshold': 'Au-dessus du seuil critique',
}
register(FR)


def infrastructure_section(data: dict, th: dict) -> None:
    warn, crit = th["disk_warn"], th["disk_crit"]
    host = data.get("host") or {}
    storage = data.get("storage") or {}
    disk = host.get("disk_percent")
    section_head(t("Infrastructure"), t("Host utilization and storage footprint"))
    host_col, storage_col = st.columns(2, gap="medium")

    with host_col, st.container(key="card-host"):
        st.markdown(
            f"<div class='h3'>{esc(t('Host utilization'))}</div>"
            f"<div class='cap'>{esc(t('Sampled on the pipeline host'))}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            meter(t("CPU"), host.get("cpu_percent"), warn, crit)
            + meter(t("Memory"), host.get("memory_percent"), warn, crit)
            + meter(t("Disk"), disk, warn, crit),
            unsafe_allow_html=True,
        )
        note = t("Amber from {w}% (warning), red from {c}% (critical); the marks on each bar show the thresholds.", w=warn, c=crit)
        if disk is not None and disk >= crit:
            note = t("Disk is above the {c}% critical threshold.", c=crit) + " " + note
        elif disk is not None and disk >= warn:
            note = t("Disk is above the {w}% warning threshold.", w=warn) + " " + note
        st.caption(note)

    with storage_col, st.container(key="card-storage"):
        st.markdown(f"<div class='h3'>{esc(t('Storage footprint'))}</div>", unsafe_allow_html=True)
        level = hd.level(disk, warn, crit)
        pill = {"crit": ("crit", "Above critical threshold"), "warn": ("warn", "Above warning threshold")}.get(
            level, ("ok", "Within limits")
        )
        rows = [
            [esc(t("Raw images")), esc(gb(storage.get("raw_bytes"))), esc(t("Retention and storage growth"))],
            [esc(t("Processed images")), esc(gb(storage.get("processed_bytes"))), esc(t("Annotated image retention"))],
            [esc(t("PostgreSQL")), esc(gb(storage.get("database_bytes"))), esc(t("Database capacity"))],
            [
                esc(t("Disk utilization")), "–" if disk is None else f"{round(disk)}%",
                f"<span class='pill {pill[0]}'>{esc(t(pill[1]))}</span>",
            ],
        ]
        st.markdown(html_table([
            (t("Resource"), ""), (t("Size"), "num"), (t("Health implication"), "muted"),
        ], rows), unsafe_allow_html=True)
