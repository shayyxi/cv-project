"""Pipeline and delivery: processing outcomes, WordPress delivery and API diagnostics."""

from __future__ import annotations

from html import escape as esc

import streamlit as st

from features.format import int_, num, pct, sec, span
from features.i18n import register, t
from features.widgets import html_table, section_head


FR = {
    'Pipeline & delivery': 'Pipeline et livraison',
    'Processing outcomes, WordPress delivery and API diagnostics': "Résultats du traitement, livraison WordPress et diagnostics de l'API",
    'Processing outcomes': 'Résultats du traitement',
    'Metric': 'Mesure',
    'Value': 'Valeur',
    'Interpretation': 'Interprétation',
    'Completed jobs': 'Tâches terminées',
    'Jobs successfully processed': 'Tâches traitées avec succès',
    'Failed jobs': 'Tâches échouées',
    'Processing errors': 'Erreurs de traitement',
    'Oldest pending job: {age}': 'Tâche en attente la plus ancienne : {age}',
    'Success rate': 'Taux de réussite',
    'Completed / (completed + failed)': 'Terminées / (terminées + échouées)',
    'Average workers per image': 'Travailleurs en moyenne par image',
    'Person detections per processed image': 'Détections de personnes par image traitée',
    'WordPress API delivery': "Livraison via l'API WordPress",
    'Outcome': 'Résultat',
    'Count': 'Nombre',
    'Meaning': 'Signification',
    'Successfully delivered to WordPress': 'Livrées avec succès à WordPress',
    'Failed': 'Échouées',
    'Eligible deliveries that failed': 'Livraisons admissibles ayant échoué',
    'Skipped, nobody in the frame': 'Ignorées, personne dans l’image',
    'No person detections; intentionally not sent': 'Aucune détection de personne; volontairement non envoyées',
    'Delivery success': 'Réussite des livraisons',
    'Delivered / eligible attempts': 'Livrées / tentatives admissibles',
    'API diagnostics': "Diagnostics de l'API",
    'Signal': 'Signal',
    'What to investigate': 'Quoi vérifier',
    'P95 response latency': 'Latence de réponse P95',
    'WordPress response performance': 'Performance de réponse de WordPress',
    'Multipart metadata or validation issues': 'Problèmes de métadonnées multipart ou de validation',
    'Rate limiting / backoff': 'Limitation de débit / attente',
    'WordPress availability': 'Disponibilité de WordPress',
    'Other HTTP 4xx': 'Autres HTTP 4xx',
    'Authentication or endpoint configuration': 'Authentification ou configuration du point d’accès',
    'Skipped images (nobody in the frame) are excluded from the WordPress delivery success denominator.': 'Les images ignorées (personne dans l’image) sont exclues du dénominateur de réussite des livraisons WordPress.',
}
register(FR)


def pipeline_section(data: dict) -> None:
    p = data.get("processing") or {}
    d = data.get("delivery") or {}
    http = d.get("http") or {}
    section_head(t("Pipeline & delivery"), t("Processing outcomes, WordPress delivery and API diagnostics"))
    cols = st.columns(3, gap="medium")

    def table3(col, title: str, head: tuple[str, str, str], rows: list[list[str]], raw_last: bool = False) -> None:
        with col, st.container(key=f"card-{head[0]}"):
            st.markdown(f"<div class='h3'>{esc(title)}</div>", unsafe_allow_html=True)
            cells = [[esc(a), esc(b), c if raw_last else esc(c)] for a, b, c in rows]
            st.markdown(html_table([(head[0], ""), (head[1], "num"), (head[2], "muted")], cells), unsafe_allow_html=True)

    pending_note = t("Oldest pending job: {age}", age=span(p.get("oldest_pending_seconds")) if p.get("pending") else t("none"))
    table3(cols[0], t("Processing outcomes"), (t("Metric"), t("Value"), t("Interpretation")), [
        [t("Completed jobs"), int_(p.get("completed")), t("Jobs successfully processed")],
        [t("Failed jobs"), int_(p.get("failed")), t("Processing errors")],
        [t("Pending jobs"), int_(p.get("pending")), pending_note],
        [t("Success rate"), pct(p.get("success_rate")), t("Completed / (completed + failed)")],
        [
            t("Average workers per image"),
            "–" if p.get("workers_per_image") is None else num(p["workers_per_image"], 1),
            t("Person detections per processed image"),
        ],
    ])
    table3(cols[1], t("WordPress API delivery"), (t("Outcome"), t("Count"), t("Meaning")), [
        [t("Delivered"), int_(d.get("delivered")), t("Successfully delivered to WordPress")],
        [t("Failed"), int_(d.get("failed")), t("Eligible deliveries that failed")],
        [t("Skipped, nobody in the frame"), int_(d.get("skipped")), t("No person detections; intentionally not sent")],
        [t("Delivery success"), pct(d.get("success_rate")), t("Delivered / eligible attempts")],
    ])
    table3(cols[2], t("API diagnostics"), (t("Signal"), t("Value"), t("What to investigate")), [
        [t("P95 response latency"), sec(d.get("p95_latency_ms")), t("WordPress response performance")],
        ["HTTP 422", int_(http.get("422")), t("Multipart metadata or validation issues")],
        ["HTTP 429", int_(http.get("429")), t("Rate limiting / backoff")],
        ["HTTP 5xx", int_(http.get("5xx")), t("WordPress availability")],
        [t("Other HTTP 4xx"), int_(http.get("4xx")), t("Authentication or endpoint configuration")],
    ])
    st.caption(t("Skipped images (nobody in the frame) are excluded from the WordPress delivery success denominator."))
