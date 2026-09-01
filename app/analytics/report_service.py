import logging
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.analytics.analytics_repository import AnalyticsRepository
from app.analytics.scoring_service import ScoringService
from app.config import settings

logger = logging.getLogger(__name__)


class ReportService:
    """
    PPE compliance report over a rolling window: violation totals,
    trend against the prior period, top risk categories and a
    per-camera breakdown, written as PDF.
    """

    def __init__(
        self,
        repository: AnalyticsRepository,
        scoring_service: ScoringService,
    ) -> None:
        self._repository = repository
        self._scoring_service = scoring_service

        self._output_dir = settings.local_analytics_dir

    def generate_report(
        self,
        days: int = 7,
    ) -> Path:
        today = date.today()

        window_start = today - timedelta(days=days)
        prior_start = today - timedelta(days=2 * days)

        rows = self._repository.daily_stats(since=prior_start)

        current = [row for row in rows if row["day"] >= window_start]
        prior = [row for row in rows if row["day"] < window_start]

        summary = self._summarize(current, prior)

        score = self._scoring_service.overall_score()

        self._output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            self._output_dir / f"report_{today.isoformat()}.pdf"
        )

        self._write_pdf(
            output_path,
            days,
            window_start,
            today,
            summary,
            score,
            current,
        )

        logger.info("Report -> %s", output_path)

        return output_path

    @staticmethod
    def _summarize(
        current: list[dict],
        prior: list[dict],
    ) -> dict:
        def totals(rows: list[dict]) -> dict:
            return {
                "workers": sum(r["workers"] for r in rows),
                "helmet": sum(r["helmet_viol"] for r in rows),
                "vest": sum(r["vest_viol"] for r in rows),
                "boots": sum(r["boots_viol"] for r in rows),
            }

        current_totals = totals(current)
        prior_totals = totals(prior)

        total_violations = (
            current_totals["helmet"]
            + current_totals["vest"]
            + current_totals["boots"]
        )

        prior_violations = (
            prior_totals["helmet"]
            + prior_totals["vest"]
            + prior_totals["boots"]
        )

        if prior_violations:
            change = (
                100.0
                * (total_violations - prior_violations)
                / prior_violations
            )
            trend = f"{change:+.0f}% vs prior period"
        else:
            trend = "no prior-period data"

        categories = sorted(
            [
                ("helmet", current_totals["helmet"]),
                ("vest", current_totals["vest"]),
                ("boots", current_totals["boots"]),
            ],
            key=lambda item: -item[1],
        )

        return {
            "workers": current_totals["workers"],
            "violations": total_violations,
            "trend": trend,
            "categories": categories,
        }

    def _write_pdf(
        self,
        output_path: Path,
        days: int,
        window_start: date,
        today: date,
        summary: dict,
        score: dict | None,
        current: list[dict],
    ) -> None:
        styles = getSampleStyleSheet()

        document = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
        )

        score_text = (
            f"{score['score']}/100"
            if score is not None
            else "no detections today"
        )

        elements = [
            Paragraph("PPE Compliance Report", styles["Title"]),
            Paragraph(
                f"{window_start.isoformat()} to {today.isoformat()} "
                f"({days} days)",
                styles["Normal"],
            ),
            Spacer(1, 0.5 * cm),
            Paragraph(
                f"Workers seen: {summary['workers']} &nbsp;|&nbsp; "
                f"Violations: {summary['violations']} "
                f"({summary['trend']}) &nbsp;|&nbsp; "
                f"Safety score today: {score_text}",
                styles["Normal"],
            ),
            Spacer(1, 0.5 * cm),
            Paragraph("Top risk categories", styles["Heading2"]),
        ]

        for name, count in summary["categories"]:
            elements.append(
                Paragraph(
                    f"missing {name}: {count}",
                    styles["Normal"],
                )
            )

        elements.append(Spacer(1, 0.5 * cm))
        elements.append(
            Paragraph("Daily breakdown", styles["Heading2"])
        )

        table_data = [
            ["day", "camera", "workers", "helmet", "vest", "boots"]
        ]

        for row in current:
            table_data.append(
                [
                    str(row["day"]),
                    row["camera_id"],
                    row["workers"],
                    row["helmet_viol"],
                    row["vest_viol"],
                    row["boots_viol"],
                ]
            )

        if len(table_data) == 1:
            table_data.append(["-", "-", "0", "0", "0", "0"])

        table = Table(table_data)

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )

        elements.append(table)

        document.build(elements)
