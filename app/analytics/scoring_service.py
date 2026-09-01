from datetime import date

from app.analytics.analytics_config import load_analytics_config
from app.analytics.analytics_repository import AnalyticsRepository


def weighted_score(
    workers: int,
    helmet_viol: int,
    vest_viol: int,
    boots_viol: int,
    severity: dict[str, float],
) -> float:
    """
    Severity-weighted compliance score from 0 (every worker missing
    everything) to 100 (no violations).
    """

    if workers <= 0:
        return 100.0

    weight_sum = sum(severity.values())

    penalty = (
        helmet_viol * severity["helmet"]
        + vest_viol * severity["vest"]
        + boots_viol * severity["boots"]
    ) / (workers * weight_sum)

    return round(100.0 * (1.0 - penalty), 1)


class ScoringService:
    """
    Safety scores over the detection history.

    Each camera is its own project, so camera_score is the
    project-level score; overall_score composes all cameras.
    """

    def __init__(
        self,
        repository: AnalyticsRepository,
    ) -> None:
        self._repository = repository

        self._severity = load_analytics_config()["severity"]

    def camera_scores(
        self,
        day: date | None = None,
    ) -> list[dict]:
        """
        Score per camera for one day (default: today).
        """

        day = day or date.today()

        rows = self._repository.daily_stats(day=day)

        return [
            {
                "day": row["day"],
                "camera_id": row["camera_id"],
                "workers": row["workers"],
                "score": weighted_score(
                    workers=row["workers"],
                    helmet_viol=row["helmet_viol"],
                    vest_viol=row["vest_viol"],
                    boots_viol=row["boots_viol"],
                    severity=self._severity,
                ),
            }
            for row in rows
        ]

    def overall_score(
        self,
        day: date | None = None,
    ) -> dict | None:
        """
        Composite score across all cameras for one day, weighted by
        how many workers each camera saw.

        Returns None when no detections exist for that day.
        """

        scores = self.camera_scores(day)

        if not scores:
            return None

        total_workers = sum(row["workers"] for row in scores)

        composite = round(
            sum(row["score"] * row["workers"] for row in scores)
            / total_workers,
            1,
        )

        return {
            "day": scores[0]["day"],
            "workers": total_workers,
            "score": composite,
            "cameras": scores,
        }

