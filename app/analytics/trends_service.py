import logging
from datetime import date, datetime, timedelta

from app.analytics.analytics_config import load_analytics_config
from app.analytics.analytics_repository import AnalyticsRepository

logger = logging.getLogger(__name__)


class TrendsService:
    """
    Violation trends and repeated non-compliance flags
    over the detection history.
    """

    def __init__(
        self,
        repository: AnalyticsRepository,
    ) -> None:
        self._repository = repository

        config = load_analytics_config()

        self._trend_window_days = int(
            config["trends"]["window_days"]
        )

        self._repeat_window_days = int(
            config["repeat_noncompliance"]["window_days"]
        )

        self._repeat_min_violations = int(
            config["repeat_noncompliance"]["min_violations"]
        )

    def frequency_trends(
        self,
        days: int | None = None,
        weekly: bool = False,
        camera_id: str | None = None,
    ) -> list[dict]:
        """
        Violation counts per class per period (day, or ISO week
        when weekly=True), oldest first.
        """

        days = days or self._trend_window_days

        since = date.today() - timedelta(days=days)

        if weekly:
            return self._repository.weekly_stats(
                since=since,
                camera_id=camera_id,
            )

        return self._repository.daily_stats(
            since=since,
            camera_id=camera_id,
        )

    def repeat_noncompliance(
        self,
        days: int | None = None,
        min_violations: int | None = None,
    ) -> list[dict]:
        """
        Camera x class combinations whose violation count inside a
        rolling window reaches min_violations.
        """

        days = days or self._repeat_window_days

        min_violations = (
            min_violations
            if min_violations is not None
            else self._repeat_min_violations
        )

        since = datetime.utcnow() - timedelta(days=days)

        rows = self._repository.violation_counts_by_camera(
            since=since,
        )

        flags = []

        for row in rows:
            for ppe_class in ("helmet", "vest", "boots"):

                violations = int(row[ppe_class] or 0)

                if violations >= min_violations:

                    flags.append(
                        {
                            "camera_id": row["camera_id"],
                            "class": ppe_class,
                            "violations": violations,
                            "window_days": days,
                        }
                    )

                    logger.info(
                        "Repeated non-compliance camera_id=%s class=%s "
                        "violations=%d window_days=%d",
                        row["camera_id"],
                        ppe_class,
                        violations,
                        days,
                    )

        return flags
