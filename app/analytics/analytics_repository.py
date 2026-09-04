from datetime import date, datetime

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session


class AnalyticsRepository:
    """
    Read-only queries over the detection history
    (daily_stats view and detections/image_jobs tables).
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def daily_stats(
        self,
        since: date | None = None,
        day: date | None = None,
        camera_id: str | None = None,
    ) -> list[dict]:
        """
        Rows from the daily_stats view, oldest first.

        since: include days >= since
        day:   include exactly this day
        """

        query = (
            "SELECT day, camera_id, workers, compliant, "
            "helmet_viol, vest_viol, boots_viol "
            "FROM daily_stats"
        )

        conditions = []
        params: dict = {}

        if since is not None:
            conditions.append("day >= :since")
            params["since"] = since

        if day is not None:
            conditions.append("day = :day")
            params["day"] = day

        if camera_id is not None:
            conditions.append("camera_id = :camera_id")
            params["camera_id"] = camera_id

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY day, camera_id"

        rows = self.session.execute(
            text(query),
            params,
        )

        return [dict(row) for row in rows.mappings()]

    def weekly_stats(
        self,
        since: date,
        camera_id: str | None = None,
    ) -> list[dict]:
        """
        Rows from the weekly_stats materialized view, summed across
        cameras per ISO week (period = the week's Monday).
        """

        query = (
            "SELECT period, "
            "SUM(workers) AS workers, "
            "SUM(helmet_viol) AS helmet_viol, "
            "SUM(vest_viol) AS vest_viol, "
            "SUM(boots_viol) AS boots_viol "
            "FROM weekly_stats WHERE period >= :since"
        )

        params: dict = {"since": since}

        if camera_id is not None:
            query += " AND camera_id = :camera_id"
            params["camera_id"] = camera_id

        query += " GROUP BY period ORDER BY period"

        rows = self.session.execute(
            text(query),
            params,
        )

        return [dict(row) for row in rows.mappings()]

    def refresh_views(self) -> None:
        """
        Recompute the daily_stats and weekly_stats materialized
        views from the detections table (daily first - weekly
        derives from it).
        """

        self.session.execute(
            text("REFRESH MATERIALIZED VIEW daily_stats")
        )
        self.session.execute(
            text("REFRESH MATERIALIZED VIEW weekly_stats")
        )
        self.session.commit()

    def daily_mean_hours(self) -> dict[date, float]:
        """
        Mean capture hour (0-23) of person detections per day -
        the time-of-day feature for risk scoring.
        """

        query = (
            "SELECT "
            "CAST(COALESCE(j.captured_at, j.downloaded_at, j.created_at) AS DATE) AS day, "
            "AVG(EXTRACT(HOUR FROM COALESCE(j.captured_at, j.downloaded_at, j.created_at))) AS mean_hour "
            "FROM detections d "
            "JOIN image_jobs j ON j.id = d.image_job_id "
            "WHERE d.label = 'person' "
            "GROUP BY CAST(COALESCE(j.captured_at, j.downloaded_at, j.created_at) AS DATE)"
        )

        rows = self.session.execute(text(query))

        return {
            row["day"]: float(row["mean_hour"])
            for row in rows.mappings()
        }

    def violation_boxes(
        self,
        camera_id: str,
        since: datetime,
    ) -> list[dict]:
        """
        Bounding boxes of non-compliant person detections for one
        camera, captured on or after `since`.
        """

        query = (
            "SELECT d.x_min, d.y_min, d.x_max, d.y_max "
            "FROM detections d "
            "JOIN image_jobs j ON j.id = d.image_job_id "
            "WHERE j.camera_id = :camera_id "
            "AND d.label = 'person' "
            "AND d.is_compliant IS FALSE "
            "AND COALESCE(j.captured_at, j.downloaded_at, j.created_at) >= :since"
        )

        rows = self.session.execute(
            text(query),
            {"camera_id": str(camera_id), "since": since},
        )

        return [dict(row) for row in rows.mappings()]

    def violation_records(
        self,
        since: datetime,
        limit: int = 200,
        include_compliant: bool = False,
    ) -> list[dict]:
        """
        Person detections with their source image path, for exporting
        crops to the labelling queue. By default only non-compliant
        persons; include_compliant=True exports every person.
        """

        compliance_filter = (
            "" if include_compliant else "AND d.is_compliant IS FALSE "
        )

        query = (
            "SELECT j.id AS image_job_id, "
            "j.camera_id, j.raw_image_path, d.person_id, "
            "d.x_min, d.y_min, d.x_max, d.y_max, "
            "COALESCE(j.captured_at, j.downloaded_at, j.created_at) AS ts "
            "FROM detections d "
            "JOIN image_jobs j ON j.id = d.image_job_id "
            "WHERE d.label = 'person' "
            + compliance_filter +
            "AND COALESCE(j.captured_at, j.downloaded_at, j.created_at) >= :since "
            "ORDER BY ts "
            "LIMIT :limit"
        )

        rows = self.session.execute(
            text(query),
            {"since": since, "limit": limit},
        )

        return [dict(row) for row in rows.mappings()]

    def ppe_records(
        self,
        image_job_ids: list[str],
    ) -> list[dict]:
        """
        PPE item detections (helmet / vest / boots boxes, in
        original-image coordinates) for the given image jobs.
        """

        if not image_job_ids:
            return []

        query = text(
            "SELECT d.image_job_id, d.person_id, d.label, "
            "d.confidence, d.x_min, d.y_min, d.x_max, d.y_max "
            "FROM detections d "
            "WHERE d.label != 'person' "
            "AND d.image_job_id IN :image_job_ids"
        ).bindparams(
            bindparam("image_job_ids", expanding=True)
        )

        rows = self.session.execute(
            query,
            {"image_job_ids": image_job_ids},
        )

        return [dict(row) for row in rows.mappings()]

    def violation_counts_by_camera(
        self,
        since: datetime,
    ) -> list[dict]:
        """
        Per-camera violation counts over person detections
        captured on or after `since`.
        """

        query = (
            "SELECT j.camera_id, "
            "COUNT(*) FILTER (WHERE d.helmet_compliant IS FALSE) AS helmet, "
            "COUNT(*) FILTER (WHERE d.vest_compliant IS FALSE) AS vest, "
            "COUNT(*) FILTER (WHERE d.boots_compliant IS FALSE) AS boots "
            "FROM detections d "
            "JOIN image_jobs j ON j.id = d.image_job_id "
            "WHERE d.label = 'person' "
            "AND COALESCE(j.captured_at, j.downloaded_at, j.created_at) >= :since "
            "GROUP BY j.camera_id "
            "ORDER BY j.camera_id"
        )

        rows = self.session.execute(
            text(query),
            {"since": since},
        )

        return [dict(row) for row in rows.mappings()]
