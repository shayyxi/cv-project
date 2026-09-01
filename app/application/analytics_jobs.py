import logging
from datetime import datetime

from app.analytics.heatmap_service import HeatmapService
from app.analytics.labeling_service import LabelingService
from app.analytics.report_service import ReportService
from app.analytics.risk_service import RiskService
from app.config import settings
from app.delivery.report_delivery import ReportDeliveryService
from app.storage.repositories.job_run_repository import JobRunRepository

logger = logging.getLogger(__name__)


class AnalyticsJobs:
    """
    Daily Phase 2 jobs run from the main loop:

        - compliance report (with optional webhook/email delivery)
        - predictive risk score attempt
        - violation heatmap per camera
        - label-queue export for retraining

    Jobs run once per calendar day, on the first pipeline cycle at
    or after report_hour. The last-run date is persisted in the
    job_runs table, so jobs do not run twice on the same day after
    a restart. A failing job is logged and never stops the pipeline
    loop.
    """

    JOB_NAME = "daily_analytics"

    def __init__(
        self,
        report_service: ReportService,
        risk_service: RiskService,
        labeling_service: LabelingService,
        delivery_service: ReportDeliveryService,
        heatmap_service: HeatmapService,
        job_run_repository: JobRunRepository,
    ) -> None:
        self._report_service = report_service
        self._risk_service = risk_service
        self._labeling_service = labeling_service
        self._delivery_service = delivery_service
        self._heatmap_service = heatmap_service
        self._job_run_repository = job_run_repository

    def run_due(
        self,
        now: datetime | None = None,
    ) -> bool:
        """
        Run the daily jobs when due. Returns True when they ran.
        """

        now = now or datetime.now()

        if now.hour < settings.report_hour:
            return False

        last_run = self._job_run_repository.get_last_run_date(self.JOB_NAME)

        if last_run == now.date():
            return False

        self._job_run_repository.set_last_run_date(self.JOB_NAME, now.date())

        logger.info("Running daily analytics jobs.")

        self._run_job("report", self._report_job)
        self._run_job("risk", self._risk_job)
        self._run_job("heatmap", self._heatmap_job)
        self._run_job("label-queue", self._labeling_job)

        return True

    @staticmethod
    def _run_job(name, job) -> None:
        try:
            job()
        except Exception:
            logger.exception(
                "Daily analytics job failed: %s",
                name,
            )

    def _report_job(self) -> None:
        report_path = self._report_service.generate_report(
            days=settings.report_days,
        )

        if settings.report_deliver_webhook:
            self._delivery_service.deliver_webhook(report_path)

        if settings.report_deliver_email:
            self._delivery_service.deliver_email(report_path)

    def _risk_job(self) -> None:
        self._risk_service.score()

    def _heatmap_job(self) -> None:
        for camera_id in settings.camera_ids:
            try:
                self._heatmap_service.violation_heatmap(
                    camera_id=camera_id,
                    days=settings.report_days,
                )
            except ValueError as error:
                # e.g. no raw frame on disk yet for this camera
                logger.info(
                    "Heatmap skipped for camera_id=%s: %s",
                    camera_id,
                    error,
                )

    def _labeling_job(self) -> None:
        self._labeling_service.export_violation_crops(
            days=settings.report_days,
        )
