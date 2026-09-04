import logging
import time

from sqlalchemy.orm import Session

from app.config import settings
from app.application.analytics_jobs import AnalyticsJobs
from app.application.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


class ApplicationRunner:

    def __init__(
        self,
        pipeline: PipelineOrchestrator,
        analytics_jobs: AnalyticsJobs | None = None,
        db_session: Session | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._analytics_jobs = analytics_jobs
        self._db_session = db_session

    def run(self) -> None:
        logger.info("Application started.")

        while True:
            try:
                self._pipeline.run_cycle()

                if self._analytics_jobs is not None:
                    self._analytics_jobs.run_due()

            except KeyboardInterrupt:
                logger.info("Application stopped.")
                raise
            except Exception:
                logger.exception(
                    "Pipeline cycle failed; retrying next interval."
                )
                self._rollback()

            time.sleep(settings.ftp_poll_interval_seconds)

    def _rollback(self) -> None:
        """
        Clear any aborted transaction on the shared session so the
        next cycle starts clean instead of failing forever with
        InFailedSqlTransaction.
        """

        if self._db_session is None:
            return

        try:
            self._db_session.rollback()
        except Exception:
            logger.exception("Session rollback failed.")

    def run_once(self) -> None:
        logger.info("Running single pipeline cycle.")

        self._pipeline.run_cycle()

        if self._analytics_jobs is not None:
            self._analytics_jobs.run_due()

        logger.info("Pipeline finished.")