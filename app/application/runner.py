import logging
import time

from app.config import settings
from app.application.analytics_jobs import AnalyticsJobs
from app.application.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


class ApplicationRunner:

    def __init__(
        self,
        pipeline: PipelineOrchestrator,
        analytics_jobs: AnalyticsJobs | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._analytics_jobs = analytics_jobs

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

            time.sleep(settings.ftp_poll_interval_seconds)

    def run_once(self) -> None:
        logger.info("Running single pipeline cycle.")

        self._pipeline.run_cycle()

        if self._analytics_jobs is not None:
            self._analytics_jobs.run_due()

        logger.info("Pipeline finished.")