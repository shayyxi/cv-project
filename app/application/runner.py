import logging
import time

from app.config import settings
from app.application.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


class ApplicationRunner:

    def __init__(
        self,
        pipeline: PipelineOrchestrator,
    ) -> None:
        self._pipeline = pipeline

    def run(self) -> None:
        logger.info("Application started.")

        while True:
            self._pipeline.run_cycle()
            time.sleep(settings.ftp_poll_interval_seconds)

    def run_once(self) -> None:
        logger.info("Running single pipeline cycle.")

        self._pipeline.run_cycle()

        logger.info("Pipeline finished.")