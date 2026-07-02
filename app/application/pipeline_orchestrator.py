import logging

from app.ingestion.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        self._ingestion_service = ingestion_service

    def run_cycle(self) -> None:
        logger.info("Starting pipeline cycle.")

        self._ingestion_service.poll()

        logger.info("Pipeline cycle completed.")