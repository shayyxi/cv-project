import logging

from app.ingestion.ingestion_service import IngestionService
from app.processing.processing_service import ProcessingService

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(
        self,
        ingestion_service: IngestionService,
        processing_service: ProcessingService,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._processing_service = processing_service

    def run_cycle(self) -> None:
        logger.info("Starting pipeline cycle.")

        self._ingestion_service.poll()

        processed_count = 0

        while self._processing_service.process_next():
            processed_count += 1

        logger.info("Pipeline cycle completed. Processed=%d", processed_count)