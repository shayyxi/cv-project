import logging

from app.analytics.analytics_repository import AnalyticsRepository
from app.ingestion.ingestion_service import IngestionService
from app.processing.processing_service import ProcessingService

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(
        self,
        ingestion_service: IngestionService,
        processing_service: ProcessingService,
        analytics_repository: AnalyticsRepository | None = None,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._processing_service = processing_service
        self._analytics_repository = analytics_repository

    def run_cycle(self) -> None:
        logger.info("Starting pipeline cycle.")

        self._ingestion_service.poll()

        processed_count = 0

        while self._processing_service.process_next():
            processed_count += 1

        if processed_count and self._analytics_repository is not None:
            self._analytics_repository.refresh_views()
            logger.info("Analytics views refreshed.")

        logger.info("Pipeline cycle completed. Processed=%d", processed_count)