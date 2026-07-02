from requests import Session

from app.config import settings
from app.ingestion.http_client import HTTPClient
from app.ingestion.ingestion_service import IngestionService
from app.ingestion.panomax_client import PanomaxClient
from app.storage.database import SessionLocal
from app.storage.local_storage import LocalStorage
from app.storage.repositories import DetectionRepository
from app.storage.repositories.image_job_repository import ImageJobRepository
from app.application.pipeline_orchestrator import (
    PipelineOrchestrator,
)
from app.processing.cv import DummyVisionEngine
from app.processing.image_validator import ImageValidator
from app.processing.privacy import DummyPrivacyService
from app.processing.processing_service import ProcessingService


class Application:
    def __init__(self) -> None:
        self.db = SessionLocal()

        self.http_session = Session()

        self.http_client = HTTPClient(
            session=self.http_session,
        )

        self.panomax_client = PanomaxClient(
            http_client=self.http_client,
        )

        self.storage = LocalStorage()

        self.storage.ensure_directories()

        self.image_job_repository = ImageJobRepository(
            self.db,
        )

        self.ingestion_service = IngestionService(
            camera_ids=settings.camera_ids,
            panomax_client=self.panomax_client,
            storage=self.storage,
            image_job_repository=self.image_job_repository,
        )

        self.detection_repository = DetectionRepository(self.db)

        self.image_validator = ImageValidator()
        self.vision_engine = DummyVisionEngine() #This changes when the real cv implementation is done
        self.privacy_service = DummyPrivacyService() # This changes when the real cv implementation is done


        self.processing_service = ProcessingService(
            object_storage=self.storage,
            image_job_repository=self.image_job_repository,
            detection_repository=self.detection_repository,
            image_validator=self.image_validator,
            vision_engine=self.vision_engine,
            privacy_service=self.privacy_service,
        )

        self.pipeline = PipelineOrchestrator(
            ingestion_service=self.ingestion_service,
            processing_service=self.processing_service,
        )