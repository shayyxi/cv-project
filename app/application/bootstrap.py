from requests import Session

from app.analytics.analytics_repository import AnalyticsRepository
from app.analytics.heatmap_service import HeatmapService
from app.analytics.labeling_service import LabelingService
from app.analytics.report_service import ReportService
from app.analytics.risk_service import RiskService
from app.analytics.scoring_service import ScoringService
from app.application.analytics_jobs import AnalyticsJobs
from app.config import settings
from app.delivery.report_delivery import ReportDeliveryService
from app.ingestion.http_client import HTTPClient
from app.ingestion.ingestion_service import IngestionService
from app.ingestion.panomax_client import PanomaxClient
from app.storage.database import SessionLocal
from app.storage.local_storage import LocalStorage
from app.storage.repositories import DetectionRepository
from app.storage.repositories.image_job_repository import ImageJobRepository
from app.storage.repositories.job_run_repository import JobRunRepository
from app.application.pipeline_orchestrator import (
    PipelineOrchestrator,
)
from app.processing.cv import PPEVisionEngine, VisionRenderer
from app.processing.image_validator import ImageValidator
from app.processing.privacy import FaceBlurPrivacyService
from app.processing.processing_service import ProcessingService
from app.processing.Image_cropper import ImageCropper


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
        #self.vision_engine = DummyVisionEngine() #This changes when the real cv implementation is done
        self.vision_engine = PPEVisionEngine()
        #self.privacy_service = DummyPrivacyService() # This changes when the real cv implementation is done
        self.privacy_service=FaceBlurPrivacyService()
        self.vision_renderer=VisionRenderer()

        self.image_cropper = ImageCropper()

        self.processing_service = ProcessingService(
            object_storage=self.storage,
            image_job_repository=self.image_job_repository,
            detection_repository=self.detection_repository,
            image_validator=self.image_validator,
            vision_engine=self.vision_engine,
            privacy_service=self.privacy_service,
            vision_renderer=self.vision_renderer,
            image_cropper=self.image_cropper,
        )

        self.analytics_repository = AnalyticsRepository(self.db)

        self.pipeline = PipelineOrchestrator(
            ingestion_service=self.ingestion_service,
            processing_service=self.processing_service,
            analytics_repository=self.analytics_repository,
        )

        self.analytics_jobs = AnalyticsJobs(
            report_service=ReportService(
                repository=self.analytics_repository,
                scoring_service=ScoringService(
                    self.analytics_repository,
                ),
            ),
            risk_service=RiskService(self.analytics_repository),
            labeling_service=LabelingService(
                self.analytics_repository,
            ),
            delivery_service=ReportDeliveryService(),
            heatmap_service=HeatmapService(
                self.analytics_repository,
            ),
            job_run_repository=JobRunRepository(self.db),
        )