import logging
import time

from app.processing.cv import VisionEngine, VisionRenderer
from app.processing.image_validator import ImageValidator
from app.processing.privacy import PrivacyService
from app.storage.models.image_job import ImageJob
from app.storage.object_storage import ObjectStorage
from app.storage.repositories.detection_repository import DetectionRepository
from app.storage.repositories.image_job_repository import ImageJobRepository

logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(
        self,
        object_storage: ObjectStorage,
        image_job_repository: ImageJobRepository,
        detection_repository: DetectionRepository,
        image_validator: ImageValidator,
        vision_engine: VisionEngine,
        privacy_service: PrivacyService,
        vision_renderer: VisionRenderer,
    ) -> None:
        self._object_storage = object_storage
        self._image_job_repository = image_job_repository
        self._detection_repository = detection_repository
        self._image_validator = image_validator
        self._vision_engine = vision_engine
        self._privacy_service = privacy_service
        self._vision_renderer = vision_renderer

    def process_next(self) -> bool:
        image_job = self._image_job_repository.get_next_downloaded()

        if image_job is None:
            logger.debug("No downloaded image jobs waiting for processing.")
            return False

        self._process_image_job(image_job)
        return True

    def _process_image_job(self, image_job: ImageJob) -> None:
        start_time = time.perf_counter()

        try:
            logger.info(
                "Processing image_job_id=%s camera_id=%s",
                image_job.id,
                image_job.camera_id,
            )

            self._image_job_repository.mark_processing(image_job)

            raw_image_bytes = self._object_storage.load_image(
                image_job.raw_image_path,
            )

            self._image_validator.validate(raw_image_bytes)

            vision_result = self._vision_engine.process_image(
                raw_image_bytes
            )

            processed_image_bytes = self._privacy_service.apply_privacy_blur(
                image_bytes=raw_image_bytes,
                vision_result=vision_result,
            )

            processed_image_bytes = self._vision_renderer.draw_original(
                image_bytes=processed_image_bytes,
                result=vision_result,
            )

            processed_path = self._object_storage.save_processed_image(
                camera_id=image_job.camera_id,
                image_bytes=processed_image_bytes,
            )

            self._detection_repository.create_many(
                image_job_id=image_job.id,
                detections=vision_result.detections,
            )

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            self._image_job_repository.mark_processed(
                image_job=image_job,
                processed_image_path=str(processed_path),
                processing_duration_ms=duration_ms,
            )

            logger.info(
                "Processed image_job_id=%s camera_id=%s duration_ms=%d detections=%d",
                image_job.id,
                image_job.camera_id,
                duration_ms,
                len(vision_result.detections),
            )

        except Exception as exc:
            self._image_job_repository.mark_failed(
                image_job=image_job,
                error_message=str(exc),
            )

            logger.exception(
                "Processing failed image_job_id=%s camera_id=%s",
                image_job.id,
                image_job.camera_id,
            )