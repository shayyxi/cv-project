from unittest.mock import create_autospec

from app.dto import (
    BoundingBoxDTO,
    VisionDetectionDTO,
    VisionResultDTO,
)
from app.processing.cv import VisionEngine
from app.processing.image_validator import ImageValidator
from app.processing.processing_service import ProcessingService
from app.processing.privacy import PrivacyService
from app.storage.models import ImageJob
from app.storage.models.enums import ImageStatus
from app.storage.object_storage import ObjectStorage
from app.storage.repositories.detection_repository import DetectionRepository
from app.storage.repositories.image_job_repository import ImageJobRepository


def create_image_job() -> ImageJob:
    return ImageJob(
        id="job-1",
        camera_id="6168",
        remote_url="https://live-image.panomax.com/cams/6168/recent_thumb.jpg",
        raw_image_path="data/raw/6168/test.jpg",
        processed_image_path=None,
        sha256="abc123",
        status=ImageStatus.DOWNLOADED,
    )


def create_processing_service() -> tuple[
    ProcessingService,
    ObjectStorage,
    ImageJobRepository,
    DetectionRepository,
    ImageValidator,
    VisionEngine,
    PrivacyService,
]:
    object_storage = create_autospec(ObjectStorage)
    image_job_repository = create_autospec(ImageJobRepository)
    detection_repository = create_autospec(DetectionRepository)
    image_validator = create_autospec(ImageValidator)
    vision_engine = create_autospec(VisionEngine)
    privacy_service = create_autospec(PrivacyService)

    service = ProcessingService(
        object_storage=object_storage,
        image_job_repository=image_job_repository,
        detection_repository=detection_repository,
        image_validator=image_validator,
        vision_engine=vision_engine,
        privacy_service=privacy_service,
    )

    return (
        service,
        object_storage,
        image_job_repository,
        detection_repository,
        image_validator,
        vision_engine,
        privacy_service,
    )


def test_process_next_returns_false_when_no_jobs() -> None:
    (
        service,
        _,
        image_job_repository,
        _,
        _,
        _,
        _,
    ) = create_processing_service()

    image_job_repository.get_next_downloaded.return_value = None

    result = service.process_next()

    assert result is False

    image_job_repository.mark_processing.assert_not_called()


def test_process_next_processes_image_job() -> None:
    (
        service,
        object_storage,
        image_job_repository,
        detection_repository,
        image_validator,
        vision_engine,
        privacy_service,
    ) = create_processing_service()

    image_job = create_image_job()

    image_job_repository.get_next_downloaded.return_value = image_job

    raw_bytes = b"raw-image"
    processed_bytes = b"processed-image"

    object_storage.load_image.return_value = raw_bytes

    vision_result = VisionResultDTO(
        worker_count=1,
        detections=[
            VisionDetectionDTO(
                label="person",
                confidence=0.95,
                box=BoundingBoxDTO(
                    x_min=10,
                    y_min=20,
                    x_max=100,
                    y_max=200,
                ),
                is_sensitive=False,
            )
        ],
    )

    vision_engine.process_image.return_value = vision_result

    privacy_service.apply_privacy_blur.return_value = processed_bytes

    object_storage.save_processed_image.return_value = (
        "data/processed/6168/test.jpg"
    )

    result = service.process_next()

    assert result is True

    image_job_repository.mark_processing.assert_called_once_with(
        image_job
    )

    object_storage.load_image.assert_called_once_with(
        image_job.raw_image_path
    )

    image_validator.validate.assert_called_once_with(raw_bytes)

    vision_engine.process_image.assert_called_once_with(raw_bytes)

    privacy_service.apply_privacy_blur.assert_called_once_with(
        image_bytes=raw_bytes,
        vision_result=vision_result,
    )

    object_storage.save_processed_image.assert_called_once_with(
        camera_id="6168",
        image_bytes=processed_bytes,
    )

    detection_repository.create_many.assert_called_once_with(
        image_job_id=image_job.id,
        detections=vision_result.detections,
    )

    image_job_repository.mark_processed.assert_called_once()


def test_process_next_marks_job_failed_on_exception() -> None:
    (
        service,
        object_storage,
        image_job_repository,
        _,
        _,
        _,
        _,
    ) = create_processing_service()

    image_job = create_image_job()

    image_job_repository.get_next_downloaded.return_value = image_job

    object_storage.load_image.side_effect = RuntimeError(
        "Storage failure"
    )

    result = service.process_next()

    assert result is True

    image_job_repository.mark_failed.assert_called_once()

    image_job_repository.mark_processed.assert_not_called()