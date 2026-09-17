from unittest.mock import create_autospec

from app.dto import (
    BoundingBoxDTO,
    VisionDetectionDTO,
    VisionResultDTO, ComplianceDTO,
)
from app.processing.cv import VisionEngine, VisionRenderer
from app.processing.image_validator import ImageValidator
from app.processing.processing_service import ProcessingService
from app.processing.privacy import PrivacyService
from app.storage.models import ImageJob
from app.storage.models.enums import ImageStatus
from app.storage.object_storage import ObjectStorage
from app.storage.repositories.detection_repository import DetectionRepository
from app.storage.repositories.image_job_repository import ImageJobRepository
from app.processing.Image_cropper import ImageCropper

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
import requests

from app.delivery.wordpress_delivery import (
    WordPressDeliveryError,
    WordPressDeliveryService,
)

class TestProcessingService:
    def setup_method(self) -> None:
        self.object_storage = create_autospec(ObjectStorage)
        self.image_job_repository = create_autospec(
            ImageJobRepository
        )
        self.detection_repository = create_autospec(
            DetectionRepository
        )
        self.image_validator = create_autospec(
            ImageValidator
        )
        self.vision_engine = create_autospec(
            VisionEngine
        )
        self.privacy_service = create_autospec(
            PrivacyService
        )
        self.vision_renderer = create_autospec(
            VisionRenderer
        )
        self.image_cropper = create_autospec(
            ImageCropper
        )
        self.delivery_service = create_autospec(
            WordPressDeliveryService
        )

        self.service = ProcessingService(
            object_storage=self.object_storage,
            image_job_repository=self.image_job_repository,
            detection_repository=self.detection_repository,
            image_validator=self.image_validator,
            vision_engine=self.vision_engine,
            privacy_service=self.privacy_service,
            vision_renderer=self.vision_renderer,
            image_cropper=self.image_cropper,
            delivery_service=self.delivery_service,
        )

        self.image_job = self._create_image_job()

        self.raw_bytes = b"raw-image"
        self.cropped_bytes = b"cropped-image"
        self.annotated_bytes = b"annotated-image"
        self.processed_bytes = b"processed-image"

        self.crop_region = object()

        self.vision_result = self._create_vision_result()

        # -----------------------------------------------------
        # Default successful processing flow
        # -----------------------------------------------------

        self.object_storage.load_image.return_value = (
            self.raw_bytes
        )

        self.image_cropper.crop.return_value = (
            self.cropped_bytes,
            self.crop_region,
        )

        self.vision_engine.process_image.return_value = (
            self.vision_result
        )

        self.image_cropper.translate_result_to_original.return_value = (
            self.vision_result
        )

        self.vision_renderer.draw_original.return_value = (
            self.annotated_bytes
        )

        self.privacy_service.apply_privacy_blur.return_value = (
            self.processed_bytes
        )

        self.object_storage.save_processed_image.return_value = (
            "data/processed/6168/test.jpg"
        )

    @staticmethod
    def _create_image_job() -> ImageJob:
        return ImageJob(
            id="job-1",
            camera_id="6168",
            remote_url=(
                "https://live-image.panomax.com/"
                "cams/6168/recent_thumb.jpg"
            ),
            raw_image_path="data/raw/6168/test.jpg",
            processed_image_path=None,
            sha256="abc123",
            status=ImageStatus.DOWNLOADED,
        )

    @staticmethod
    def _create_vision_result() -> VisionResultDTO:
        return VisionResultDTO(
            worker_count=1,
            detections=[
                VisionDetectionDTO(
                    person_id=0,
                    label="person",
                    confidence=0.95,
                    box=BoundingBoxDTO(
                        x_min=10,
                        y_min=20,
                        x_max=100,
                        y_max=200,
                    ),
                    compliance=ComplianceDTO(
                        helmet=True,
                        vest=False,
                        boots=False,
                        compliant=False,
                    ),
                    is_sensitive=False,
                )
            ],
        )

    def test_process_next_returns_false_when_no_jobs(
        self,
    ) -> None:
        self.image_job_repository.get_next_downloaded.return_value = (
            None
        )

        result = self.service.process_next()

        assert result is False

        self.image_job_repository.mark_processing.assert_not_called()
        self.delivery_service.deliver.assert_not_called()

    def test_process_next_processes_image_job(
        self,
    ) -> None:
        self.image_job_repository.get_next_downloaded.return_value = (
            self.image_job
        )

        result = self.service.process_next()

        assert result is True

        self.image_job_repository.mark_processing.assert_called_once_with(
            self.image_job
        )

        self.object_storage.load_image.assert_called_once_with(
            self.image_job.raw_image_path
        )

        self.image_cropper.crop.assert_called_once_with(
            image_bytes=self.raw_bytes,
            camera_id=self.image_job.camera_id,
        )

        self.image_validator.validate.assert_called_once_with(
            self.cropped_bytes
        )

        self.vision_engine.process_image.assert_called_once_with(
            self.cropped_bytes
        )

        self.image_cropper.translate_result_to_original.assert_called_once_with(
            result=self.vision_result,
            crop_region=self.crop_region,
        )

        self.vision_renderer.draw_original.assert_called_once_with(
            image_bytes=self.raw_bytes,
            result=self.vision_result,
        )

        self.privacy_service.apply_privacy_blur.assert_called_once_with(
            image_bytes=self.annotated_bytes,
            vision_result=self.vision_result,
        )

        self.object_storage.save_processed_image.assert_called_once_with(
            camera_id="6168",
            image_bytes=self.processed_bytes,
        )

        self.detection_repository.create_many.assert_called_once_with(
            image_job_id=self.image_job.id,
            detections=self.vision_result.detections,
        )

        self.image_job_repository.mark_processed.assert_called_once()

        # There is a person detection, therefore WordPress
        # delivery must be attempted.
        self.delivery_service.deliver.assert_called_once_with(
            image_job=self.image_job,
            vision_result=self.vision_result,
            processed_image_bytes=self.processed_bytes,
        )

        self.image_job_repository.mark_delivered.assert_called_once_with(
            image_job=self.image_job,
        )

    def test_process_next_marks_job_failed_on_exception(
        self,
    ) -> None:
        self.image_job_repository.get_next_downloaded.return_value = (
            self.image_job
        )

        self.object_storage.load_image.side_effect = RuntimeError(
            "Storage failure"
        )

        result = self.service.process_next()

        assert result is True

        self.image_job_repository.rollback.assert_called_once()

        self.image_job_repository.mark_failed.assert_called_once_with(
            image_job=self.image_job,
            error_message="Storage failure",
        )

        self.image_job_repository.mark_processed.assert_not_called()

        self.delivery_service.deliver.assert_not_called()

        self.image_job_repository.mark_delivered.assert_not_called()

        self.image_job_repository.mark_delivery_failed.assert_not_called()

    def test_delivery_failure_does_not_mark_processing_failed(
        self,
    ) -> None:
        self.delivery_service.deliver.side_effect = RuntimeError(
            "WordPress unavailable"
        )

        self.service._process_image_job(
            self.image_job
        )

        # CV processing completed successfully.
        self.image_job_repository.mark_processed.assert_called_once()

        self.delivery_service.deliver.assert_called_once_with(
            image_job=self.image_job,
            vision_result=self.vision_result,
            processed_image_bytes=self.processed_bytes,
        )

        # Delivery has its own failure state.
        self.image_job_repository.rollback.assert_called_once()

        self.image_job_repository.mark_delivery_failed.assert_called_once_with(
            image_job=self.image_job,
            error_message="WordPress unavailable",
        )

        # Must NOT convert a WordPress failure into
        # an image-processing failure.
        self.image_job_repository.mark_failed.assert_not_called()

        self.image_job_repository.mark_delivered.assert_not_called()

    def test_no_delivery_when_no_detections(
        self,
    ) -> None:
        self.vision_result.detections = []
        self.vision_result.worker_count = 0

        self.service._process_image_job(
            self.image_job
        )

        # The image itself should still be processed.
        self.image_job_repository.mark_processed.assert_called_once()

        self.detection_repository.create_many.assert_called_once_with(
            image_job_id=self.image_job.id,
            detections=[],
        )

        # No detection means no WordPress API call.
        self.delivery_service.deliver.assert_not_called()

        self.image_job_repository.mark_delivered.assert_not_called()

        self.image_job_repository.mark_delivery_failed.assert_not_called()

    def test_processed_privacy_image_is_sent_to_wordpress(
        self,
    ) -> None:
        self.service._process_image_job(
            self.image_job
        )

        self.delivery_service.deliver.assert_called_once_with(
            image_job=self.image_job,
            vision_result=self.vision_result,
            processed_image_bytes=self.processed_bytes,
        )

        # This additionally proves that raw/annotated bytes
        # aren't what gets passed to delivery.
        call_kwargs = (
            self.delivery_service.deliver.call_args.kwargs
        )

        assert (
            call_kwargs["processed_image_bytes"]
            == self.processed_bytes
        )

        assert (
            call_kwargs["processed_image_bytes"]
            != self.raw_bytes
        )

        assert (
            call_kwargs["processed_image_bytes"]
            != self.annotated_bytes
        )