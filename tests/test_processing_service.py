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
    VisionRenderer,
    ImageCropper,
    WordPressDeliveryService,
]:
    object_storage = create_autospec(ObjectStorage)
    image_job_repository = create_autospec(ImageJobRepository)
   
    detection_repository = create_autospec(DetectionRepository)
    image_validator = create_autospec(ImageValidator)
    vision_engine = create_autospec(VisionEngine)
    privacy_service = create_autospec(PrivacyService)
    vision_renderer = create_autospec(VisionRenderer)
    image_cropper = create_autospec(ImageCropper)
    delivery_service = create_autospec(WordPressDeliveryService)

    service = ProcessingService(
        object_storage=object_storage,
        image_job_repository=image_job_repository,
        detection_repository=detection_repository,
        image_validator=image_validator,
        vision_engine=vision_engine,
        privacy_service=privacy_service,
        vision_renderer=vision_renderer,
        image_cropper=image_cropper,
        delivery_service=delivery_service,
    )

    return (
        service,
        object_storage,
        image_job_repository,
        detection_repository,
        image_validator,
        vision_engine,
        privacy_service,
        vision_renderer,
        image_cropper,
        delivery_service,
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
        vision_renderer,
        image_cropper,
        delivery_service,
    ) = create_processing_service()

    image_job = create_image_job()

    image_job_repository.get_next_downloaded.return_value = image_job

    raw_bytes = b"raw-image"
    cropped_bytes = b"cropped-image"
    annotated_bytes = b"annotated-image"
    processed_bytes = b"processed-image"

    crop_region = object()

    object_storage.load_image.return_value = raw_bytes

    image_cropper.crop.return_value = (
        cropped_bytes,
        crop_region,
    )

    vision_result = VisionResultDTO(
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

    vision_engine.process_image.return_value = vision_result

    image_cropper.translate_result_to_original.return_value = (
        vision_result
    )

    vision_renderer.draw_original.return_value = annotated_bytes

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


    image_cropper.crop.assert_called_once_with(
        image_bytes=raw_bytes,
        camera_id=image_job.camera_id,
    )


    image_validator.validate.assert_called_once_with(
        cropped_bytes
    )



    vision_engine.process_image.assert_called_once_with(
        cropped_bytes
    )


    image_cropper.translate_result_to_original.assert_called_once_with(
        result=vision_result,
        crop_region=crop_region,
    )


    vision_renderer.draw_original.assert_called_once_with(
        image_bytes=raw_bytes,
        result=vision_result,
    )

    privacy_service.apply_privacy_blur.assert_called_once_with(
        image_bytes=annotated_bytes,
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

class TestWordPressDeliveryService:
    def setup_method(self) -> None:
        self.session = Mock(spec=requests.Session)

        self.service = WordPressDeliveryService(
            session=self.session,
        )

        # Never actually wait for rate limiting during unit tests.
        self.service._rate_limiter.wait = Mock()

        self.image_job = Mock()
        self.image_job.id = "job-123"
        self.image_job.camera_id = "12846"
        self.image_job.captured_at = datetime(
            2026,
            9,
            11,
            10,
            30,
            tzinfo=timezone.utc,
        )
        self.image_job.downloaded_at = None
        self.image_job.created_at = datetime(
            2026,
            9,
            11,
            10,
            31,
            tzinfo=timezone.utc,
        )

        self.compliant_detection = self._create_person_detection(
            person_id=0,
            helmet=True,
            vest=True,
            boots=True,
            compliant=True,
        )

        self.non_compliant_detection = self._create_person_detection(
            person_id=1,
            helmet=True,
            vest=False,
            boots=True,
            compliant=False,
        )

        self.vision_result = Mock()
        self.vision_result.detections = [
            self.compliant_detection,
            self.non_compliant_detection,
        ]

        self.processed_image_bytes = b"processed-jpeg-data"

    @staticmethod
    def _create_person_detection(
        *,
        person_id: int,
        helmet: bool,
        vest: bool,
        boots: bool,
        compliant: bool,
    ) -> Mock:
        detection = Mock()

        detection.label = "person"
        detection.person_id = person_id

        detection.compliance = Mock()
        detection.compliance.helmet = helmet
        detection.compliance.vest = vest
        detection.compliance.boots = boots
        detection.compliance.compliant = compliant

        return detection

    @staticmethod
    def _response(
        status_code: int,
        *,
        text: str = "",
        headers: dict | None = None,
    ) -> Mock:
        response = Mock(spec=requests.Response)

        response.status_code = status_code
        response.text = text
        response.headers = headers or {}

        return response

    def test_deliver_success(self) -> None:
        self.session.post.return_value = self._response(200)

        self.service.deliver(
            image_job=self.image_job,
            vision_result=self.vision_result,
            processed_image_bytes=self.processed_image_bytes,
        )

        self.session.post.assert_called_once()

        self.service._rate_limiter.wait.assert_called_once()

    def test_deliver_sends_metadata_and_image(self) -> None:
        self.session.post.return_value = self._response(200)

        self.service.deliver(
            image_job=self.image_job,
            vision_result=self.vision_result,
            processed_image_bytes=self.processed_image_bytes,
        )

        _, kwargs = self.session.post.call_args

        assert "files" in kwargs

        files = kwargs["files"]

        assert "metadata" in files
        assert "image" in files

        metadata_part = files["metadata"]
        image_part = files["image"]

        # metadata = (
        #     None,
        #     json_string,
        #     "application/json",
        # )
        assert metadata_part[0] is None
        assert metadata_part[2] == "application/json"

        # image = (
        #     filename,
        #     image_bytes,
        #     "image/jpeg",
        # )
        assert image_part[0] == "job-123.jpg"
        assert image_part[1] == self.processed_image_bytes
        assert image_part[2] == "image/jpeg"

    def test_metadata_contains_expected_summary(self) -> None:
        metadata = self.service._build_metadata(
            image_job=self.image_job,
            vision_result=self.vision_result,
        )

        assert metadata["event_id"] == "job-123"
        assert metadata["camera_id"] == "12846"

        assert metadata["summary"] == {
            "worker_count": 2,
            "compliant_count": 1,
            "non_compliant_count": 1,
        }

        assert len(metadata["workers"]) == 2

    def test_metadata_contains_worker_compliance(self) -> None:
        metadata = self.service._build_metadata(
            image_job=self.image_job,
            vision_result=self.vision_result,
        )

        first_worker = metadata["workers"][0]
        second_worker = metadata["workers"][1]

        assert first_worker == {
            "person_id": 0,
            "helmet": True,
            "vest": True,
            "boots": True,
            "compliant": True,
        }

        assert second_worker == {
            "person_id": 1,
            "helmet": True,
            "vest": False,
            "boots": True,
            "compliant": False,
        }

    def test_metadata_ignores_non_person_detections(self) -> None:
        ppe_detection = Mock()
        ppe_detection.label = "helmet"

        self.vision_result.detections.append(
            ppe_detection
        )

        metadata = self.service._build_metadata(
            image_job=self.image_job,
            vision_result=self.vision_result,
        )

        assert metadata["summary"]["worker_count"] == 2
        assert len(metadata["workers"]) == 2

    def test_metadata_uses_captured_at_first(self) -> None:
        metadata = self.service._build_metadata(
            image_job=self.image_job,
            vision_result=self.vision_result,
        )

        assert (
            metadata["captured_at"]
            == "2026-09-11T10:30:00Z"
        )

    def test_metadata_falls_back_to_downloaded_at(self) -> None:
        self.image_job.captured_at = None

        self.image_job.downloaded_at = datetime(
            2026,
            9,
            11,
            10,
            40,
            tzinfo=timezone.utc,
        )

        metadata = self.service._build_metadata(
            image_job=self.image_job,
            vision_result=self.vision_result,
        )

        assert (
            metadata["captured_at"]
            == "2026-09-11T10:40:00Z"
        )

    def test_metadata_falls_back_to_created_at(self) -> None:
        self.image_job.captured_at = None
        self.image_job.downloaded_at = None

        metadata = self.service._build_metadata(
            image_job=self.image_job,
            vision_result=self.vision_result,
        )

        assert (
            metadata["captured_at"]
            == "2026-09-11T10:31:00Z"
        )

    def test_non_retryable_4xx_raises_immediately(
        self,
    ) -> None:
        self.session.post.return_value = self._response(
            401,
            text="Unauthorized",
        )

        with pytest.raises(
            WordPressDeliveryError
        ) as exc_info:
            self.service.deliver(
                image_job=self.image_job,
                vision_result=self.vision_result,
                processed_image_bytes=self.processed_image_bytes,
            )

        assert "401" in str(exc_info.value)

        assert self.session.post.call_count == 1

    def test_retries_on_429(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.session.post.side_effect = [
            self._response(
                429,
                text="Too Many Requests",
                headers={
                    "Retry-After": "10",
                },
            ),
            self._response(200),
        ]

        sleep_mock = Mock()

        monkeypatch.setattr(
            "app.delivery.wordpress_delivery.time.sleep",
            sleep_mock,
        )

        self.service.deliver(
            image_job=self.image_job,
            vision_result=self.vision_result,
            processed_image_bytes=self.processed_image_bytes,
        )

        assert self.session.post.call_count == 2

        sleep_mock.assert_called_with(10.0)

    def test_429_without_retry_after_uses_60_seconds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.session.post.side_effect = [
            self._response(
                429,
                text="Too Many Requests",
            ),
            self._response(200),
        ]

        sleep_mock = Mock()

        monkeypatch.setattr(
            "app.delivery.wordpress_delivery.time.sleep",
            sleep_mock,
        )

        self.service.deliver(
            image_job=self.image_job,
            vision_result=self.vision_result,
            processed_image_bytes=self.processed_image_bytes,
        )

        assert self.session.post.call_count == 2

        sleep_mock.assert_called_with(60.0)

    def test_retries_on_server_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.session.post.side_effect = [
            self._response(
                503,
                text="Service unavailable",
            ),
            self._response(200),
        ]

        sleep_mock = Mock()

        monkeypatch.setattr(
            "app.delivery.wordpress_delivery.time.sleep",
            sleep_mock,
        )

        self.service.deliver(
            image_job=self.image_job,
            vision_result=self.vision_result,
            processed_image_bytes=self.processed_image_bytes,
        )

        assert self.session.post.call_count == 2

        # First retry uses 2 ** 1 = 2 seconds.
        sleep_mock.assert_called_with(2)

    def test_retries_network_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.session.post.side_effect = [
            requests.Timeout("Connection timed out"),
            self._response(200),
        ]

        sleep_mock = Mock()

        monkeypatch.setattr(
            "app.delivery.wordpress_delivery.time.sleep",
            sleep_mock,
        )

        self.service.deliver(
            image_job=self.image_job,
            vision_result=self.vision_result,
            processed_image_bytes=self.processed_image_bytes,
        )

        assert self.session.post.call_count == 2

        sleep_mock.assert_called_with(2)

    def test_raises_after_maximum_network_attempts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.session.post.side_effect = requests.Timeout(
            "WordPress unavailable"
        )

        monkeypatch.setattr(
            "app.delivery.wordpress_delivery.time.sleep",
            Mock(),
        )

        with pytest.raises(
            WordPressDeliveryError
        ) as exc_info:
            self.service.deliver(
                image_job=self.image_job,
                vision_result=self.vision_result,
                processed_image_bytes=self.processed_image_bytes,
            )

        assert self.session.post.call_count == 4

        assert "WordPress request failed" in str(
            exc_info.value
        )

    def test_raises_after_maximum_server_error_attempts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.session.post.return_value = self._response(
            503,
            text="Service unavailable",
        )

        monkeypatch.setattr(
            "app.delivery.wordpress_delivery.time.sleep",
            Mock(),
        )

        with pytest.raises(
            WordPressDeliveryError
        ) as exc_info:
            self.service.deliver(
                image_job=self.image_job,
                vision_result=self.vision_result,
                processed_image_bytes=self.processed_image_bytes,
            )

        assert self.session.post.call_count == 4

        assert "failed after retries" in str(
            exc_info.value
        )

    def test_rate_limiter_is_used_for_every_attempt(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.session.post.side_effect = [
            self._response(503),
            self._response(503),
            self._response(200),
        ]

        monkeypatch.setattr(
            "app.delivery.wordpress_delivery.time.sleep",
            Mock(),
        )

        self.service.deliver(
            image_job=self.image_job,
            vision_result=self.vision_result,
            processed_image_bytes=self.processed_image_bytes,
        )

        assert self.session.post.call_count == 3

        assert (
            self.service._rate_limiter.wait.call_count
            == 3
        )

def test_no_delivery_when_no_detections():
    object_storage = Mock()
    image_job_repository = Mock()
    detection_repository = Mock()
    image_validator = Mock()
    vision_engine = Mock()
    privacy_service = Mock()
    vision_renderer = Mock()
    image_cropper = Mock()
    delivery_service = Mock()

    image_job = Mock()
    image_job.id = "job-123"
    image_job.camera_id = "12846"
    image_job.raw_image_path = "/data/raw/test.jpg"

    raw_image_bytes = b"raw-image"
    cropped_image_bytes = b"cropped-image"
    annotated_image_bytes = b"annotated-image"
    processed_image_bytes = b"processed-image"

    crop_region = Mock()

    vision_result = Mock()
    vision_result.detections = []

    object_storage.load_image.return_value = raw_image_bytes

    image_cropper.crop.return_value = (
        cropped_image_bytes,
        crop_region,
    )

    vision_engine.process_image.return_value = vision_result

    image_cropper.translate_result_to_original.return_value = (
        vision_result
    )

    vision_renderer.draw_original.return_value = (
        annotated_image_bytes
    )

    privacy_service.apply_privacy_blur.return_value = (
        processed_image_bytes
    )

    object_storage.save_processed_image.return_value = (
        "/data/processed/test.jpg"
    )

    service = ProcessingService(
        object_storage=object_storage,
        image_job_repository=image_job_repository,
        detection_repository=detection_repository,
        image_validator=image_validator,
        vision_engine=vision_engine,
        privacy_service=privacy_service,
        vision_renderer=vision_renderer,
        image_cropper=image_cropper,
        delivery_service=delivery_service,
    )

    service._process_image_job(image_job)

    image_job_repository.mark_processed.assert_called_once()

    delivery_service.deliver.assert_not_called()
    image_job_repository.mark_delivered.assert_not_called()
    image_job_repository.mark_delivery_failed.assert_not_called()