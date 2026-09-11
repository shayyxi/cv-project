from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
import requests

from app.delivery.wordpress_delivery import (
    WordPressDeliveryError,
    WordPressDeliveryService,
)


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