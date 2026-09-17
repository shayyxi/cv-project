import json
import logging
import threading
import time
from datetime import datetime, timezone

import requests

from app.config import settings
from app.dto import VisionResultDTO


logger = logging.getLogger(__name__)


class WordPressDeliveryError(Exception):
    pass


class RateLimiter:
    """
    Simple process-local rate limiter.

    Default is intentionally 50 requests/minute even though the API
    permits 60/minute, leaving some safety margin.
    """

    def __init__(self, requests_per_minute: int = 50) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")

        self._interval = 60.0 / requests_per_minute
        self._last_request_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()

            elapsed = now - self._last_request_at
            remaining = self._interval - elapsed

            if remaining > 0:
                time.sleep(remaining)

            self._last_request_at = time.monotonic()


class WordPressDeliveryService:
    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(
        self,
        session: requests.Session | None = None,
    ) -> None:
        self._session = session or requests.Session()

        self._endpoint = settings.wordpress_api_endpoint
        self._api_key = settings.wordpress_api_key

        self._timeout = settings.wordpress_timeout_seconds

        self._rate_limiter = RateLimiter(
            settings.wordpress_requests_per_minute
        )

    def deliver(
        self,
        *,
        image_job,
        vision_result: VisionResultDTO,
        processed_image_bytes: bytes,
    ) -> None:
        metadata = self._build_metadata(
            image_job=image_job,
            vision_result=vision_result,
        )

        self._send(
            metadata=metadata,
            image_bytes=processed_image_bytes,
            filename=f"{image_job.id}.jpg",
        )

    def _send(
            self,
            *,
            metadata: dict,
            image_bytes: bytes,
            filename: str,
    ) -> None:
        headers = {
            settings.wordpress_api_key_header: (
                f"{self._api_key}"
            )
        }

        form_data = {
            "metadata[event_id]": metadata["event_id"],
            "metadata[camera_id]": metadata["camera_id"],
            "metadata[captured_at]": metadata["captured_at"],
            "metadata[summary][worker_count]": str(
                metadata["summary"]["worker_count"]
            ),
            "metadata[summary][compliant_count]": str(
                metadata["summary"]["compliant_count"]
            ),
            "metadata[summary][non_compliant_count]": str(
                metadata["summary"]["non_compliant_count"]
            ),
        }

        for index, worker in enumerate(metadata["workers"]):
            prefix = f"metadata[workers][{index}]"

            form_data[f"{prefix}[person_id]"] = str(
                worker["person_id"]
            )

            form_data[f"{prefix}[helmet]"] = (
                "1" if worker["helmet"] else "0"
            )

            form_data[f"{prefix}[vest]"] = (
                "1" if worker["vest"] else "0"
            )

            form_data[f"{prefix}[boots]"] = (
                "1" if worker["boots"] else "0"
            )

            form_data[f"{prefix}[compliant]"] = (
                "1" if worker["compliant"] else "0"
            )

        max_attempts = 4

        for attempt in range(1, max_attempts + 1):
            self._rate_limiter.wait()

            try:
                response = self._session.post(
                    self._endpoint,
                    headers=headers,
                    data=form_data,
                    files={
                        "image": (
                            filename,
                            image_bytes,
                            "image/jpeg",
                        ),
                    },
                    timeout=self._timeout,
                )

            except requests.RequestException as error:
                if attempt == max_attempts:
                    raise WordPressDeliveryError(
                        f"WordPress request failed: {error}"
                    ) from error

                self._backoff(attempt)
                continue

            if 200 <= response.status_code < 300:
                logger.info(
                    "Delivered image_job=%s to WordPress status=%d",
                    metadata["event_id"],
                    response.status_code,
                )
                return

            if response.status_code not in self.RETRYABLE_STATUS_CODES:
                raise WordPressDeliveryError(
                    "WordPress rejected delivery "
                    f"status={response.status_code} "
                    f"body={response.text[:500]}"
                )

            if attempt == max_attempts:
                raise WordPressDeliveryError(
                    "WordPress delivery failed after retries "
                    f"status={response.status_code} "
                    f"body={response.text[:500]}"
                )

            if response.status_code == 429:
                self._wait_for_retry_after(response)
            else:
                self._backoff(attempt)

    @staticmethod
    def _backoff(attempt: int) -> None:
        # 2, 4, 8 seconds
        time.sleep(2 ** attempt)

    @staticmethod
    def _wait_for_retry_after(
        response: requests.Response,
    ) -> None:
        retry_after = response.headers.get("Retry-After")

        try:
            seconds = float(retry_after) if retry_after else 60.0
        except ValueError:
            seconds = 60.0

        logger.warning(
            "WordPress rate limited request. Retrying after %.1fs",
            seconds,
        )

        time.sleep(seconds)

    @staticmethod
    def _build_metadata(
        *,
        image_job,
        vision_result: VisionResultDTO,
    ) -> dict:

        workers = []

        for detection in vision_result.detections:
            if detection.label != "person":
                continue

            workers.append(
                {
                    "person_id": detection.person_id,
                    "helmet": detection.compliance.helmet,
                    "vest": detection.compliance.vest,
                    "boots": detection.compliance.boots,
                    "compliant": detection.compliance.compliant,
                }
            )

        compliant_count = sum(
            1
            for worker in workers
            if worker["compliant"]
        )

        captured_at = (
            image_job.captured_at
            or image_job.downloaded_at
            or image_job.created_at
        )

        return {
            "event_id": image_job.id,
            "camera_id": str(image_job.camera_id),
            "captured_at": _iso_timestamp(captured_at),
            "processed_at": _iso_timestamp(
                datetime.now(timezone.utc)
            ),
            "summary": {
                "worker_count": len(workers),
                "compliant_count": compliant_count,
                "non_compliant_count": (
                    len(workers) - compliant_count
                ),
            },
            "workers": workers,
        }


def _iso_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )