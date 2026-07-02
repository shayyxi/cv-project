from unittest.mock import Mock, create_autospec

from app.ingestion.panomax_client import PanomaxClient
from app.dto import DownloadedImageDTO
from app.ingestion.ingestion_service import IngestionService
from app.storage.repositories.image_job_repository import ImageJobRepository
from app.storage.object_storage import ObjectStorage


def test__poll_camera_downloads_new_image() -> None:
    panomax_client = Mock()
    storage = Mock()
    image_job_repository = Mock()

    panomax_client.download_latest_image.return_value = DownloadedImageDTO(
        url="https://live-image.panomax.com/cams/6168/recent_thumb.jpg",
        content=b"new-image",
        content_type="image/jpeg",
        content_length=len(b"new-image"),
        etag=None,
        last_modified=None,
    )

    image_job_repository.is_new_image.return_value = True

    storage.save_raw_image.return_value = "./data/raw/6168/test.jpg"

    service = IngestionService(
        camera_ids=["87654", "47543"],
        panomax_client=panomax_client,
        storage=storage,
        image_job_repository=image_job_repository,
    )

    result = service._poll_camera("6168")

    assert result is True

    panomax_client.download_latest_image.assert_called_once_with(
        camera_id="6168",
    )

    image_job_repository.is_new_image.assert_called_once()

    storage.save_raw_image.assert_called_once_with(
        camera_id="6168",
        image_bytes=b"new-image",
    )

    image_job_repository.create_downloaded.assert_called_once()


def test__poll_camera_skips_duplicate_image() -> None:
    panomax_client = Mock()
    storage = Mock()
    image_job_repository = Mock()

    panomax_client.download_latest_image.return_value = DownloadedImageDTO(
        url="https://live-image.panomax.com/cams/6168/recent_thumb.jpg",
        content=b"same-image",
        content_type="image/jpeg",
        content_length=len(b"same-image"),
        etag=None,
        last_modified=None,
    )

    image_job_repository.is_new_image.return_value = False

    service = IngestionService(
        camera_ids=["87654", "47543"],
        panomax_client=panomax_client,
        storage=storage,
        image_job_repository=image_job_repository,
    )

    result = service._poll_camera("6168")

    assert result is False

    panomax_client.download_latest_image.assert_called_once_with(
        camera_id="6168",
    )

    image_job_repository.is_new_image.assert_called_once()

    storage.save_raw_image.assert_not_called()
    image_job_repository.create_downloaded.assert_not_called()

def test_poll_calls_all_configured_cameras() -> None:
    service = IngestionService(
        camera_ids=["6168", "6170", "6182"],
        panomax_client=create_autospec(PanomaxClient),
        storage=create_autospec(ObjectStorage),
        image_job_repository=create_autospec(ImageJobRepository),
    )

    service._poll_camera = Mock(return_value=True)

    service.poll()

    assert service._poll_camera.call_count == 3

    service._poll_camera.assert_any_call("6168")
    service._poll_camera.assert_any_call("6170")
    service._poll_camera.assert_any_call("6182")