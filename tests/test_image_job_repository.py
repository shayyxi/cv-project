from datetime import datetime

from sqlalchemy.orm import Session

from app.storage.models.enums import DeliveryStatus, ImageStatus
from app.storage.repositories.image_job_repository import ImageJobRepository


def test_create_downloaded_image_job(db_session: Session) -> None:
    job_repo = ImageJobRepository(db_session)

    job = job_repo.create_downloaded(
        camera_id="8456",
        remote_url="https://live-image.panomax.com/cams/6168/recent_thumb.jpg",
        raw_image_path="./data/raw/6168/test.jpg",
        sha256="abc123",
        downloaded_at=datetime.utcnow(),
    )

    assert job.id is not None
    assert job.status == ImageStatus.DOWNLOADED
    assert job.delivery_status == DeliveryStatus.PENDING
    assert job.sha256 == "abc123"


def test_get_next_downloaded_image_job(db_session: Session) -> None:
    job_repo = ImageJobRepository(db_session)

    created = job_repo.create_downloaded(
        camera_id="8765",
        remote_url="https://live-image.panomax.com/cams/6168/recent_thumb.jpg",
        raw_image_path="./data/raw/6168/test.jpg",
        sha256="abc123",
        downloaded_at=datetime.utcnow(),
    )

    next_job = job_repo.get_next_downloaded()

    assert next_job is not None
    assert next_job.id == created.id


def test_mark_job_lifecycle(db_session: Session) -> None:
    job_repo = ImageJobRepository(db_session)

    job = job_repo.create_downloaded(
        camera_id="87654",
        remote_url="https://live-image.panomax.com/cams/6168/recent_thumb.jpg",
        raw_image_path="./data/raw/6168/test.jpg",
        sha256="abc123",
        downloaded_at=datetime.utcnow(),
    )

    job = job_repo.mark_processing(job)
    assert job.status == ImageStatus.PROCESSING

    job = job_repo.mark_processed(
        image_job=job,
        processed_image_path="./data/processed/6168/test.jpg",
        processing_duration_ms=1500,
    )
    assert job.status == ImageStatus.PROCESSED
    assert job.processing_duration_ms == 1500

    job = job_repo.mark_delivered(job)
    assert job.status == ImageStatus.DELIVERED
    assert job.delivery_status == DeliveryStatus.DELIVERED
    assert job.delivered_at is not None