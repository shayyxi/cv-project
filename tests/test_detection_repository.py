from datetime import datetime

from sqlalchemy.orm import Session

from app.dto import (
    BoundingBoxDTO,
    ComplianceDTO,
    VisionDetectionDTO,
)
from app.storage.repositories.detection_repository import DetectionRepository
from app.storage.repositories.image_job_repository import ImageJobRepository


def test_create_many_detections(db_session: Session) -> None:
    job_repo = ImageJobRepository(db_session)
    detection_repo = DetectionRepository(db_session)

    job = job_repo.create_downloaded(
        camera_id="2343",
        remote_url="https://live-image.panomax.com/cams/6168/recent_thumb.jpg",
        raw_image_path="./data/raw/6168/test.jpg",
        sha256="abc123",
        downloaded_at=datetime.utcnow(),
    )

    detections = [
        VisionDetectionDTO(
            person_id=0,
            label="person",
            confidence=0.95,
            box=BoundingBoxDTO(
                x_min=10,
                y_min=20,
                x_max=100,
                y_max=120,
            ),
            is_sensitive=False,
            compliance=ComplianceDTO(
                helmet=True,
                vest=False,
                boots=True,
                compliant=False,
            ),
            ppe=[],
        )
    ]

    saved = detection_repo.create_many(job.id, detections)

    assert len(saved) == 1
    assert saved[0].label == "person"
    assert saved[0].is_sensitive is False