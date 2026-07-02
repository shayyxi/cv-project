from sqlalchemy.orm import Session

from app.dto import VisionDetectionDTO
from app.storage.models.detection import Detection


class DetectionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(
        self,
        image_job_id: str,
        detections: list[VisionDetectionDTO],
    ) -> list[Detection]:
        db_detections = [
            Detection(
                image_job_id=image_job_id,
                label=detection.label,
                confidence=detection.confidence,
                x_min=detection.box.x_min,
                y_min=detection.box.y_min,
                x_max=detection.box.x_max,
                y_max=detection.box.y_max,
                is_sensitive=detection.is_sensitive,
            )
            for detection in detections
        ]

        self.session.add_all(db_detections)
        self.session.commit()

        for detection in db_detections:
            self.session.refresh(detection)

        return db_detections