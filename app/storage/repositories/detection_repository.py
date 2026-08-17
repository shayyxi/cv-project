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
        db_detections = []

        for person in detections:
            db_detections.append(
                Detection(
                    image_job_id=image_job_id,
                    person_id=person.person_id,
                    label=person.label,
                    confidence=person.confidence,
                    x_min=person.box.x_min,
                    y_min=person.box.y_min,
                    x_max=person.box.x_max,
                    y_max=person.box.y_max,
                    is_sensitive=person.is_sensitive,
                    helmet_compliant=person.compliance.helmet,
                    vest_compliant=person.compliance.vest,
                    boots_compliant=person.compliance.boots,
                    is_compliant=person.compliance.compliant,
                )
            )

            for ppe in person.ppe:
                db_detections.append(
                    Detection(
                        image_job_id=image_job_id,
                        person_id=person.person_id,
                        label=ppe.label,
                        confidence=ppe.confidence,
                        x_min=ppe.box.x_min,
                        y_min=ppe.box.y_min,
                        x_max=ppe.box.x_max,
                        y_max=ppe.box.y_max,
                        is_sensitive=False,
                    )
                )

        self.session.add_all(
            db_detections,
        )

        self.session.commit()

        for detection in db_detections:
            self.session.refresh(
                detection,
            )

        return db_detections