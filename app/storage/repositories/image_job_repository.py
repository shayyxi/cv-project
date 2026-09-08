from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.models.enums import DeliveryStatus, ImageStatus
from app.storage.models.image_job import ImageJob


class ImageJobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_downloaded(
        self,
        camera_id: str,
        remote_url: str,
        raw_image_path: str,
        sha256: str,
        downloaded_at: datetime,
    ) -> ImageJob:
        image_job = ImageJob(
            camera_id=camera_id,
            remote_url=remote_url,
            raw_image_path=raw_image_path,
            sha256=sha256,
            downloaded_at=downloaded_at,
            status=ImageStatus.DOWNLOADED,
            delivery_status=DeliveryStatus.PENDING,
        )
        self.session.add(image_job)
        self.session.commit()
        self.session.refresh(image_job)
        return image_job

    def get_next_downloaded(self) -> ImageJob | None:
        statement = (
            select(ImageJob)
            .where(ImageJob.status == ImageStatus.DOWNLOADED)
            .order_by(ImageJob.downloaded_at.asc())
            .limit(1)
        )
        return self.session.scalar(statement)

    def mark_processing(self, image_job: ImageJob) -> ImageJob:
        image_job.status = ImageStatus.PROCESSING
        self.session.commit()
        self.session.refresh(image_job)
        return image_job

    def mark_processed(
        self,
        image_job: ImageJob,
        processed_image_path: str,
        processing_duration_ms: int,
    ) -> ImageJob:
        image_job.status = ImageStatus.PROCESSED
        image_job.processed_image_path = processed_image_path
        image_job.processed_at = datetime.utcnow()
        image_job.processing_duration_ms = processing_duration_ms
        self.session.commit()
        self.session.refresh(image_job)
        return image_job

    def mark_delivered(self, image_job: ImageJob) -> ImageJob:
        image_job.status = ImageStatus.DELIVERED
        image_job.delivery_status = DeliveryStatus.DELIVERED
        image_job.delivered_at = datetime.utcnow()
        self.session.commit()
        self.session.refresh(image_job)
        return image_job

    def mark_failed(self, image_job: ImageJob, error_message: str) -> ImageJob:
        image_job.status = ImageStatus.FAILED
        image_job.error_message = error_message
        image_job.retry_count += 1
        self.session.commit()
        self.session.refresh(image_job)
        return image_job

    def get_latest_hash(
            self,
            camera_id: str,
    ) -> str | None:
        statement = (
            select(ImageJob.sha256)
            .where(ImageJob.camera_id == camera_id)
            .order_by(ImageJob.downloaded_at.desc())
            .limit(1)
        )

        return self.session.scalar(statement)

    def is_new_image(
            self,
            camera_id: str,
            image_hash: str,
    ) -> bool:
        latest_hash = self.get_latest_hash(camera_id)

        return latest_hash != image_hash

    def delete_all(self) -> int:
        jobs = self.session.query(ImageJob).all()

        count = len(jobs)

        for job in jobs:
            self.session.delete(job)

        self.session.commit()

        return count