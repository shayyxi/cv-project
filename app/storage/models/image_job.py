from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.database import Base
from app.storage.models.enums import DeliveryStatus, ImageStatus


class ImageJob(Base):
    __tablename__ = "image_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))

    camera_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
    )

    remote_url: Mapped[str] = mapped_column(Text)
    raw_image_path: Mapped[str] = mapped_column(Text)
    processed_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    sha256: Mapped[str] = mapped_column(String(64), index=True)

    captured_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    downloaded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    processing_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[ImageStatus] = mapped_column(
        Enum(ImageStatus),
        default=ImageStatus.DISCOVERED,
        index=True,
    )

    delivery_status: Mapped[DeliveryStatus] = mapped_column(
        Enum(DeliveryStatus),
        default=DeliveryStatus.PENDING,
        index=True,
    )

    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    detections = relationship("Detection", back_populates="image_job", cascade="all, delete-orphan")