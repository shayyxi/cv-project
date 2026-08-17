from uuid import uuid4

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))

    image_job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("image_jobs.id"),
        index=True,
    )

    label: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)

    x_min: Mapped[int] = mapped_column(Integer)
    y_min: Mapped[int] = mapped_column(Integer)
    x_max: Mapped[int] = mapped_column(Integer)
    y_max: Mapped[int] = mapped_column(Integer)

    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)

    image_job = relationship("ImageJob", back_populates="detections")

    person_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    helmet_compliant: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    vest_compliant: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    boots_compliant: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    is_compliant: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )