from datetime import date, datetime

from sqlalchemy import Date, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.database import Base


class JobRun(Base):
    __tablename__ = "job_runs"

    job_name: Mapped[str] = mapped_column(String(64), primary_key=True)

    last_run_date: Mapped[date] = mapped_column(Date)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
