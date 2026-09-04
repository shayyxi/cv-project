from datetime import date

from sqlalchemy.orm import Session

from app.storage.models.job_run import JobRun


class JobRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_last_run_date(self, job_name: str) -> date | None:
        job_run = self.session.get(JobRun, job_name)

        return job_run.last_run_date if job_run is not None else None

    def set_last_run_date(self, job_name: str, run_date: date) -> None:
        job_run = self.session.get(JobRun, job_name)

        if job_run is None:
            job_run = JobRun(job_name=job_name, last_run_date=run_date)
            self.session.add(job_run)
        else:
            job_run.last_run_date = run_date

        self.session.commit()
