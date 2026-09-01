from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

from app.application.analytics_jobs import AnalyticsJobs

# settings.report_hour defaults to 18


class FakeJobRunRepository:
    def __init__(self) -> None:
        self.last_runs: dict[str, date] = {}

    def get_last_run_date(self, job_name: str) -> date | None:
        return self.last_runs.get(job_name)

    def set_last_run_date(self, job_name: str, run_date: date) -> None:
        self.last_runs[job_name] = run_date


def _jobs(
    job_run_repository: FakeJobRunRepository | None = None,
) -> tuple[AnalyticsJobs, MagicMock]:
    report_service = MagicMock()
    report_service.generate_report.return_value = Path("report.pdf")

    jobs = AnalyticsJobs(
        report_service=report_service,
        risk_service=MagicMock(),
        labeling_service=MagicMock(),
        delivery_service=MagicMock(),
        heatmap_service=MagicMock(),
        job_run_repository=job_run_repository or FakeJobRunRepository(),
    )

    return jobs, report_service


def test_not_due_before_report_hour() -> None:
    jobs, report_service = _jobs()

    ran = jobs.run_due(now=datetime(2026, 8, 31, 9, 0))

    assert ran is False
    report_service.generate_report.assert_not_called()


def test_due_at_report_hour() -> None:
    jobs, report_service = _jobs()

    ran = jobs.run_due(now=datetime(2026, 8, 31, 18, 5))

    assert ran is True
    report_service.generate_report.assert_called_once()


def test_runs_only_once_per_day() -> None:
    jobs, report_service = _jobs()

    assert jobs.run_due(now=datetime(2026, 8, 31, 18, 5)) is True
    assert jobs.run_due(now=datetime(2026, 8, 31, 19, 5)) is False
    assert jobs.run_due(now=datetime(2026, 9, 1, 18, 5)) is True

    assert report_service.generate_report.call_count == 2


def test_does_not_rerun_after_restart_same_day() -> None:
    repository = FakeJobRunRepository()

    jobs, report_service = _jobs(repository)
    assert jobs.run_due(now=datetime(2026, 8, 31, 18, 5)) is True

    # Simulate a restart: a fresh instance sharing the same storage.
    restarted_jobs, restarted_report_service = _jobs(repository)
    assert restarted_jobs.run_due(now=datetime(2026, 8, 31, 20, 0)) is False

    report_service.generate_report.assert_called_once()
    restarted_report_service.generate_report.assert_not_called()


def test_failing_job_does_not_stop_others() -> None:
    jobs, report_service = _jobs()

    report_service.generate_report.side_effect = RuntimeError("boom")

    ran = jobs.run_due(now=datetime(2026, 8, 31, 18, 5))

    assert ran is True
    jobs._risk_service.score.assert_called_once()
    jobs._labeling_service.export_violation_crops.assert_called_once()


def test_heatmap_runs_for_every_camera() -> None:
    from app.config import settings

    jobs, _ = _jobs()

    jobs.run_due(now=datetime(2026, 8, 31, 18, 5))

    assert (
        jobs._heatmap_service.violation_heatmap.call_count
        == len(settings.camera_ids)
    )


def test_no_delivery_when_disabled() -> None:
    jobs, _ = _jobs()

    jobs.run_due(now=datetime(2026, 8, 31, 18, 5))

    jobs._delivery_service.deliver_webhook.assert_not_called()
    jobs._delivery_service.deliver_email.assert_not_called()
