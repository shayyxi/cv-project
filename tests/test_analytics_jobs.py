from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from app.application.analytics_jobs import AnalyticsJobs

# settings.report_hour defaults to 18

# What the report service hands back: site-wide first, then per camera.
REPORT_PATHS = (Path("report.pdf"), Path("report_camera_1.pdf"))


class FakeJobRunRepository:
    def __init__(self) -> None:
        self.last_runs: dict[str, date] = {}
        self.session = MagicMock()

    def get_last_run_date(self, job_name: str) -> date | None:
        return self.last_runs.get(job_name)

    def set_last_run_date(self, job_name: str, run_date: date) -> None:
        self.last_runs[job_name] = run_date


def _jobs(
    job_run_repository: FakeJobRunRepository | None = None,
) -> tuple[AnalyticsJobs, MagicMock]:
    report_service = MagicMock()
    report_service.generate_reports.return_value = list(REPORT_PATHS)

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
    report_service.generate_reports.assert_not_called()


def test_due_at_report_hour() -> None:
    jobs, report_service = _jobs()

    ran = jobs.run_due(now=datetime(2026, 8, 31, 18, 5))

    assert ran is True
    report_service.generate_reports.assert_called_once()


def test_runs_only_once_per_day() -> None:
    jobs, report_service = _jobs()

    assert jobs.run_due(now=datetime(2026, 8, 31, 18, 5)) is True
    assert jobs.run_due(now=datetime(2026, 8, 31, 19, 5)) is False
    assert jobs.run_due(now=datetime(2026, 9, 1, 18, 5)) is True

    assert report_service.generate_reports.call_count == 2


def test_does_not_rerun_after_restart_same_day() -> None:
    repository = FakeJobRunRepository()

    jobs, report_service = _jobs(repository)
    assert jobs.run_due(now=datetime(2026, 8, 31, 18, 5)) is True

    # Simulate a restart: a fresh instance sharing the same storage.
    restarted_jobs, restarted_report_service = _jobs(repository)
    assert restarted_jobs.run_due(now=datetime(2026, 8, 31, 20, 0)) is False

    report_service.generate_reports.assert_called_once()
    restarted_report_service.generate_reports.assert_not_called()


def test_failing_job_does_not_stop_others() -> None:
    jobs, report_service = _jobs()

    report_service.generate_reports.side_effect = RuntimeError("boom")

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


def test_no_delivery_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import settings

    # Pin the toggles so the test does not depend on .env.
    monkeypatch.setattr(settings, "report_deliver_webhook", False)
    monkeypatch.setattr(settings, "report_deliver_email", False)

    jobs, _ = _jobs()

    jobs.run_due(now=datetime(2026, 8, 31, 18, 5))

    jobs._delivery_service.deliver_webhook.assert_not_called()
    jobs._delivery_service.deliver_email.assert_not_called()


# 2026-09-04 is a Friday, 09-05 Saturday, 09-06 Sunday, 09-07 Monday.
SATURDAY = datetime(2026, 9, 5, 18, 5)
SUNDAY = datetime(2026, 9, 6, 18, 5)


@pytest.fixture()
def weekdays_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "report_weekdays_only", True)
    monkeypatch.setattr(settings, "report_deliver_webhook", True)
    monkeypatch.setattr(settings, "report_deliver_email", True)


@pytest.mark.parametrize("now", [SATURDAY, SUNDAY])
def test_report_not_generated_or_sent_on_weekend(
    weekdays_only: None,
    now: datetime,
) -> None:
    jobs, report_service = _jobs()

    ran = jobs.run_due(now=now)

    # The daily slot is still consumed; only the report is skipped.
    assert ran is True
    report_service.generate_reports.assert_not_called()
    jobs._delivery_service.deliver_webhook.assert_not_called()
    jobs._delivery_service.deliver_email.assert_not_called()


@pytest.mark.parametrize("now", [SATURDAY, SUNDAY])
def test_other_daily_jobs_still_run_on_weekend(
    weekdays_only: None,
    now: datetime,
) -> None:
    jobs, _ = _jobs()

    jobs.run_due(now=now)

    jobs._risk_service.score.assert_called_once()
    jobs._labeling_service.export_violation_crops.assert_called_once()


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 9, 7, 18, 5),  # Monday
        datetime(2026, 9, 8, 18, 5),  # Tuesday
        datetime(2026, 9, 9, 18, 5),  # Wednesday
        datetime(2026, 9, 10, 18, 5),  # Thursday
        datetime(2026, 9, 4, 18, 5),  # Friday
    ],
)
def test_report_generated_and_sent_monday_to_friday(
    weekdays_only: None,
    now: datetime,
) -> None:
    jobs, report_service = _jobs()

    ran = jobs.run_due(now=now)

    assert ran is True
    report_service.generate_reports.assert_called_once()
    # Every PDF is posted separately; all of them go in one email.
    assert jobs._delivery_service.deliver_webhook.call_args_list == [
        call(path) for path in REPORT_PATHS
    ]
    jobs._delivery_service.deliver_email.assert_called_once_with(
        list(REPORT_PATHS)
    )


def test_weekend_skip_does_not_rerun_next_weekday_twice(
    weekdays_only: None,
) -> None:
    jobs, report_service = _jobs()

    assert jobs.run_due(now=datetime(2026, 9, 4, 18, 5)) is True  # Fri
    assert jobs.run_due(now=SATURDAY) is True
    assert jobs.run_due(now=SUNDAY) is True
    assert jobs.run_due(now=datetime(2026, 9, 7, 18, 5)) is True  # Mon
    assert jobs.run_due(now=datetime(2026, 9, 7, 19, 5)) is False

    assert report_service.generate_reports.call_count == 2


@pytest.mark.parametrize("now", [SATURDAY, SUNDAY])
def test_report_runs_on_weekend_when_weekdays_only_disabled(
    monkeypatch: pytest.MonkeyPatch,
    now: datetime,
) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "report_weekdays_only", False)

    jobs, report_service = _jobs()

    jobs.run_due(now=now)

    report_service.generate_reports.assert_called_once()


def test_is_report_day_covers_monday_to_friday(
    weekdays_only: None,
) -> None:
    week = [date(2026, 9, 7 + offset) for offset in range(7)]  # Mon..Sun

    assert [AnalyticsJobs.is_report_day(day) for day in week] == [
        True, True, True, True, True, False, False,
    ]


def test_failed_webhook_does_not_stop_other_deliveries(
    weekdays_only: None,
) -> None:
    jobs, _ = _jobs()

    jobs._delivery_service.deliver_webhook.side_effect = [
        RuntimeError("boom"),
        200,
    ]

    assert jobs.run_due(now=datetime(2026, 9, 7, 18, 5)) is True  # Mon

    assert jobs._delivery_service.deliver_webhook.call_count == 2
    jobs._delivery_service.deliver_email.assert_called_once_with(
        list(REPORT_PATHS)
    )
