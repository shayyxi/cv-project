import re
from datetime import date, datetime, timedelta
from pathlib import Path

from PIL import Image as PILImage

from app.analytics.report_service import ReportService
from app.analytics.scoring_service import ScoringService

TODAY = date(2026, 9, 7)

CAMERAS = ("12846", "12875", "12990")


class FakeRepository:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def daily_stats(self, since=None, day=None, camera_id=None) -> list[dict]:
        selected = [
            row
            for row in self.rows
            if (since is None or row["day"] >= since)
            and (day is None or row["day"] == day)
            and (camera_id is None or row["camera_id"] == camera_id)
        ]
        return sorted(selected, key=lambda r: (r["day"], r["camera_id"]))


class FakeTrends:
    def repeat_noncompliance(self, days=None, min_violations=None) -> list[dict]:
        return [
            {"camera_id": "12990", "class": "vest", "violations": 22, "window_days": days or 7},
            {"camera_id": "12990", "class": "boots", "violations": 22, "window_days": days or 7},
        ]


FEATURE_LABELS = {
    "crew": "workers seen",
    "viol": "violations",
    "rate": "violations per worker",
    "dow": "day of week",
    "hour": "mean capture hour",
    "rate_7d": "7-day mean violations per worker",
}


class FakeRisk:
    def details(self, min_days: int = 14) -> dict:
        days = [TODAY - timedelta(days=offset) for offset in range(20, -1, -1)]

        return {
            "available": True,
            "min_days": min_days,
            "history_days": 21,
            "training_days": 20,
            "feature_names": list(FEATURE_LABELS),
            "feature_labels": FEATURE_LABELS,
            "trained_at": datetime(2026, 9, 7, 18, 0),
            "feature_importances": {
                "crew": 0.3, "viol": 0.25, "rate": 0.2,
                "dow": 0.05, "hour": 0.05, "rate_7d": 0.15,
            },
            "latest": {
                "day": TODAY, "crew": 6, "viol": 18, "rate": 3.0,
                "dow": 0, "hour": 13.5, "rate_7d": 2.4,
            },
            "predicted_violations": 41.2,
            "risk_score": 63.0,
            "worst_day": {"day": TODAY - timedelta(days=3), "violations": 65},
            "backtest": [
                {"day": day, "actual": 30 + index, "predicted": 31.0 + index * 0.9}
                for index, day in enumerate(days[1:])
            ],
            "mean_absolute_error": 3.2,
        }


class PendingRisk:
    def details(self, min_days: int = 14) -> dict:
        return {
            "available": False,
            "min_days": min_days,
            "history_days": 4,
            "training_days": 3,
            "risk_score": None,
            "predicted_violations": None,
            "backtest": [],
        }


class FailingRisk:
    def details(self, min_days: int = 14) -> dict:
        raise RuntimeError("model exploded")


class FakeHeatmaps:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.calls: list[str] = []

    def violation_heatmap(self, camera_id: str, days=None, **_) -> Path | None:
        self.calls.append(camera_id)

        if camera_id == "12846":
            raise ValueError("no raw frame on disk")

        path = self.directory / f"heatmap_{camera_id}.jpg"
        PILImage.new("RGB", (320, 180), (200, 80, 40)).save(path, "JPEG")
        return path


def _history(days: int = 21) -> list[dict]:
    rows = []

    for offset in range(days):
        day = TODAY - timedelta(days=offset)

        for index, camera in enumerate(CAMERAS):
            workers = 3 + (offset * 5 + index * 7) % 20
            helmet = (offset + index) % 4
            rows.append(
                {
                    "day": day,
                    "camera_id": camera,
                    "workers": workers,
                    "compliant": 0,
                    "helmet_viol": min(helmet, workers),
                    "vest_viol": workers,
                    "boots_viol": workers,
                }
            )

    return rows


def _page_count(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


def _service(rows, tmp_path: Path, **extras) -> ReportService:
    repository = FakeRepository(rows)

    return ReportService(
        repository=repository,
        scoring_service=ScoringService(repository),
        output_dir=tmp_path,
        **extras,
    )


def test_full_report_with_history_and_optional_services(tmp_path: Path) -> None:
    heatmaps = FakeHeatmaps(tmp_path)

    service = _service(
        _history(),
        tmp_path,
        trends_service=FakeTrends(),
        risk_service=FakeRisk(),
        heatmap_service=heatmaps,
    )

    path = service.generate_report(days=7, today=TODAY)

    assert path == tmp_path / "report_2026-09-07.pdf"
    assert path.stat().st_size > 50_000
    assert _page_count(path) >= 3
    # Every camera with violations is asked for a heatmap; the one
    # without a raw frame is skipped, not fatal.
    assert sorted(heatmaps.calls) == sorted(CAMERAS)


def test_report_without_optional_services(tmp_path: Path) -> None:
    path = _service(_history(days=8), tmp_path).generate_report(days=7, today=TODAY)

    assert path.exists()
    assert _page_count(path) >= 2


def test_report_with_no_data_still_renders(tmp_path: Path) -> None:
    path = _service([], tmp_path).generate_report(days=7, today=TODAY)

    assert path.exists()
    assert _page_count(path) >= 1


def test_report_with_pending_risk_model(tmp_path: Path) -> None:
    path = _service(
        _history(days=4),
        tmp_path,
        risk_service=PendingRisk(),
    ).generate_report(days=7, today=TODAY)

    assert path.exists()


def test_optional_service_failure_does_not_block_report(tmp_path: Path) -> None:
    path = _service(
        _history(days=3),
        tmp_path,
        risk_service=FailingRisk(),
    ).generate_report(days=7, today=TODAY)

    assert path.exists()
