from datetime import date, timedelta
from pathlib import Path

import pytest

from app.analytics.risk_service import FEATURE_NAMES, RiskService
from app.config import settings

TODAY = date(2026, 9, 7)


class FakeRepository:
    """One camera, `days` consecutive days ending today."""

    def __init__(self, days: int) -> None:
        self.days = days

    def daily_stats(self, since=None, day=None, camera_id=None) -> list[dict]:
        rows = []

        for offset in range(self.days):
            workers = 10 + offset % 5

            rows.append(
                {
                    "day": TODAY - timedelta(days=self.days - 1 - offset),
                    "camera_id": "12990",
                    "workers": workers,
                    "compliant": 0,
                    "helmet_viol": offset % 4,
                    "vest_viol": workers,
                    "boots_viol": workers - 1,
                }
            )

        return rows

    def daily_mean_hours(self) -> dict[date, float]:
        return {}


@pytest.fixture()
def make_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "local_analytics_dir", tmp_path)

    return lambda days: RiskService(FakeRepository(days))


def test_details_unavailable_with_short_history(make_service) -> None:
    details = make_service(5).details()

    assert details["available"] is False
    assert details["history_days"] == 5
    assert details["training_days"] == 4
    assert details["risk_score"] is None
    assert details["backtest"] == []
    assert details["feature_labels"]["rate"] == "violations per worker"


def test_details_agree_with_score_and_describe_the_fit(
    make_service, tmp_path: Path
) -> None:
    service = make_service(20)

    details = service.details()

    assert details["available"] is True
    assert details["training_days"] == 19
    assert (tmp_path / "models" / "risk_model.joblib").exists()
    assert details["trained_at"] is not None

    score = service.score()

    assert details["predicted_violations"] == score["predicted_violations"]
    assert details["risk_score"] == score["risk_score"]

    assert len(details["backtest"]) == 19
    assert details["backtest"][-1]["day"] == TODAY
    assert all(entry["predicted"] >= 0 for entry in details["backtest"])

    assert set(details["feature_importances"]) == set(FEATURE_NAMES)
    assert abs(sum(details["feature_importances"].values()) - 1.0) < 1e-6

    assert details["latest"]["day"] == TODAY
    assert details["worst_day"]["violations"] >= max(
        entry["actual"] for entry in details["backtest"]
    )
    assert details["mean_absolute_error"] >= 0


class TwoCameraRepository(FakeRepository):
    """The same series on two cameras; the second has extra helmet violations."""

    def daily_stats(self, since=None, day=None, camera_id=None) -> list[dict]:
        rows = []

        for camera, extra in (("12990", 0), ("12846", 3)):
            for row in super().daily_stats():
                copy = dict(row)
                copy["camera_id"] = camera
                copy["helmet_viol"] += extra
                rows.append(copy)

        return [
            row for row in rows
            if camera_id is None or row["camera_id"] == camera_id
        ]


def test_camera_scoped_service_uses_its_own_history_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "local_analytics_dir", tmp_path)

    repository = TwoCameraRepository(20)
    overall = RiskService(repository)
    camera = overall.for_camera("12846")

    details = camera.details()

    assert details["available"] is True
    assert (tmp_path / "models" / "risk_model_12846.joblib").exists()
    assert not (tmp_path / "models" / "risk_model.joblib").exists()

    # Only that camera's days: crew is one camera's workers, not both.
    camera_rows = repository.daily_stats(camera_id="12846")
    assert details["latest"]["crew"] == camera_rows[-1]["workers"]
    assert overall.details()["latest"]["crew"] == 2 * camera_rows[-1]["workers"]
    assert (tmp_path / "models" / "risk_model.joblib").exists()
