from datetime import date, timedelta
from pathlib import Path

import joblib
import pytest

from app.analytics.risk_service import (
    CAMERA_MIN_DAYS,
    FEATURE_NAMES,
    MODEL_VERSION,
    RiskService,
)
from app.config import settings

TODAY = date(2026, 9, 7)


class FakeRepository:
    """
    One series per camera, given as {camera_id: days} for `days`
    consecutive days ending today, or {camera_id: (days, end_offset)}
    to end `end_offset` days before today.
    """

    def __init__(self, cameras: dict[str, int | tuple[int, int]]) -> None:
        self.cameras = {
            camera_id: spec if isinstance(spec, tuple) else (spec, 0)
            for camera_id, spec in cameras.items()
        }

    def daily_stats(self, since=None, day=None, camera_id=None) -> list[dict]:
        rows = []

        for camera, (days, end_offset) in self.cameras.items():
            end = TODAY - timedelta(days=end_offset)

            for offset in range(days):
                workers = 10 + offset % 5

                rows.append(
                    {
                        "day": end - timedelta(days=days - 1 - offset),
                        "camera_id": camera,
                        "workers": workers,
                        "compliant": 0,
                        "helmet_viol": offset % 4,
                        "vest_viol": workers,
                        "boots_viol": workers - 1,
                    }
                )

        return sorted(rows, key=lambda row: (row["day"], row["camera_id"]))

    def daily_mean_hours(self) -> dict[tuple[str, date], float]:
        return {}


@pytest.fixture()
def make_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "local_analytics_dir", tmp_path)

    return lambda cameras: RiskService(FakeRepository(cameras))


def test_details_unavailable_with_short_history(make_service) -> None:
    details = make_service({"12990": 5}).details()

    assert details["available"] is False
    assert details["history_days"] == 5
    assert details["training_days"] == 4
    assert details["top_camera"] is None
    assert details["feature_labels"]["rate"] == "violations per worker"

    (camera,) = details["cameras"]

    assert camera["camera_id"] == "12990"
    assert camera["available"] is False
    assert camera["history_days"] == 5
    assert camera["last_day"] == TODAY
    assert camera["risk_score"] is None
    assert camera["backtest"] == []


def test_details_agree_with_score_and_describe_the_fit(
    make_service, tmp_path: Path
) -> None:
    service = make_service({"12990": 20, "12875": 3})

    details = service.details()

    assert details["available"] is True
    assert details["training_days"] == 19
    assert details["training_rows"] == 21
    assert details["camera_min_days"] == CAMERA_MIN_DAYS
    assert (tmp_path / "models" / "risk_model.joblib").exists()
    assert details["trained_at"] is not None

    assert [c["camera_id"] for c in details["cameras"]] == ["12875", "12990"]
    pending, camera = details["cameras"]

    assert pending["available"] is False
    assert pending["history_days"] == 3
    assert pending["risk_score"] is None
    assert pending["backtest"] == []

    assert camera["available"] is True
    assert len(camera["backtest"]) == 19
    assert camera["backtest"][-1]["day"] == TODAY
    assert all(entry["predicted"] >= 0 for entry in camera["backtest"])
    assert camera["latest"]["day"] == TODAY
    assert camera["worst_day"]["violations"] >= max(
        entry["actual"] for entry in camera["backtest"]
    )
    assert camera["mean_absolute_error"] >= 0
    assert 0 <= camera["risk_score"] <= 100

    assert details["top_camera"] is camera
    assert details["mean_absolute_error"] == camera["mean_absolute_error"]
    assert set(details["feature_importances"]) == set(FEATURE_NAMES)
    assert abs(sum(details["feature_importances"].values()) - 1.0) < 1e-6

    assert service.score() == {
        "12990": {
            "predicted_violations": camera["predicted_violations"],
            "risk_score": camera["risk_score"],
        }
    }


def test_top_camera_has_the_highest_risk_score(make_service) -> None:
    details = make_service({"a": 20, "b": 20, "c": 20}).details()

    ready = [c for c in details["cameras"] if c["available"]]

    assert len(ready) == 3
    assert details["top_camera"]["risk_score"] == max(
        c["risk_score"] for c in ready
    )


def test_training_days_count_distinct_days_across_cameras(
    make_service,
) -> None:
    # 14 camera-days of targets, but only 7 distinct days: too little.
    service = make_service({"a": 8, "b": 8})

    details = service.details()

    assert details["training_rows"] == 14
    assert details["training_days"] == 7
    assert details["available"] is False
    assert service.score() is None


def test_model_available_while_every_camera_is_pending(make_service) -> None:
    # Three cameras with six days each on disjoint dates: 15 distinct
    # training days for the pooled model, no camera at CAMERA_MIN_DAYS.
    service = make_service({"a": (6, 0), "b": (6, 6), "c": (6, 12)})

    details = service.details()

    assert details["available"] is True
    assert details["training_days"] == 15
    assert all(not c["available"] for c in details["cameras"])
    assert details["top_camera"] is None
    assert details["mean_absolute_error"] is None
    assert service.score() == {}


def test_stale_model_file_is_retrained(make_service, tmp_path: Path) -> None:
    path = tmp_path / "models" / "risk_model.joblib"
    path.parent.mkdir(parents=True)
    joblib.dump("legacy site-wide model", path)

    details = make_service({"12990": 20}).details()

    assert details["available"] is True

    payload = joblib.load(path)

    assert payload["version"] == MODEL_VERSION
    assert hasattr(payload["model"], "predict")


def test_saved_model_is_reused(make_service, tmp_path: Path) -> None:
    service = make_service({"12990": 20})
    path = tmp_path / "models" / "risk_model.joblib"

    first = service.details()
    saved_at = path.stat().st_mtime_ns

    second = service.details()

    assert path.stat().st_mtime_ns == saved_at
    assert second["top_camera"]["risk_score"] == first["top_camera"]["risk_score"]
