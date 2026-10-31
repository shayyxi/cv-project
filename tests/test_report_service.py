import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image as PILImage

from app.analytics.report_service import (
    ReportService,
    slice_risk,
    slice_today_score,
)
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


def _forecast_camera(camera_id, risk_score, predicted, worst, offset=0) -> dict:
    days = [TODAY - timedelta(days=o) for o in range(20, -1, -1)]

    return {
        "camera_id": camera_id,
        "history_days": 21,
        "last_day": TODAY,
        "available": True,
        "latest": {
            "day": TODAY, "crew": 6, "viol": 18, "rate": 3.0,
            "dow": 0, "hour": 13.5, "rate_7d": 2.4,
        },
        "predicted_violations": predicted,
        "risk_score": risk_score,
        "worst_day": {"day": TODAY - timedelta(days=3), "violations": worst},
        "backtest": [
            {
                "day": day,
                "actual": 30 + index + offset,
                "predicted": 31.0 + index * 0.9 + offset,
            }
            for index, day in enumerate(days[1:])
        ],
        "mean_absolute_error": 3.2,
    }


def _pending_camera(camera_id, history_days) -> dict:
    return {
        "camera_id": camera_id,
        "history_days": history_days,
        "last_day": TODAY,
        "available": False,
        "latest": None,
        "predicted_violations": None,
        "risk_score": None,
        "worst_day": None,
        "backtest": [],
        "mean_absolute_error": None,
    }


def _trained(cameras: list[dict], min_days: int) -> dict:
    ready = [camera for camera in cameras if camera["available"]]

    return {
        "available": True,
        "min_days": min_days,
        "camera_min_days": 7,
        "history_days": 21,
        "training_days": 20,
        "training_rows": 42,
        "feature_names": list(FEATURE_LABELS),
        "feature_labels": FEATURE_LABELS,
        "trained_at": datetime(2026, 9, 7, 18, 0),
        "feature_importances": {
            "crew": 0.3, "viol": 0.25, "rate": 0.2,
            "dow": 0.05, "hour": 0.05, "rate_7d": 0.15,
        },
        "mean_absolute_error": 3.2 if ready else None,
        "cameras": cameras,
        "top_camera": max(ready, key=lambda c: c["risk_score"], default=None),
    }


class FakeRisk:
    """Two cameras forecast, one still short of history."""

    def details(self, min_days: int = 14) -> dict:
        return _trained(
            [
                _forecast_camera("12846", 63.0, 41.2, 65),
                _forecast_camera("12875", 28.0, 9.4, 33, offset=-20),
                _pending_camera("12990", 3),
            ],
            min_days,
        )


class AllPendingRisk:
    """Model trained, but no camera has enough logged days yet."""

    def details(self, min_days: int = 14) -> dict:
        return _trained(
            [_pending_camera(camera, 3) for camera in CAMERAS],
            min_days,
        )


class PendingRisk:
    def details(self, min_days: int = 14) -> dict:
        return {
            "available": False,
            "min_days": min_days,
            "camera_min_days": 7,
            "history_days": 4,
            "training_days": 3,
            "cameras": [_pending_camera("12990", 4)],
            "top_camera": None,
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

    # Per-camera reports come from the data unless a test configures
    # cameras explicitly; never from .env.
    extras.setdefault("camera_ids", [])

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


def test_report_with_trained_model_but_every_camera_pending(
    tmp_path: Path,
) -> None:
    path = _service(
        _history(days=8),
        tmp_path,
        risk_service=AllPendingRisk(),
    ).generate_report(days=7, today=TODAY)

    assert path.exists()


def test_optional_service_failure_does_not_block_report(tmp_path: Path) -> None:
    path = _service(
        _history(days=3),
        tmp_path,
        risk_service=FailingRisk(),
    ).generate_report(days=7, today=TODAY)

    assert path.exists()


# ----------------------------------------------------------------------
# Site-wide plus one report per camera
# ----------------------------------------------------------------------


def test_generate_reports_writes_site_and_per_camera_pdfs(tmp_path: Path) -> None:
    heatmaps = FakeHeatmaps(tmp_path)

    paths = _service(
        _history(),
        tmp_path,
        trends_service=FakeTrends(),
        risk_service=FakeRisk(),
        heatmap_service=heatmaps,
    ).generate_reports(days=7, today=TODAY)

    assert [path.name for path in paths] == [
        "report_2026-09-07.pdf",
        "report_2026-09-07_camera_12846.pdf",
        "report_2026-09-07_camera_12875.pdf",
        "report_2026-09-07_camera_12990.pdf",
    ]
    assert all(path.exists() for path in paths)
    assert all(_page_count(path) >= 1 for path in paths)
    # Heatmaps are rendered once by the site run and reused per camera.
    assert sorted(heatmaps.calls) == sorted(CAMERAS)


def test_per_camera_report_uses_only_that_cameras_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.analytics.report_service as module

    calls: list[dict] = []
    real_analyze = module.analyze

    def recording_analyze(**kwargs):
        calls.append(kwargs)
        return real_analyze(**kwargs)

    monkeypatch.setattr(module, "analyze", recording_analyze)

    _service(
        _history(),
        tmp_path,
        trends_service=FakeTrends(),
        risk_service=FakeRisk(),
        heatmap_service=FakeHeatmaps(tmp_path),
    ).generate_reports(days=7, today=TODAY)

    assert len(calls) == 1 + len(CAMERAS)

    site = calls[0]
    assert {row["camera_id"] for row in site["rows"]} == set(CAMERAS)
    assert len(site["repeat_flags"]) == 2

    for call in calls[1:]:
        (camera_id,) = {row["camera_id"] for row in call["rows"]}

        assert all(
            str(flag["camera_id"]) == camera_id for flag in call["repeat_flags"]
        )
        assert [c["camera_id"] for c in call["risk"]["cameras"]] == [camera_id]
        assert [c["camera_id"] for c in call["today_score"]["cameras"]] == [
            camera_id
        ]

    by_camera = {call["rows"][0]["camera_id"]: call for call in calls[1:]}

    # The flagged camera keeps its two flags; the others get none.
    assert len(by_camera["12990"]["repeat_flags"]) == 2
    assert by_camera["12846"]["repeat_flags"] == []
    # FakeRisk: 12846 forecast, 12990 pending.
    assert by_camera["12846"]["risk"]["top_camera"]["camera_id"] == "12846"
    assert by_camera["12990"]["risk"]["top_camera"] is None


def test_configured_camera_without_rows_gets_a_report(tmp_path: Path) -> None:
    paths = _service(
        _history(days=8),
        tmp_path,
        risk_service=FakeRisk(),
        camera_ids=["99999"],
    ).generate_reports(days=7, today=TODAY)

    names = [path.name for path in paths]

    assert len(names) == 1 + len(CAMERAS) + 1
    assert "report_2026-09-07_camera_99999.pdf" in names
    assert (tmp_path / "report_2026-09-07_camera_99999.pdf").exists()


def test_generate_reports_with_no_data_and_no_cameras(tmp_path: Path) -> None:
    paths = _service([], tmp_path).generate_reports(days=7, today=TODAY)

    assert [path.name for path in paths] == ["report_2026-09-07.pdf"]


def test_generate_report_for_one_camera_standalone(tmp_path: Path) -> None:
    heatmaps = FakeHeatmaps(tmp_path)

    path = _service(
        _history(),
        tmp_path,
        trends_service=FakeTrends(),
        risk_service=FakeRisk(),
        heatmap_service=heatmaps,
    ).generate_report(days=7, today=TODAY, camera_id="12875")

    assert path == tmp_path / "report_2026-09-07_camera_12875.pdf"
    assert path.exists()
    assert heatmaps.calls == ["12875"]


def test_slice_today_score() -> None:
    cameras = [
        {"day": TODAY, "camera_id": "12846", "workers": 10, "score": 50.0},
        {"day": TODAY, "camera_id": "12990", "workers": 20, "score": 42.5},
    ]
    today_score = {"day": TODAY, "workers": 30, "score": 45.0, "cameras": cameras}

    assert slice_today_score(today_score, "12990") == {
        "day": TODAY,
        "workers": 20,
        "score": 42.5,
        "cameras": [cameras[1]],
    }
    assert slice_today_score(today_score, "99999") is None
    assert slice_today_score(None, "12990") is None


def test_slice_risk() -> None:
    risk = FakeRisk().details()

    ready = slice_risk(risk, "12875")

    assert [c["camera_id"] for c in ready["cameras"]] == ["12875"]
    assert ready["top_camera"]["camera_id"] == "12875"
    assert ready["mean_absolute_error"] == 3.2
    assert ready["feature_importances"] == risk["feature_importances"]
    assert ready["available"] is True

    pending = slice_risk(risk, "12990")

    assert [c["camera_id"] for c in pending["cameras"]] == ["12990"]
    assert pending["top_camera"] is None
    assert pending["mean_absolute_error"] is None

    unknown = slice_risk(risk, "99999")

    assert unknown["cameras"] == []
    assert unknown["top_camera"] is None

    assert slice_risk(None, "12990") is None
