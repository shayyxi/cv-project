from datetime import date

from app.analytics.risk_service import build_daily_features


def _row(day, workers, helmet=0, vest=0, boots=0, camera="12990"):
    return {
        "day": day,
        "camera_id": camera,
        "workers": workers,
        "compliant": workers - max(helmet, vest, boots),
        "helmet_viol": helmet,
        "vest_viol": vest,
        "boots_viol": boots,
    }


def test_empty_history() -> None:
    assert build_daily_features([]) == []


def test_aggregates_cameras_per_day() -> None:
    day = date(2026, 8, 31)

    features = build_daily_features(
        [
            _row(day, workers=2, helmet=1, camera="a"),
            _row(day, workers=3, vest=2, camera="b"),
        ]
    )

    assert len(features) == 1
    assert features[0]["crew"] == 5
    assert features[0]["viol"] == 3
    assert features[0]["rate"] == 3 / 5
    assert features[0]["dow"] == day.weekday()


def test_next_viol_is_following_days_total() -> None:
    features = build_daily_features(
        [
            _row(date(2026, 8, 29), workers=1, helmet=1),
            _row(date(2026, 8, 30), workers=1, vest=2),
            _row(date(2026, 8, 31), workers=1),
        ]
    )

    assert features[0]["next_viol"] == 2
    assert features[1]["next_viol"] == 0
    assert "next_viol" not in features[2]


def test_hour_feature_from_mean_hours() -> None:
    day = date(2026, 8, 31)

    features = build_daily_features(
        [_row(day, workers=1)],
        mean_hours={day: 14.5},
    )

    assert features[0]["hour"] == 14.5


def test_hour_defaults_to_noon_when_unknown() -> None:
    features = build_daily_features(
        [_row(date(2026, 8, 31), workers=1)]
    )

    assert features[0]["hour"] == 12.0


def test_rolling_rate_averages_trailing_days() -> None:
    features = build_daily_features(
        [
            _row(date(2026, 8, 30), workers=1, helmet=1),  # rate 1.0
            _row(date(2026, 8, 31), workers=1),            # rate 0.0
        ]
    )

    assert features[1]["rate_7d"] == 0.5
