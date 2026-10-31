from datetime import date

from app.analytics.risk_service import build_camera_features


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
    assert build_camera_features([]) == {}


def test_groups_rows_into_one_series_per_camera() -> None:
    day = date(2026, 8, 31)

    features = build_camera_features(
        [
            _row(day, workers=2, helmet=1, camera="b"),
            _row(day, workers=3, vest=2, camera="a"),
        ]
    )

    assert list(features) == ["a", "b"]

    assert features["a"][0]["camera_id"] == "a"
    assert features["a"][0]["crew"] == 3
    assert features["a"][0]["viol"] == 2
    assert features["a"][0]["rate"] == 2 / 3

    assert features["b"][0]["crew"] == 2
    assert features["b"][0]["viol"] == 1
    assert features["b"][0]["dow"] == day.weekday()


def test_camera_ids_are_normalised_to_strings() -> None:
    features = build_camera_features(
        [_row(date(2026, 8, 31), workers=1, camera=12990)]
    )

    assert list(features) == ["12990"]


def test_next_viol_is_the_cameras_own_following_day() -> None:
    features = build_camera_features(
        [
            _row(date(2026, 8, 29), workers=1, helmet=1),
            _row(date(2026, 8, 30), workers=1, vest=2),
            _row(date(2026, 8, 31), workers=1),
            # Another camera's day must not become this camera's target.
            _row(date(2026, 8, 30), workers=1, boots=5, camera="other"),
        ]
    )

    series = features["12990"]

    assert series[0]["next_viol"] == 2
    assert series[1]["next_viol"] == 0
    assert "next_viol" not in series[2]
    assert "next_viol" not in features["other"][0]


def test_hour_feature_is_per_camera() -> None:
    day = date(2026, 8, 31)

    features = build_camera_features(
        [
            _row(day, workers=1, camera="a"),
            _row(day, workers=1, camera="b"),
        ],
        mean_hours={("a", day): 14.5},
    )

    assert features["a"][0]["hour"] == 14.5
    assert features["b"][0]["hour"] == 12.0


def test_hour_defaults_to_noon_when_unknown() -> None:
    features = build_camera_features(
        [_row(date(2026, 8, 31), workers=1)]
    )

    assert features["12990"][0]["hour"] == 12.0


def test_rolling_rate_averages_the_cameras_trailing_days() -> None:
    features = build_camera_features(
        [
            _row(date(2026, 8, 30), workers=1, helmet=1),  # rate 1.0
            _row(date(2026, 8, 31), workers=1),            # rate 0.0
            # Same day, other camera: must not leak into the window.
            _row(date(2026, 8, 31), workers=1, helmet=1, camera="other"),
        ]
    )

    assert features["12990"][1]["rate_7d"] == 0.5
    assert features["other"][0]["rate_7d"] == 1.0
