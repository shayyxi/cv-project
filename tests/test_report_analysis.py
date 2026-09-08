from datetime import date, timedelta

from app.analytics.report_analysis import (
    aggregate_by_camera,
    analyze,
    class_breakdown,
    fill_days,
    percent_change,
    totals,
    trend_direction,
    weekday_pattern,
)

SEVERITY = {"helmet": 3.0, "vest": 2.0, "boots": 1.0}

TODAY = date(2026, 9, 7)


def _row(day, camera="12990", workers=1, helmet=0, vest=0, boots=0, compliant=None):
    if compliant is None:
        compliant = max(workers - max(helmet, vest, boots), 0)

    return {
        "day": day,
        "camera_id": camera,
        "workers": workers,
        "compliant": compliant,
        "helmet_viol": helmet,
        "vest_viol": vest,
        "boots_viol": boots,
    }


def _analyze(rows, days=7, **kwargs):
    return analyze(
        rows=rows,
        window_start=TODAY - timedelta(days=days),
        prior_start=TODAY - timedelta(days=2 * days),
        history_start=TODAY - timedelta(days=30),
        today=TODAY,
        days=days,
        severity=SEVERITY,
        **kwargs,
    )


def test_totals_sums_rows_and_derives_rates() -> None:
    result = totals(
        [
            _row(TODAY, workers=4, helmet=1, vest=2, boots=4, compliant=0),
            _row(TODAY, camera="12875", workers=6, compliant=6),
        ],
        SEVERITY,
    )

    assert result["workers"] == 10
    assert result["violations"] == 7
    assert result["rate"] == 0.7
    assert result["compliance_rate"] == 0.6
    # penalty = (3 + 4 + 4) / (10 * 6)
    assert result["score"] == 81.7
    assert result["days_active"] == 1


def test_totals_of_nothing_has_no_score() -> None:
    result = totals([], SEVERITY)

    assert result["workers"] == 0
    assert result["score"] is None
    assert result["compliance_rate"] is None


def test_percent_change_handles_zero_prior() -> None:
    assert percent_change(110, 100) == 10.0
    assert percent_change(5, 0) is None


def test_fill_days_pads_missing_days_with_none() -> None:
    start = date(2026, 9, 1)

    filled = fill_days(
        [{"day": date(2026, 9, 2), "workers": 3, "score": 50.0}],
        start,
        date(2026, 9, 3),
    )

    assert [bucket["day"] for bucket in filled] == [
        date(2026, 9, 1),
        date(2026, 9, 2),
        date(2026, 9, 3),
    ]
    assert filled[0]["workers"] is None
    assert filled[1]["workers"] == 3
    assert filled[2]["score"] is None


def test_cameras_sorted_by_violations_with_share() -> None:
    cameras = aggregate_by_camera(
        [
            _row(TODAY, camera="a", workers=2, helmet=1),
            _row(TODAY, camera="b", workers=3, vest=3),
        ],
        SEVERITY,
    )

    assert [c["camera_id"] for c in cameras] == ["b", "a"]
    assert cameras[0]["share"] == 0.75
    assert cameras[1]["rate"] == 0.5


def test_class_breakdown_orders_by_count_and_compares_prior() -> None:
    current = totals([_row(TODAY, workers=10, helmet=2, vest=10, boots=8)], SEVERITY)
    prior = totals([_row(TODAY, workers=5, helmet=1, vest=5)], SEVERITY)

    classes = class_breakdown(current, prior)

    assert [c["name"] for c in classes] == ["vest", "boots", "helmet"]
    assert classes[0]["missing_rate"] == 1.0
    assert classes[0]["prior_missing_rate"] == 1.0
    assert classes[0]["change_pct"] == 100.0
    assert classes[1]["change_pct"] is None  # no boots violations before
    assert classes[2]["share"] == 0.1


def test_trend_direction_rising_flat_and_too_short() -> None:
    def series(rates):
        return [
            {"workers": 10 if rate is not None else None, "rate": rate}
            for rate in rates
        ]

    rising = trend_direction(series([0.5, None, 1.0, 1.5, 2.0]))
    assert rising["direction"] == "rising"
    assert rising["points"] == 4
    assert rising["start"] < rising["end"]

    assert trend_direction(series([1.0, 1.0, 1.0]))["direction"] == "flat"
    assert trend_direction(series([2.0, 1.0, 0.5]))["direction"] == "falling"
    assert trend_direction(series([1.0, 2.0])) is None


def test_weekday_pattern_buckets_by_weekday() -> None:
    monday = date(2026, 8, 31)

    pattern = weekday_pattern(
        [
            _row(monday, workers=2, helmet=2),
            _row(monday + timedelta(days=7), workers=2),
            _row(monday + timedelta(days=1), workers=4, boots=1),
        ],
        SEVERITY,
    )

    assert pattern[0]["name"] == "Mon"
    assert pattern[0]["days"] == 2
    assert pattern[0]["rate"] == 0.5
    assert pattern[1]["rate"] == 0.25
    assert pattern[2]["rate"] is None


def test_analyze_compares_periods_and_lists_coverage_gaps() -> None:
    rows = [
        _row(TODAY - timedelta(days=10), workers=10, helmet=2),
        _row(TODAY - timedelta(days=3), workers=10, helmet=3, vest=1),
        _row(TODAY, workers=5, helmet=1),
    ]

    analysis = _analyze(rows)

    assert analysis["current"]["violations"] == 5
    assert analysis["prior"]["violations"] == 2
    assert analysis["violations_change_pct"] == 150.0
    assert analysis["prior_available"] is True
    assert len(analysis["per_day"]) == 8
    assert analysis["coverage"]["days_with_data"] == 2
    assert TODAY - timedelta(days=1) in analysis["coverage"]["missing"]
    assert analysis["history_days"] == 3
    assert analysis["worst_day"]["day"] == TODAY - timedelta(days=3)
    assert analysis["best_day"]["day"] == TODAY


def test_findings_call_out_systemic_classes_and_no_prior() -> None:
    rows = [
        _row(TODAY, camera="12990", workers=20, helmet=5, vest=20, boots=20),
        _row(TODAY - timedelta(days=1), camera="12875", workers=5, vest=5, boots=5),
    ]

    findings = _analyze(rows)["findings"]
    text = " ".join(findings)

    assert "No data exists for the prior 7-day period" in text
    assert "Missing vest and boots was flagged on virtually every sighting" in text
    assert "Missing helmet was flagged on 20% of sightings (5 of 25)" in text
    assert "Camera 12990 accounted for 82% of all violations" in text
    assert "No worker sighting was fully compliant" in text
    assert "risk forecast is not available" in text


def test_findings_include_flags_risk_and_weekday_pattern() -> None:
    rows = [
        _row(TODAY - timedelta(days=offset), workers=4, helmet=offset % 3)
        for offset in range(21)
    ]

    analysis = _analyze(
        rows,
        repeat_flags=[
            {"camera_id": "12990", "class": "helmet", "violations": 9, "window_days": 7}
        ],
        risk={"predicted_violations": 4.4, "risk_score": 55.0},
    )

    text = " ".join(analysis["findings"])

    assert "1 camera/PPE combination hit the repeated non-compliance threshold" in text
    assert "camera 12990 helmet (9x)" in text
    assert "forecasts about 4 violations tomorrow (risk score 55.0/100" in text
    assert "Over the last 21 logged days" in text
    assert analysis["trend"] is not None


def test_risk_findings_name_top_features_or_history_count() -> None:
    rows = [_row(TODAY, workers=4, helmet=1)]

    available = _analyze(
        rows,
        risk={
            "predicted_violations": 4.4,
            "risk_score": 55.0,
            "feature_importances": {"crew": 0.5, "rate": 0.3, "viol": 0.2},
            "feature_labels": {"crew": "workers seen", "rate": "violations per worker"},
        },
    )

    assert available["risk_available"] is True
    assert (
        "It leans most on workers seen and violations per worker."
        in " ".join(available["findings"])
    )

    pending = _analyze(
        rows,
        risk={"available": False, "min_days": 14, "training_days": 3, "risk_score": None},
    )

    assert pending["risk_available"] is False
    assert "needs at least 14 days of logged history and has 3." in " ".join(
        pending["findings"]
    )


def test_findings_for_empty_period() -> None:
    findings = _analyze([])["findings"]

    assert findings[0].startswith("No worker detections were logged")
    assert any("risk forecast" in finding for finding in findings)
