"""
Analysis behind the PPE compliance report.

Pure functions over daily_stats rows (one row per day per camera):
period totals, per-day and per-camera aggregates, PPE-class
breakdowns, trend direction, weekday pattern, coverage gaps and the
plain-language findings. No database or rendering here, so every
number that ends up in the PDF is unit-testable.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

from app.analytics.scoring_service import weighted_score

PPE_CLASSES = ("helmet", "vest", "boots")

# Relative change in violations-per-worker across the window (by
# linear fit) below which the trend is reported as flat.
FLAT_TREND_THRESHOLD = 0.10

# Share of sightings at or above which a missing item is called out
# as systemic (site-wide gap or detector limitation).
SYSTEMIC_MISSING_RATE = 0.95

# Distinct logged days needed before a weekday pattern is reported.
MIN_DAYS_FOR_WEEKDAY_PATTERN = 14

# History the risk model needs before it can forecast.
RISK_MODEL_MIN_DAYS = 14


def _int(value) -> int:
    return int(value or 0)


def percent_change(current: float, prior: float) -> float | None:
    """Percent change from prior to current; None when prior is 0."""

    if not prior:
        return None

    return 100.0 * (current - prior) / prior


def totals(rows: list[dict], severity: dict[str, float]) -> dict:
    """
    Sum daily_stats rows into one bucket: workers, compliant
    sightings, violations per class and overall, violations per
    worker, compliance rate and the severity-weighted score.
    """

    workers = sum(_int(r["workers"]) for r in rows)
    compliant = sum(_int(r.get("compliant")) for r in rows)

    by_class = {
        ppe_class: sum(_int(r[f"{ppe_class}_viol"]) for r in rows)
        for ppe_class in PPE_CLASSES
    }

    violations = sum(by_class.values())

    return {
        "workers": workers,
        "compliant": compliant,
        **by_class,
        "violations": violations,
        "rate": violations / workers if workers else 0.0,
        "compliance_rate": compliant / workers if workers else None,
        "score": (
            weighted_score(
                workers,
                by_class["helmet"],
                by_class["vest"],
                by_class["boots"],
                severity,
            )
            if workers
            else None
        ),
        "days_active": len({r["day"] for r in rows}),
    }


def aggregate_by_day(
    rows: list[dict],
    severity: dict[str, float],
) -> list[dict]:
    """One totals() bucket per day across all cameras, oldest first."""

    by_day: dict[date, list[dict]] = {}

    for row in rows:
        by_day.setdefault(row["day"], []).append(row)

    result = []

    for day in sorted(by_day):
        bucket = totals(by_day[day], severity)
        bucket["day"] = day
        result.append(bucket)

    return result


def fill_days(
    per_day: list[dict],
    start: date,
    end: date,
) -> list[dict]:
    """
    One entry per calendar day from start to end inclusive. Days
    without data carry None values so charts show a gap rather
    than a zero.
    """

    lookup = {bucket["day"]: bucket for bucket in per_day}

    filled = []
    day = start

    while day <= end:
        bucket = lookup.get(day)

        if bucket is None:
            bucket = {
                "day": day,
                "workers": None,
                "compliant": None,
                "helmet": None,
                "vest": None,
                "boots": None,
                "violations": None,
                "rate": None,
                "compliance_rate": None,
                "score": None,
                "days_active": 0,
            }

        filled.append(bucket)
        day += timedelta(days=1)

    return filled


def aggregate_by_camera(
    rows: list[dict],
    severity: dict[str, float],
) -> list[dict]:
    """
    One totals() bucket per camera with its share of all
    violations, most violations first.
    """

    by_camera: dict[str, list[dict]] = {}

    for row in rows:
        by_camera.setdefault(str(row["camera_id"]), []).append(row)

    grand_total = sum(
        _int(r["helmet_viol"]) + _int(r["vest_viol"]) + _int(r["boots_viol"])
        for r in rows
    )

    cameras = []

    for camera_id, camera_rows in by_camera.items():
        bucket = totals(camera_rows, severity)
        bucket["camera_id"] = camera_id
        bucket["share"] = (
            bucket["violations"] / grand_total if grand_total else 0.0
        )
        cameras.append(bucket)

    cameras.sort(key=lambda c: (-c["violations"], c["camera_id"]))

    return cameras


def class_breakdown(current: dict, prior: dict) -> list[dict]:
    """
    Per PPE class: violation count, share of all violations, share
    of sightings missing the item, and the same for the prior
    period, most violations first.
    """

    classes = []

    for ppe_class in PPE_CLASSES:
        count = current[ppe_class]
        prior_count = prior[ppe_class]

        classes.append(
            {
                "name": ppe_class,
                "count": count,
                "share": (
                    count / current["violations"]
                    if current["violations"]
                    else 0.0
                ),
                "missing_rate": (
                    count / current["workers"] if current["workers"] else None
                ),
                "prior_count": prior_count,
                "prior_missing_rate": (
                    prior_count / prior["workers"] if prior["workers"] else None
                ),
                "change_pct": percent_change(count, prior_count),
            }
        )

    classes.sort(key=lambda c: -c["count"])

    return classes


def trend_direction(per_day: list[dict]) -> dict | None:
    """
    Least-squares fit of violations per worker over the (filled)
    daily series. Returns the fitted start/end values and a
    direction of rising, falling or flat; None with fewer than
    three logged days.
    """

    points = [
        (index, bucket["rate"])
        for index, bucket in enumerate(per_day)
        if bucket["workers"]
    ]

    if len(points) < 3:
        return None

    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n

    sxx = sum((x - mean_x) ** 2 for x, _ in points)

    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in points) / sxx
        if sxx
        else 0.0
    )

    intercept = mean_y - slope * mean_x

    start = intercept + slope * points[0][0]
    end = intercept + slope * points[-1][0]

    relative = (end - start) / mean_y if mean_y else 0.0

    if relative > FLAT_TREND_THRESHOLD:
        direction = "rising"
    elif relative < -FLAT_TREND_THRESHOLD:
        direction = "falling"
    else:
        direction = "flat"

    return {
        "slope": slope,
        "start": max(start, 0.0),
        "end": max(end, 0.0),
        "relative_change": relative,
        "direction": direction,
        "points": n,
    }


def weekday_pattern(
    rows: list[dict],
    severity: dict[str, float],
) -> list[dict]:
    """
    Workers, violations, logged days and violations per worker for
    each weekday (0 = Monday) over all rows given.
    """

    buckets = {
        weekday: {"workers": 0, "violations": 0, "days": 0}
        for weekday in range(7)
    }

    for day_bucket in aggregate_by_day(rows, severity):
        bucket = buckets[day_bucket["day"].weekday()]
        bucket["workers"] += day_bucket["workers"]
        bucket["violations"] += day_bucket["violations"]
        bucket["days"] += 1

    return [
        {
            "weekday": weekday,
            "name": calendar.day_abbr[weekday],
            **bucket,
            "rate": (
                bucket["violations"] / bucket["workers"]
                if bucket["workers"]
                else None
            ),
        }
        for weekday, bucket in buckets.items()
    ]


def coverage(per_day: list[dict]) -> dict:
    """Days in the (filled) window with and without detections."""

    missing = [bucket["day"] for bucket in per_day if not bucket["workers"]]

    return {
        "days_total": len(per_day),
        "days_with_data": len(per_day) - len(missing),
        "missing": missing,
    }


def analyze(
    *,
    rows: list[dict],
    window_start: date,
    prior_start: date,
    history_start: date,
    today: date,
    days: int,
    severity: dict[str, float],
    repeat_flags: list[dict] | None = None,
    repeat_threshold: dict | None = None,
    risk: dict | None = None,
    today_score: dict | None = None,
) -> dict:
    """
    Everything the report renders, computed once.

    rows           daily_stats rows since history_start (the trend
                   window), oldest first
    window_start   first day of the report period (inclusive)
    prior_start    first day of the comparison period
    history_start  first day of the long-range trend charts
    """

    current_rows = [r for r in rows if r["day"] >= window_start]
    prior_rows = [
        r for r in rows if prior_start <= r["day"] < window_start
    ]

    current = totals(current_rows, severity)
    prior = totals(prior_rows, severity)

    per_day = fill_days(
        aggregate_by_day(current_rows, severity),
        window_start,
        today,
    )

    per_day_prior = fill_days(
        aggregate_by_day(prior_rows, severity),
        prior_start,
        window_start - timedelta(days=1),
    )

    # Long-range series: never shorter than the two compared
    # periods, never padded before the first logged day.
    earliest = min((r["day"] for r in rows), default=prior_start)
    chart_start = min(prior_start, max(history_start, earliest))

    per_day_history = fill_days(
        aggregate_by_day(rows, severity),
        chart_start,
        today,
    )

    cameras = aggregate_by_camera(current_rows, severity)

    today_by_camera = {
        str(entry["camera_id"]): entry["score"]
        for entry in (today_score or {}).get("cameras", [])
    }

    for camera in cameras:
        camera["score_today"] = today_by_camera.get(camera["camera_id"])

    active_days = [bucket for bucket in per_day if bucket["workers"]]

    worst_day = (
        min(active_days, key=lambda d: (d["score"], -d["violations"]))
        if active_days
        else None
    )
    best_day = (
        max(active_days, key=lambda d: (d["score"], -d["violations"]))
        if active_days
        else None
    )

    analysis = {
        "days": days,
        "window_start": window_start,
        "prior_start": prior_start,
        "today": today,
        "current": current,
        "prior": prior,
        "prior_available": bool(prior_rows),
        "violations_change_pct": percent_change(
            current["violations"], prior["violations"]
        ),
        "workers_change_pct": percent_change(
            current["workers"], prior["workers"]
        ),
        "score_change": (
            current["score"] - prior["score"]
            if current["score"] is not None and prior["score"] is not None
            else None
        ),
        "per_day": per_day,
        "per_day_prior": per_day_prior,
        "per_day_history": per_day_history,
        "cameras": cameras,
        "classes": class_breakdown(current, prior),
        "trend": trend_direction(per_day),
        "weekdays": weekday_pattern(rows, severity),
        "history_days": len({r["day"] for r in rows}),
        "coverage": coverage(per_day),
        "worst_day": worst_day,
        "best_day": best_day,
        "repeat_flags": list(repeat_flags or []),
        "repeat_threshold": repeat_threshold,
        "risk": risk,
        # RiskService.details() always returns a dict; the score is
        # None until the model has enough history to train.
        "risk_available": bool(risk and risk.get("risk_score") is not None),
        "today_score": today_score,
        "rows": sorted(
            current_rows,
            key=lambda r: (r["day"], str(r["camera_id"])),
        ),
    }

    analysis["findings"] = build_findings(analysis)

    return analysis


# ======================================================================
# Findings
# ======================================================================


def _pct(value: float) -> str:
    return f"{100.0 * value:.0f}%"


def _join(items) -> str:
    items = list(items)

    if len(items) <= 1:
        return "".join(items)

    return ", ".join(items[:-1]) + " and " + items[-1]


def _list_days(days: list[date], limit: int = 3) -> str:
    shown = _join(d.isoformat() for d in days[:limit])
    extra = len(days) - limit

    return shown + (f" and {extra} more" if extra > 0 else "")


def build_findings(analysis: dict) -> list[str]:
    """Plain-language findings, most important first."""

    current = analysis["current"]
    prior = analysis["prior"]
    days = analysis["days"]

    findings: list[str] = []

    if current["workers"]:
        findings.extend(_period_findings(analysis, current, prior, days))
    else:
        findings.append(
            f"No worker detections were logged in the last {days} days, "
            "so there is nothing to score for this period."
        )

    if analysis["history_days"] >= MIN_DAYS_FOR_WEEKDAY_PATTERN:
        weekdays = [w for w in analysis["weekdays"] if w["rate"] is not None]

        if len(weekdays) >= 2:
            highest = max(weekdays, key=lambda w: w["rate"])
            lowest = min(weekdays, key=lambda w: w["rate"])

            findings.append(
                f"Over the last {analysis['history_days']} logged days, "
                f"{calendar.day_name[highest['weekday']]}s had the highest "
                f"violations per worker ({highest['rate']:.2f}) and "
                f"{calendar.day_name[lowest['weekday']]}s the lowest "
                f"({lowest['rate']:.2f})."
            )

    cov = analysis["coverage"]

    if cov["missing"] and cov["days_with_data"]:
        findings.append(
            f"Detections were logged on {cov['days_with_data']} of "
            f"{cov['days_total']} days; no data on "
            f"{_list_days(cov['missing'])}. Check camera uptime before "
            "reading the gaps as compliance."
        )

    flags = analysis["repeat_flags"]

    if flags:
        listed = _join(
            f"camera {f['camera_id']} {f['class']} ({f['violations']}x)"
            for f in flags[:4]
        )
        more = f" and {len(flags) - 4} more" if len(flags) > 4 else ""

        findings.append(
            f"{len(flags)} camera/PPE combination"
            f"{'s' if len(flags) != 1 else ''} hit the repeated "
            f"non-compliance threshold in the last "
            f"{flags[0]['window_days']} days: {listed}{more}."
        )

    risk = analysis["risk"] or {}

    if analysis["risk_available"]:
        text = (
            f"The risk model forecasts about "
            f"{risk['predicted_violations']:.0f} violations tomorrow "
            f"(risk score {risk['risk_score']}/100 against the worst day "
            "on record)."
        )

        importances = risk.get("feature_importances") or {}
        labels = risk.get("feature_labels") or {}

        if importances:
            top = sorted(importances, key=importances.get, reverse=True)[:2]
            text += (
                " It leans most on "
                f"{_join(labels.get(name, name) for name in top)}."
            )

        findings.append(text)
    else:
        min_days = risk.get("min_days", RISK_MODEL_MIN_DAYS)
        logged = risk.get("training_days")

        findings.append(
            "Tomorrow's risk forecast is not available yet; the model "
            f"needs at least {min_days} days of logged history"
            + (f" and has {logged}." if logged is not None else ".")
        )

    return findings


def _period_findings(
    analysis: dict,
    current: dict,
    prior: dict,
    days: int,
) -> list[str]:
    findings: list[str] = []

    # Period-over-period movement, normalised by crew size.
    if analysis["prior_available"] and prior["violations"]:
        change = analysis["violations_change_pct"]

        if abs(change) < 0.5:
            text = (
                f"Violations were flat against the prior {days}-day period "
                f"({current['violations']} vs {prior['violations']})."
            )
        else:
            verb = "rose" if change > 0 else "fell"
            text = (
                f"Violations {verb} {abs(change):.0f}% against the prior "
                f"{days}-day period ({current['violations']} vs "
                f"{prior['violations']})."
            )

        text += (
            f" Workers seen went from {prior['workers']} to "
            f"{current['workers']}, so violations per worker moved from "
            f"{prior['rate']:.2f} to {current['rate']:.2f}."
        )

        findings.append(text)

    elif analysis["prior_available"]:
        findings.append(
            f"The prior {days}-day period had {prior['workers']} worker "
            f"sightings and no violations; this period logged "
            f"{current['violations']}."
        )

    else:
        findings.append(
            f"No data exists for the prior {days}-day period yet, so "
            "period-over-period comparisons start with the next report."
        )

    # Full compliance.
    if current["compliant"] == 0:
        findings.append(
            f"No worker sighting was fully compliant (0 of "
            f"{current['workers']} had helmet, vest and boots together)."
        )
    else:
        findings.append(
            f"{_pct(current['compliance_rate'])} of worker sightings were "
            f"fully compliant ({current['compliant']} of "
            f"{current['workers']})."
        )

    # Per-class rates, with systemic classes called out together.
    systemic = [
        c
        for c in analysis["classes"]
        if c["missing_rate"] is not None
        and c["missing_rate"] >= SYSTEMIC_MISSING_RATE
    ]

    if systemic:
        names = _join(c["name"] for c in systemic)
        rates = ", ".join(
            f"{c['name']} {_pct(c['missing_rate'])}" for c in systemic
        )

        findings.append(
            f"Missing {names} was flagged on virtually every sighting "
            f"({rates}). A rate this high usually means either a "
            "site-wide gap or a detector limitation (an item hidden at "
            "this camera distance); spot-check a few frames before "
            "acting on it."
        )

    for ppe_class in analysis["classes"]:
        if ppe_class in systemic or ppe_class["missing_rate"] is None:
            continue

        text = (
            f"Missing {ppe_class['name']} was flagged on "
            f"{_pct(ppe_class['missing_rate'])} of sightings "
            f"({ppe_class['count']} of {current['workers']})"
        )

        if ppe_class["change_pct"] is not None:
            text += f", {ppe_class['change_pct']:+.0f}% vs the prior period"

        findings.append(text + ".")

    # Cameras.
    cameras = [c for c in analysis["cameras"] if c["workers"]]

    if len(cameras) >= 2:
        top = cameras[0]
        by_rate = sorted(cameras, key=lambda c: c["rate"])
        lowest, highest = by_rate[0], by_rate[-1]

        text = (
            f"Camera {top['camera_id']} accounted for {_pct(top['share'])} "
            f"of all violations ({top['violations']} over "
            f"{top['workers']} workers)."
        )

        if round(lowest["rate"], 2) != round(highest["rate"], 2):
            text += (
                f" Violations per worker ranged from {lowest['rate']:.2f} "
                f"on camera {lowest['camera_id']} to {highest['rate']:.2f} "
                f"on camera {highest['camera_id']}."
            )

        findings.append(text)

    # Best and worst day.
    worst, best = analysis["worst_day"], analysis["best_day"]

    if worst and best and worst["day"] != best["day"]:
        findings.append(
            f"The worst day was {worst['day'].isoformat()} (safety score "
            f"{worst['score']:.1f}, {worst['violations']} violations over "
            f"{worst['workers']} workers); the best was "
            f"{best['day'].isoformat()} (score {best['score']:.1f})."
        )

    # Direction within the period.
    trend = analysis["trend"]

    if trend:
        if trend["direction"] == "flat":
            findings.append(
                "Violations per worker were broadly flat across the period "
                f"(linear fit from {trend['start']:.2f} to "
                f"{trend['end']:.2f})."
            )
        else:
            findings.append(
                f"Violations per worker were {trend['direction']} across "
                f"the period (linear fit from {trend['start']:.2f} to "
                f"{trend['end']:.2f} per worker)."
            )

    return findings
