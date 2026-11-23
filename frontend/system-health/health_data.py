"""
Pure logic of the system health page: the health.json contract and the
rules that turn it into statuses. It mirrors
docs/design/system_health_dashboard.html so both show the same state.
No Streamlit here, so tests import it directly.

health.json (times are UTC ISO-8601): generated_at, mock,
poll_interval_seconds, stale_after_seconds, window_days, thresholds
{delayed_factor, down_factor, down_failures, disk_warn, disk_crit},
scheduler {running, last_successful_cycle_at, last_cycle_ok, last_error,
process_started_at}, cameras [{id, name, interval_seconds,
last_image_at, last_processed_at, consecutive_failures, last_error,
success_24h, frames_7d, expected_7d}], uptime, processing {completed,
failed, pending, oldest_pending_seconds, success_rate,
workers_per_image, p50_ms, p95_ms}, delivery {delivered, failed,
skipped, success_rate, p95_latency_ms, http {422, 429, 5xx, 4xx}},
latency_days [{day, p50_ms, p95_ms, count}], failures [{at, kind,
camera_id, http_status, message}], host {cpu_percent, memory_percent,
disk_percent, ...}, storage {raw_bytes, processed_bytes, database_bytes}.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

UTC = timezone.utc
DEFAULT_THRESHOLDS = {
    "delayed_factor": 1.5,
    "down_factor": 3,
    "down_failures": 3,
    "disk_warn": 85,
    "disk_crit": 95,
}
# status -> (pill class, English label)
PILL = {
    "healthy": ("ok", "Healthy"),
    "delayed": ("warn", "Delayed"),
    "down": ("crit", "Down"),
    "unknown": ("info", "Unknown"),
}


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.astimezone(UTC) if d.tzinfo else d.replace(tzinfo=UTC)


def clock(data: dict) -> datetime:
    """
    Ages are measured from the file's own time for the design sample,
    so it reads the same whenever it is opened; live data uses the real
    clock.
    """

    if data.get("mock"):
        return parse_ts(data.get("generated_at")) or datetime.now(UTC)
    return datetime.now(UTC)


def age_of(value: str | None, now: datetime) -> float | None:
    d = parse_ts(value)
    return (now - d).total_seconds() if d else None


def thresholds(data: dict) -> dict:
    return {**DEFAULT_THRESHOLDS, **{k: v for k, v in (data.get("thresholds") or {}).items()
                                     if v is not None}}


def camera_status(cam: dict, th: dict, now: datetime) -> tuple[str, float | None]:
    """
    Healthy = a new frame within delayed_factor x the camera's interval.
    Delayed = none for longer. Down = none for down_factor x the interval,
    or down_failures failed polls in a row. Unknown = no frame on file.
    Returns (status, age of the last frame in seconds).
    """

    age = age_of(cam.get("last_image_at"), now)
    interval = cam.get("interval_seconds") or 3600

    if (cam.get("consecutive_failures") or 0) >= th["down_failures"]:
        return "down", age
    if age is None:
        return "unknown", None
    if age > interval * th["down_factor"]:
        return "down", age
    if age > interval * th["delayed_factor"]:
        return "delayed", age
    return "healthy", age


def status_counts(cams: list[dict], th: dict, now: datetime) -> dict[str, int]:
    counts = {"healthy": 0, "delayed": 0, "down": 0, "unknown": 0}
    for cam in cams:
        counts[camera_status(cam, th, now)[0]] += 1
    return counts


def scheduler_running(data: dict, now: datetime) -> tuple[bool, float | None]:
    """(running, seconds since the last successful cycle)."""

    s = data.get("scheduler") or {}
    last_ok = age_of(s.get("last_successful_cycle_at"), now)
    stale_after = data.get("stale_after_seconds") or 180
    running = s.get("running") is not False and last_ok is not None and last_ok <= stale_after
    return running, last_ok


def recent_delivery_failures(data: dict, now: datetime) -> list[dict]:
    day_ago = now - timedelta(days=1)
    return [
        f for f in data.get("failures") or []
        if f.get("kind") == "delivery" and (parse_ts(f.get("at")) or day_ago) > day_ago
    ]


def stale_age(data: dict, now: datetime) -> float | None:
    """Seconds by which a live file is older than allowed, or None when fresh (or a sample)."""

    if data.get("mock"):
        return None
    gen = parse_ts(data.get("generated_at"))
    if gen is None:
        return None
    allowed = data.get("stale_after_seconds") or (data.get("poll_interval_seconds") or 60) * 2
    age = (now - gen).total_seconds()
    return age if age > allowed else None


def span_parts(seconds: float | None) -> tuple[str, float] | None:
    """The unit that fits: ('s', 40), ('min', 4), ('h', 2.5), ('d', 3.1)."""

    if seconds is None:
        return None
    seconds = max(0.0, seconds)
    if seconds < 60:
        return "s", round(seconds)
    if seconds < 3600:
        return "min", round(seconds / 60)
    if seconds < 86400:
        return "h", round(seconds / 3600 * 10) / 10
    return "d", round(seconds / 86400 * 10) / 10


def interval_parts(seconds: int) -> tuple[str, float]:
    if seconds >= 3600 and seconds % 3600 == 0:
        return "h", seconds // 3600
    return "min", round(seconds / 60)


def latency_series(data: dict) -> list[tuple[str, float | None, float | None, int]]:
    """(day, p50 s, p95 s, count) for the last window_days days."""

    days = (data.get("latency_days") or [])[-(data.get("window_days") or 7):]
    return [
        (
            d["day"],
            None if d.get("p50_ms") is None else d["p50_ms"] / 1000,
            None if d.get("p95_ms") is None else d["p95_ms"] / 1000,
            d.get("count") or 0,
        )
        for d in days
    ]


def level(value: float | None, warn: float, crit: float) -> str:
    """'' / 'warn' / 'crit' for a percentage against the two thresholds."""

    if value is None:
        return ""
    return "crit" if value >= crit else "warn" if value >= warn else ""


def camera_label(data: dict, cam_id: str | None) -> str:
    if cam_id is None:
        return "–"
    for cam in data.get("cameras") or []:
        if str(cam.get("id")) == str(cam_id):
            return f"{cam['name']} ({cam_id})" if cam.get("name") else str(cam_id)
    return str(cam_id)
