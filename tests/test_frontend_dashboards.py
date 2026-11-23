"""The pure logic behind the site review and system health Streamlit apps."""

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


def load(name: str, rel: str):
    """Import a module by path (the app folders have hyphens, so they are not packages)."""

    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


rd = load("review_data", "frontend/site-review/review_data.py")
hd = load("health_data", "frontend/system-health/health_data.py")


def person(missing=(), unread=False, ok=True):
    return rd.Person(0, (0.5, 0.5, 0.05, 0.01), (0.4, 0.4, 0.2, 0.2), list(missing), unread, ok, None)


def frame(at, people):
    return rd.Frame("f.jpg", "1", at, 100, 100, people)


def test_sample_results_ingest():
    data = json.loads((ROOT / "frontend/site-review/sample_results.json").read_text())
    review = rd.ingest(data)
    assert review.cams == ["12846", "12875", "12990"]
    assert sum(len(v) for v in review.frames.values()) == 245
    assert review.generated is None and review.generated_raw == "2026-09-30 11:14"
    for frames in review.frames.values():
        assert frames == sorted(frames, key=lambda f: f.at)


def test_person_state_and_safety_score():
    at = datetime(2026, 9, 8, 14, 0, tzinfo=UTC)
    s = rd.stats([frame(at, [person(), person(), person(["helmet"]), person(unread=True)])])
    assert (s.n, s.ok, s.unk, s.known) == (4, 2, 1, 3)
    assert s.miss["helmet"] == 1
    # weights helmet 3, vest 2, boots 1 = 6; 3 x 1 missing / (6 x 3 known)
    assert rd.score(s) == 83.3
    assert rd.score_class(83.3) == "good"
    assert rd.score_class(55) == "warn" and rd.score_class(10) == "crit"
    assert rd.score(rd.stats([])) is None and rd.score_class(None) == ""
    # a red box with nothing missing is still non-compliant; unread with ok stays unknown
    assert person(ok=False).state == "bad" and person(unread=True).state == "unk"


def test_custom_range_covers_whole_site_days_across_dst_end():
    now = datetime(2026, 11, 2, 12, tzinfo=UTC)
    a, b = rd.range_bounds("c", "2026-10-31", "2026-11-01", now, rd.SITE_TZ)
    assert a.astimezone(UTC) == datetime(2026, 10, 31, 4, 0, tzinfo=UTC)  # EDT midnight
    assert b.astimezone(UTC) == datetime(2026, 11, 2, 4, 59, 59, 999000, tzinfo=UTC)  # EST
    a7, b7 = rd.range_bounds("7", "", "", now, rd.SITE_TZ)
    assert rd.day_key(a7, rd.SITE_TZ) == "2026-10-27" and b7 == now


def test_prior_period_and_filters():
    a = datetime(2026, 9, 1, tzinfo=UTC)
    b = datetime(2026, 9, 8, tzinfo=UTC)
    frames = [
        frame(datetime(2026, 8, 30, tzinfo=UTC), [person(["vest"])]),
        frame(datetime(2026, 9, 5, tzinfo=UTC), [person()]),
    ]
    assert rd.in_range(frames, a, b) == frames[1:]
    assert rd.in_range(frames, a, b, prior=True) == frames[:1]
    assert rd.filter_frames(frames, "bad") == frames[:1]
    assert rd.filter_frames(frames, "vest") == frames[:1]
    assert rd.heat_points(frames, "bad") == [(0.525, 0.51)]


def test_camera_status_rules():
    now = datetime(2026, 10, 13, 18, 46, tzinfo=UTC)
    th = hd.thresholds({})

    def cam(minutes_ago=None, failures=0):
        last = None if minutes_ago is None else (now - timedelta(minutes=minutes_ago)).isoformat()
        return {"interval_seconds": 3600, "last_image_at": last, "consecutive_failures": failures}

    assert hd.camera_status(cam(50), th, now)[0] == "healthy"
    assert hd.camera_status(cam(100), th, now)[0] == "delayed"  # > 1.5 x 60 min
    assert hd.camera_status(cam(200), th, now)[0] == "down"  # > 3 x 60 min
    assert hd.camera_status(cam(5, failures=3), th, now)[0] == "down"
    assert hd.camera_status(cam(None), th, now) == ("unknown", None)


def test_sample_health_reads_as_designed():
    data = json.loads((ROOT / "frontend/system-health/sample_health.json").read_text("utf-8"))
    now = hd.clock(data)
    assert now == datetime(2026, 10, 13, 18, 46, tzinfo=UTC)
    counts = hd.status_counts(data["cameras"], hd.thresholds(data), now)
    assert counts == {"healthy": 2, "delayed": 1, "down": 0, "unknown": 0}
    assert hd.scheduler_running(data, now) == (True, 60.0)
    assert len(hd.recent_delivery_failures(data, now)) == 2
    assert hd.stale_age(data, now) is None
    assert hd.span_parts(420) == ("min", 7) and hd.span_parts(9000) == ("h", 2.5)
    assert hd.interval_parts(3600) == ("h", 1) and hd.interval_parts(1200) == ("min", 20)
