"""
Pure logic of the site safety review: the results.json contract, date
ranges in site time, compliance stats and the safety score. It mirrors
docs/design/ppe-review-dashboard.html so both show the same figures.
No Streamlit here, so tests import it directly.

results.json: {cams: [id], generated, frames: [{c: cam, t: UTC time,
f: file, w, h, p: [{i: person id, c: conf | null, ok: 1 | 0,
h, v, b: true | false | null, x, y, bw, bh: label box in px}]}]}
Optional: imgBase, imgAlt (folders with the processed frames), preview
(folder with downscaled copies under the same names).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

SITE = "Gallion"
SITE_TZ = ZoneInfo("America/Toronto")
UTC = timezone.utc
CAM_NAMES = {"12846": "Gallion Cam 1", "12875": "Gallion Cam 2", "12990": "Gallion Cam 3"}

# key, kind, enabled. Scene checks are in the contract but switched off.
CHECKS = (
    ("helmet", "person", True),
    ("vest", "person", True),
    ("boots", "person", True),
    ("harness", "person", False),
    ("guardrail", "scene", False),
    ("rebar", "scene", False),
)
CHECK_LABELS = {
    "helmet": "Helmet",
    "vest": "Safety vest",
    "boots": "Safety boots",
    "harness": "Harness",
    "guardrail": "Guard rails",
    "rebar": "Rebar protection",
}
# Safety score weights, as on the camera dashboard.
WEIGHTS = {"helmet": 3, "vest": 2, "boots": 1, "harness": 2}
ROLES = ("admin", "group", "single")

MONTHS = {
    # en-GB short months, as the design page prints them ("29 Sept")
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"],
    "fr": [
        "janv.", "févr.", "mars", "avr.", "mai", "juin",
        "juil.", "août", "sept.", "oct.", "nov.", "déc.",
    ],
}
WEEKDAYS = {
    "en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "fr": ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."],
}


def enabled(kind: str | None = None) -> list[str]:
    return [k for k, t, on in CHECKS if on and (kind is None or t == kind)]


def camera_name(cam_id: str) -> str:
    return CAM_NAMES.get(cam_id, f"Cam {cam_id}")


@dataclass
class Person:
    no: int
    lbl: tuple[float, float, float, float]  # label box, fractions of the frame
    bbox: tuple[float, float, float, float]  # region around and below it (thumbnail, close-up)
    missing: list[str]
    unread: bool
    ok: bool
    conf: float | None

    @property
    def missing_enabled(self) -> list[str]:
        on = enabled()
        return [k for k in self.missing if k in on]

    @property
    def state(self) -> str:
        """
        'bad' = a missing item or the pipeline's red box; 'unk' = label
        unreadable, nothing to judge (kept out of every compliance
        figure); 'ok' = fully compliant.
        """

        if self.missing_enabled or not self.ok:
            return "bad"
        return "unk" if self.unread else "ok"


@dataclass
class Frame:
    file: str
    cam: str
    at: datetime
    w: int
    h: int
    people: list[Person]

    @property
    def bad(self) -> bool:
        return any(p.state == "bad" for p in self.people)

    def has(self, key: str) -> bool:
        return any(key in p.missing for p in self.people)


@dataclass
class Review:
    cams: list[str]
    frames: dict[str, list[Frame]]
    generated: datetime | None = None
    generated_raw: str = ""
    img_dirs: list[str] = field(default_factory=list)
    preview: str = ""

    def all_frames(self) -> list[Frame]:
        return [f for frames in self.frames.values() for f in frames]

    @property
    def first(self) -> datetime | None:
        frames = self.all_frames()
        return min(f.at for f in frames) if frames else None

    @property
    def last(self) -> datetime | None:
        frames = self.all_frames()
        return max(f.at for f in frames) if frames else None


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def ingest(data: dict | None) -> Review:
    """frames -> images per camera, oldest first."""

    data = data or {}
    frames: dict[str, list[Frame]] = {}

    for raw in data.get("frames") or []:
        w, h = raw["w"], raw["h"]
        people = []

        for j, p in enumerate(raw.get("p") or []):
            missing = [k for k, flag in (("helmet", "h"), ("vest", "v"), ("boots", "b"))
                       if p.get(flag) is False]
            unread = p.get("h") is None and p.get("v") is None and p.get("b") is None
            # bbox: a region around and below the label, since the data
            # holds no body box.
            rw, rh = max(6 * p["bh"], p["bw"]), 8 * p["bh"]
            cx = p["x"] + p["bw"] / 2
            x0, y0 = max(0, cx - rw / 2), max(0, p["y"] - p["bh"])
            x1, y1 = min(w, x0 + rw), min(h, y0 + rh)
            people.append(Person(
                no=j if p.get("i") is None else p["i"],
                lbl=(p["x"] / w, p["y"] / h, p["bw"] / w, p["bh"] / h),
                bbox=(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h),
                missing=missing,
                unread=unread,
                ok=bool(p.get("ok")),
                conf=None if p.get("c") is None else float(p["c"]),
            ))

        cam = str(raw["c"])
        frames.setdefault(cam, []).append(
            Frame(file=raw["f"], cam=cam, at=parse_utc(raw["t"]), w=w, h=h, people=people)
        )

    for lst in frames.values():
        lst.sort(key=lambda f: f.at)

    cams = [str(c) for c in (data.get("cams") or list(frames))]
    review = Review(cams=cams, frames=frames)

    img_dirs = [data.get("imgBase") or "processed/"]
    if data.get("imgAlt"):
        img_dirs.append(data["imgAlt"])
    review.img_dirs = img_dirs
    review.preview = data.get("preview") or ""

    gen = data.get("generated")
    if gen:
        try:
            parsed = datetime.fromisoformat(str(gen).replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None and parsed.tzinfo is not None:
            review.generated = parsed.astimezone(UTC)
        else:
            review.generated_raw = str(gen).replace("T", " ")[:16]

    return review


def scope(cams: list[str], role: str) -> list[str]:
    """The cameras a role sees: admin all, group its site (one site here), single the first."""

    if role == "single":
        return cams[:1]
    return list(cams)


# ---- time: the site's local time by default, UTC behind the switch. Calendar days are
# handled as Y-M-D strings and turned into instants with day_start / day_end, so a DST
# change never shifts a bucket.


def tz_for(utc: bool) -> tzinfo:
    return UTC if utc else SITE_TZ


def day_key(d: datetime, tz: tzinfo) -> str:
    return d.astimezone(tz).strftime("%Y-%m-%d")


def hour_of(d: datetime, tz: tzinfo) -> int:
    return d.astimezone(tz).hour


def valid_day(key: str | None) -> bool:
    try:
        date.fromisoformat(key or "")
    except ValueError:
        return False
    return True


def shift_day(key: str, n: int) -> str:
    return (date.fromisoformat(key) + timedelta(days=n)).isoformat()


def day_start(key: str, tz: tzinfo) -> datetime:
    return datetime.combine(date.fromisoformat(key), time.min, tzinfo=tz)


def day_end(key: str, tz: tzinfo) -> datetime:
    return day_start(shift_day(key, 1), tz) - timedelta(milliseconds=1)


def range_bounds(
    preset: str, from_key: str, to_key: str, now: datetime, tz: tzinfo
) -> tuple[datetime, datetime]:
    """Presets end now (site time); a custom range covers whole days."""

    if preset == "c" and valid_day(from_key) and valid_day(to_key):
        return day_start(from_key, tz), day_end(to_key, tz)

    n = 7 if preset == "c" else int(preset)
    today = day_key(now, tz)
    return day_start(shift_day(today, -(n - 1)), tz), now


def in_range(
    frames: list[Frame], a: datetime, b: datetime, prior: bool = False
) -> list[Frame]:
    """Frames in (a, b]; prior=True gives the same length of time just before it."""

    if prior:
        a, b = a - (b - a), a
    return [f for f in frames if a < f.at <= b]


@dataclass
class Stats:
    frames: list[Frame]
    n: int = 0
    ok: int = 0
    unk: int = 0
    miss: dict[str, int] = field(default_factory=dict)

    @property
    def known(self) -> int:
        """People whose label could be read; every rate uses it."""

        return self.n - self.unk

    @property
    def rate(self) -> float | None:
        return self.ok / self.known if self.known else None


def stats(frames: list[Frame]) -> Stats:
    s = Stats(frames, miss={k: 0 for k, _, _ in CHECKS})

    for f in frames:
        for p in f.people:
            s.n += 1
            state = p.state
            if state == "ok":
                s.ok += 1
            elif state == "unk":
                s.unk += 1
            for k in p.missing_enabled:
                s.miss[k] += 1

    return s


def score(s: Stats) -> float | None:
    """
    100 x (1 - sum(weight x missing) / (sum(weights) x people with a
    readable label)); None when nobody was seen.
    """

    keys = enabled("person")
    total_weight = sum(WEIGHTS.get(k, 1) for k in keys)
    if not s.known or not total_weight:
        return None
    missed = sum(WEIGHTS.get(k, 1) * s.miss[k] for k in keys)
    return round(1000 * (1 - missed / (total_weight * s.known))) / 10


def score_class(value: float | None) -> str:
    if value is None:
        return ""
    return "good" if value >= 70 else "warn" if value >= 40 else "crit"


def top_missing(s: Stats) -> str | None:
    """The enabled item missed most often, or None when nothing was missed."""

    keys = enabled()
    best = max(keys, key=lambda k: s.miss[k]) if keys else None
    return best if best and s.miss[best] else None


def trend(
    s: Stats, a: datetime, b: datetime, tz: tzinfo
) -> tuple[list[str], list[float | None], bool]:
    """
    Compliance per bucket: every day of the range (None when nobody was
    seen), or every hour for a range of a day or less. Returns (keys,
    values, hourly).
    """

    hourly = (b - a) <= timedelta(hours=25)
    keys: list[str] = []

    if hourly:
        hours = [hour_of(f.at, tz) for f in s.frames if f.people]
        lo, hi = (min(hours), max(hours)) if hours else (0, 23)
        keys = [str(h) for h in range(lo, hi + 1)]
    else:
        k, end = day_key(a, tz), day_key(b, tz)
        while k <= end and len(keys) < 3000:
            keys.append(k)
            k = shift_day(k, 1)

    bucket = {k: [0, 0] for k in keys}

    for f in s.frames:
        k = str(hour_of(f.at, tz)) if hourly else day_key(f.at, tz)
        bucket.setdefault(k, [0, 0])
        for p in f.people:
            state = p.state
            if state == "unk":
                continue
            bucket[k][0] += 1
            if state == "ok":
                bucket[k][1] += 1

    values = [bucket[k][1] / bucket[k][0] if bucket[k][0] else None for k in keys]
    return keys, values, hourly


def hour_bars(s: Stats, all_frames: list[Frame], tz: tzinfo) -> list[tuple[int, int, int]]:
    """
    (hour, non-compliant, people with a readable label) for the hours at
    which this camera has ever seen people, so the chart fits the site's
    working day.
    """

    hours = {hour_of(f.at, tz) for f in all_frames if f.people}
    lo, hi = (min(hours), max(hours)) if hours else (7, 18)
    per_hour = {h: [0, 0] for h in range(lo, hi + 1)}

    for f in s.frames:
        h = hour_of(f.at, tz)
        per_hour.setdefault(h, [0, 0])
        for p in f.people:
            state = p.state
            if state == "unk":
                continue
            per_hour[h][0] += 1
            if state != "ok":
                per_hour[h][1] += 1

    return [(h, per_hour[h][1], per_hour[h][0]) for h in range(lo, hi + 1)]


def worst_hour(bars: list[tuple[int, int, int]]) -> tuple[int, float, int, int] | None:
    """(hour, share, non-compliant, people) of the worst hour with at least 3 people."""

    worst = None
    for h, bad, total in bars:
        share = bad / total if total else 0
        if total >= 3 and (worst is None or share > worst[1]):
            worst = (h, share, bad, total)
    return worst if worst and worst[1] > 0 else None


def worst_day(
    s: Stats, keys: list[str], values: list[float | None], tz: tzinfo
) -> tuple[str, float] | None:
    """(day key, compliance) of the day with the lowest compliance and at least 3 people."""

    counts: dict[str, int] = {}
    for f in s.frames:
        k = day_key(f.at, tz)
        counts[k] = counts.get(k, 0) + sum(1 for p in f.people if p.state != "unk")

    worst = None
    for k, v in zip(keys, values):
        if v is not None and counts.get(k, 0) >= 3 and (worst is None or v < worst[1]):
            worst = (k, v)
    return worst


def detector_note(s: Stats, others: list[Stats]) -> tuple[str, float, float] | None:
    """
    One item flagged on nearly every worker of this camera, far above
    the other cameras, is more likely a detector or framing problem than
    a site problem. Returns (item, own share, others' share) or None.
    """

    if s.known < 20:
        return None
    others = [o for o in others if o.known >= 20]
    if not others:
        return None

    for k in enabled("person"):
        own = s.miss[k] / s.known
        other = sum(o.miss[k] / o.known for o in others) / len(others)
        if own >= 0.95 and own - other >= 0.3:
            return k, own, other

    return None


def filter_frames(frames: list[Frame], key: str) -> list[Frame]:
    if key == "all":
        return list(frames)
    if key == "bad":
        return [f for f in frames if f.bad]
    return [f for f in frames if f.has(key)]


def heat_points(frames: list[Frame], key: str) -> list[tuple[float, float]]:
    """Where the chosen violation happens: the foot of each label, as fractions of the frame."""

    points = []
    for f in frames:
        for p in f.people:
            if (p.state == "bad") if key == "bad" else (key in p.missing):
                x, y, w, h = p.lbl
                points.append((x + w / 2, y + h))
    return points


# ---- formatting


def fd(d: datetime, tz: tzinfo, lang: str) -> str:
    d = d.astimezone(tz)
    return f"{d.day} {MONTHS[lang][d.month - 1]}"


def fmt(d: datetime, tz: tzinfo, lang: str) -> str:
    d = d.astimezone(tz)
    return f"{WEEKDAYS[lang][d.weekday()]} {d.day} {MONTHS[lang][d.month - 1]}, {d:%H:%M}"


def tm(d: datetime, tz: tzinfo) -> str:
    return d.astimezone(tz).strftime("%H:%M:%S")


def pct(value: float | None, lang: str) -> str:
    if value is None:
        return "–"
    n = round(value * 100)
    return f"{n} %" if lang == "fr" else f"{n}%"


def nf1(value: float, lang: str) -> str:
    text = f"{value:.1f}"
    return text.replace(".", ",") if lang == "fr" else text


def nf2(value: float, lang: str) -> str:
    text = f"{value:.2f}"
    return text.replace(".", ",") if lang == "fr" else text
