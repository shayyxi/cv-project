# ADR-0001: Tracking people from one frame to the next

| | |
|---|---|
| **Status** | Proposed |
| **Date** | 2026-09-05 |
| **Scope** | `app/processing` (vision + persistence), `app/analytics`, `detections` schema |
| **Deciders** | CV / pipeline team (fill in on acceptance) |

---

## 1. Summary (read this if nothing else)

We want to know whether the person detected in one frame is the same person
as in the next frame, so that we can count real workers, flag the *same*
worker repeatedly without PPE, and stop exporting the same standing worker
seven times to the label queue.

The decisive fact: our cameras are **not video**. They publish a still every
20 to 60 minutes. Every off-the-shelf multi-object tracker (ByteTrack,
BoT-SORT, DeepSORT, StrongSORT, OC-SORT) assumes tens of frames per second
and predicts where a box will be a few milliseconds later. At one frame per
hour a worker can be anywhere on site, or gone. Those trackers would emit a
new identity almost every frame and the output would be noise.

Decision, in one paragraph: **we do not adopt a video tracker.** We add a
persistent `track_id` to detections and a small, pluggable
`PersonAssociator` whose first implementation links a person to the previous
frame of the same camera only when they are still in (almost) the same
place, using position gating plus optimal assignment
(`scipy.optimize.linear_sum_assignment`). We call this what it is,
*same-spot association*, and report unique workers as a bracket
(max concurrent persons in a frame ≤ true headcount ≤ unique tracks ≤
person-observations) rather than a single number we cannot defend.
Appearance-based re-identification is deferred to a gated offline experiment
because of unproven accuracy on uniform PPE and because it creates a new
category of personal data. If the business needs true per-person tracking,
the camera interval has to change first; the schema and interface we add now
let us swap in ByteTrack at that point without touching analytics.

---

## 2. Context: what the pipeline does today

### 2.1 Cameras and frame cadence (measured)

Frames come from three Panomax still cameras. The pipeline polls
`recent_full.jpg` every 60 s and skips frames whose SHA-256 has not changed,
so the effective frame rate is whatever the camera publishes. Measured from
the Panomax day-listing API on 2026-09-05 (`/1.0/cams/{id}/images/day/{date}`):

| Camera | Resolution | Frames/day (2026-09-01 → 09-04) | Interval | Typical hours |
|---|---|---|---|---|
| 12846 | 6960 × 3904 | 11, 12, 9, 7 | **60 min** | 06:20–17:50 |
| 12875 | 6000 × 4000 | 36, 35, 37, 36 | **20 min** | 06:00–18:00 |
| 12990 | 6000 × 4000 | 12, 10, 12, 12 | **60 min** | 06:40–17:50 |

Occasional frames are missing (12990 on 09-02 skipped 07:40 and 12:50), so
the gap between two consecutive *processed* frames can be 2 h or more.

### 2.2 Detection

Per frame, `PPEVisionEngine` ([ppe_vision_engine.py](../../app/processing/cv/ppe_vision_engine.py)):

1. Crops the camera's configured polygon (`ImageCropper`).
2. Runs YOLOv8x **person** detection through SAHI sliced inference
   (`get_sliced_prediction`, ~1000 px tiles) because workers are small in a
   24 MP frame.
3. Merges overlapping tiles' boxes (IoU ≥ 0.45), drops non-person-shaped
   boxes (aspect ≥ 0.8).
4. For each person, takes a fixed-aspect "smart crop" and runs the PPE model
   (helmet / vest / boots), then sets `compliance`.
5. Assigns `person_id = enumerate(persons)` — **the index of the box within
   this frame**, ordered by merge order (confidence-sorted). It restarts at 0
   every frame and has no relation to the previous frame's indices.

### 2.3 Storage and analytics

- `detections` stores one row per person per frame plus one row per PPE item,
  both carrying that per-frame `person_id`.
- `daily_stats` (materialized view) counts person rows as `workers`. That
  column is really **person-observations**: a worker standing in view of
  camera 12875 all day is counted up to 36 times.
- `TrendsService.repeat_noncompliance` flags *camera × PPE class* counts, not
  a repeated person.
- `LabelingService` exports one crop per `(image_job_id, person_id)`, so the
  same stationary worker is exported once per frame.
- `HeatmapService` accumulates foot points; unaffected by identity.

### 2.4 Timestamps

`image_jobs.captured_at` exists but is **never written**
(`IngestionService` only passes `downloaded_at`). The downloader already
receives the HTTP `Last-Modified` header
([http_client.py](../../app/ingestion/http_client.py)) and throws it away.
Any time gap used for association must be capture-to-capture, not
download-to-download (downloads lag by up to 60 s, or by hours if the
service was down).

### 2.5 Privacy posture

Faces are blurred in the processed output (`FaceBlurPrivacyService`); raw
frames are kept on disk. Nothing today identifies an individual across
frames. Storing appearance embeddings would introduce person-level,
biometric-adjacent data that the current privacy posture does not cover.

---

## 3. What cross-frame identity would give us

Ranked by value to the PPE compliance product:

1. **Honest worker counts.** Today's `workers` over-counts by the number of
   frames a person stays in view. Reports and the risk model's `crew`
   feature inherit the inflation.
2. **Per-worker repeat non-compliance.** "The same worker was without a
   helmet in 3 consecutive frames" is a stronger, more actionable signal than
   "camera 12875 logged 3 helmet violations this week".
3. **Fewer false violations.** One missed helmet detection flips a person to
   non-compliant. With a track, compliance can be reported as a majority
   over the last *k* observations instead of a single frame.
4. **Violation duration.** How long a non-compliant state persisted
   (bounded below by the frame interval).
5. **Label-queue de-duplication.** One crop per worker-violation per day
   instead of one per frame.

Items 1, 2 and 5 are achievable with same-spot association for stationary
workers. Items 3 and 4 need reliable identity and are only partially
achievable at the current cadence.

---

## 4. The constraint that drives everything: the inter-frame gap

Video trackers work because the world barely changes between frames:

| | Video (25 fps) | Camera 12875 (20 min) | Cameras 12846 / 12990 (60 min) |
|---|---|---|---|
| Δt between frames | 0.04 s | 1 200 s | 3 600 s |
| Walking worker (1.3 m/s) moves | 5 cm | 1.5 km | 4.7 km |
| Same worker's box IoU, frame to frame | ≈ 0.9 | ≈ 0 unless stationary | ≈ 0 unless stationary |
| Constant-velocity Kalman prediction | accurate | meaningless | meaningless |
| Lighting / shadows | identical | shifted | strongly shifted |

Consequences:

- **Motion cues are useless.** IoU or Kalman-predicted position can only ever
  link a worker who has *not* moved more than about one body width. That is
  machine operators in a cab and workers at a fixed station (formwork, rebar,
  a saw bench), and nobody else.
- **Appearance is the only cue that survives the gap**, and it is weak here:
  every worker wears the same hi-vis vest and helmet; the very events we
  detect (helmet off, vest off) change the appearance; shadows move an hour
  at a time. With no labelled identity data on these sites we cannot even
  measure how often it would be wrong.
- Whatever we build, an "identity" that spans a 60 min gap is a guess. The
  design must be explicit about that in naming and in reported metrics.

---

## 5. Options considered

### Option A — No cross-frame identity; fix the vocabulary

Keep detections independent. Rename/document `workers` as
*person-observations*; add `max_concurrent` (largest person count in any one
frame of the day) as the defensible lower bound on headcount; keep compliance
*rates*, which do not suffer from double counting.

- ✅ Zero cost, zero privacy change, honest.
- ❌ No per-person repeat flags, no de-duplication, no duration.

### Option B — Same-spot association (position gating + optimal assignment)

Link each person to at most one person in the previous processed frame of
the same camera when (a) the capture gap is below a limit, (b) the foot
points are within *n* body heights, (c) box heights are similar. Solve the
many-to-many choice with the Hungarian algorithm so that two neighbours who
both shift slightly are not cross-linked.

- ✅ ~150 lines, deterministic, unit-testable with synthetic boxes, no new
  model, no weights, no GPU, no personal data beyond a random UUID.
- ✅ Correctly links the population we *can* link (stationary workers), which
  is also where repeat violations by one person are most plausible.
- ❌ Splits a track whenever the worker walks away: unique tracks
  over-count true persons. Must be reported as an upper bound.
- ❌ Two different people occupying the same station an hour apart will be
  linked. Mitigated by a tight distance gate and the height-ratio gate;
  cannot be eliminated without appearance.

### Option C — Appearance re-identification (ReID embeddings)

Embed each smart crop with a person-ReID network (OSNet via `torchreid`, or
the ReID module of `boxmot`), keep a per-camera gallery for the day, match by
cosine similarity with a position prior, Hungarian assignment.

- ✅ The only approach that can follow a *moving* worker across an hour.
- ❌ Accuracy on uniform PPE is unknown and likely poor; helmet/vest changes
  hurt; no ground truth exists to validate.
- ❌ New heavy dependency and weights; CPU-only host (fine for a few crops an
  hour, but adds latency to every frame).
- ❌ **New personal-data category.** Embeddings are a re-identification
  template. Requires a privacy assessment, a retention rule (in-memory for
  the working day at most) and documentation for site workers, before any
  code ships.

### Option D — Off-the-shelf video tracker (ByteTrack / BoT-SORT)

`supervision.ByteTrack` (v0.24.0 is already installed) or ultralytics'
built-in `model.track()`.

- ✅ Mature, one import, well-understood.
- ❌ Both assume video: ByteTrack's constructor takes `frame_rate=30` and
  `lost_track_buffer=30` **frames** (that is 30 hours here), and its
  association is Kalman-predicted IoU. At our cadence it would create a new
  ID nearly every frame.
- ❌ `model.track()` is not reachable anyway: our person model runs through
  SAHI's `get_sliced_prediction`, not `model.predict`/`model.track`. We would
  have to feed merged boxes to a standalone tracker, at which point we are
  writing the association ourselves regardless.
- Verdict: **not now**. Becomes the right choice if the camera interval
  drops to a few seconds or the feed becomes video.

### Option E — Change the input (business decision)

Ask Panomax for a shorter interval on the compliance cameras (they already
run 12875 at 20 min; 1–2 min is a product option), or add an RTSP camera.

- ✅ Prerequisite for anything that deserves the name "tracking".
- ❌ Bandwidth/storage cost, contract change, not a software decision.

---

## 6. Decision

1. **Do not adopt a video multi-object tracker at the current cadence**
   (rejects Option D as an implementation, keeps it as the future swap-in).
2. **Add `track_id` (UUID string, nullable) to `detections`**, written on the
   person row and on its PPE rows, plus an index. Migration:
   `add_track_id_to_detections`. Add `track_id: str | None = None` to
   `VisionDetectionDTO`.
3. **Introduce a `PersonAssociator` interface** in
   `app/processing/tracking/` and ship **`PositionAssociator`** (Option B) as
   its first implementation, called from `ProcessingService` after
   `translate_result_to_original` and before rendering/persistence. Name it
   *same-spot association* in config, code and docs, never "tracking".
4. **Populate `image_jobs.captured_at`** from the download's `Last-Modified`
   header (fallback: `downloaded_at`). Association gates on capture time.
   This is a prerequisite and a bug fix in its own right.
5. **Change analytics semantics** (Option A's vocabulary) once `track_id`
   exists: keep `workers` but document it as observations, add
   `unique_tracks` and `max_concurrent` to `daily_stats`, add a per-track
   variant of repeat non-compliance, and de-duplicate the label queue by
   `(track_id, day, violated class)`.
6. **Defer appearance ReID (Option C)** to a time-boxed offline experiment on
   already-stored raw frames, with the go/no-go criteria and privacy
   conditions in §9. No embedding is persisted anywhere in the meantime.
7. **Record the dependency on Option E**: if stakeholders require true
   per-person tracking (violation duration, moving workers), the camera
   interval must change first. The interface from (3) then takes a
   ByteTrack-backed implementation with no schema or analytics change.

---

## 7. How the chosen approach works

### 7.1 Data flow

```
IngestionService ──▶ image_jobs (captured_at from Last-Modified)   ← D4
        │
ProcessingService.process_next()   (frames per camera processed in
        │                            downloaded_at order, so "previous
        │                            frame" is well defined)
        ├─ ImageCropper.crop
        ├─ PPEVisionEngine.process_image           person_id = 0..N-1 (per frame)
        ├─ ImageCropper.translate_result_to_original
        ├─ PersonAssociator.assign(camera_id, captured_at, persons)   ← NEW
        │        sets person.track_id for every person
        ├─ VisionRenderer.draw_original              (may show short track tag)
        ├─ PrivacyService.apply_privacy_blur
        └─ DetectionRepository.create_many          writes track_id on person + PPE rows
```

### 7.2 Association rule (`PositionAssociator`)

For camera *c*, let *P* be the persons of the previous processed frame
(with their `track_id`s) and *Q* the persons of the current frame.

1. **Gap gate.** If `captured_at(Q) − captured_at(P) > max_gap_minutes`, or
   there is no previous frame, every person in *Q* gets a fresh `uuid4()`.
2. **Cost matrix.** For each pair (p, q):
   - foot point `f = ((x_min + x_max) / 2, y_max)` — where the worker stands;
     robust to arm/tool changes in the box.
   - scale `h = (height(p) + height(q)) / 2` — normalising by box height
     makes the gate perspective-aware (far workers are small and move fewer
     pixels).
   - **height-ratio gate:** if `max(h_p, h_q) / min(h_p, h_q) > max_height_ratio`
     the pair is impossible (a near and a far worker).
   - `cost = ‖f_p − f_q‖ / h`; if `cost > max_move_body_heights` the pair is
     impossible.
3. **Optimal assignment.** `scipy.optimize.linear_sum_assignment` on the
   matrix with impossible cells set to a large constant; discard assignments
   that landed on an impossible cell.
4. Matched *q* inherit the *p*'s `track_id`; unmatched *q* get a new
   `uuid4()`; unmatched *p* simply end (there is no "lost" buffer — at this
   cadence a person absent from one frame has been gone for up to an hour).
5. Remember *Q* as the new previous frame for camera *c*.

Complexity is O(|P|·|Q|) with |P|, |Q| ≲ 20, i.e. negligible next to
inference.

### 7.3 Sketch

```python
# app/processing/tracking/position_associator.py  (proposed)
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np
from scipy.optimize import linear_sum_assignment

IMPOSSIBLE = 1e6


@dataclass
class FrameState:
    captured_at: datetime
    persons: list  # VisionDetectionDTO with track_id set


class PositionAssociator(PersonAssociator):
    def __init__(self, repository, max_gap_minutes=75,
                 max_move_body_heights=1.5, max_height_ratio=1.5):
        self._repository = repository          # to reload last frame after restart
        self._max_gap = timedelta(minutes=max_gap_minutes)
        self._max_move = max_move_body_heights
        self._max_ratio = max_height_ratio
        self._last: dict[str, FrameState] = {}

    def assign(self, camera_id, captured_at, persons):
        prev = self._last.get(camera_id) or self._load_previous(camera_id)

        if prev is None or captured_at - prev.captured_at > self._max_gap:
            for p in persons:
                p.track_id = str(uuid4())
        else:
            cost = self._cost_matrix(prev.persons, persons)
            rows, cols = linear_sum_assignment(cost)
            matched = set()
            for r, c in zip(rows, cols):
                if cost[r, c] < IMPOSSIBLE:
                    persons[c].track_id = prev.persons[r].track_id
                    matched.add(c)
            for i, p in enumerate(persons):
                if i not in matched:
                    p.track_id = str(uuid4())

        self._last[camera_id] = FrameState(captured_at, persons)

    def _cost_matrix(self, prev, cur):
        cost = np.full((len(prev), len(cur)), IMPOSSIBLE)
        for i, a in enumerate(prev):
            for j, b in enumerate(cur):
                ha, hb = _height(a.box), _height(b.box)
                if max(ha, hb) / max(min(ha, hb), 1) > self._max_ratio:
                    continue
                d = _dist(_foot(a.box), _foot(b.box)) / ((ha + hb) / 2)
                if d <= self._max_move:
                    cost[i, j] = d
        return cost
```

`_load_previous` fetches the latest PROCESSED job for the camera and its
person rows (with `track_id`) so a restart does not break every track. A
`DummyAssociator` that assigns fresh UUIDs keeps tests and the
`DummyVisionEngine` path trivial.

### 7.4 Configuration (`vision_config.yaml`)

```yaml
tracking:
  enabled: true
  # Frames further apart than this never link (covers one missed hourly frame).
  max_gap_minutes: 75
  # Foot-point displacement allowed, in body heights (~2.5 m for a 1.7 m worker).
  max_move_body_heights: 1.5
  # Near/far confusion guard.
  max_height_ratio: 1.5
  cameras:
    "12875":            # 20-minute camera: tighter gap, same movement gate
      max_gap_minutes: 25
```

Defaults are starting points; tune after the replay in §11.

### 7.5 Schema and analytics changes

- `detections.track_id VARCHAR(36) NULL`, index on `(track_id)`.
- `daily_stats` / `weekly_stats` (materialized views, drop and recreate in the
  migration):
  - keep `workers` (observations) for backward compatibility;
  - add `unique_tracks = COUNT(DISTINCT d.track_id)`;
  - add `max_concurrent = MAX(persons per image_job)` via a subquery.
- Report the bracket: `max_concurrent ≤ headcount ≤ unique_tracks ≤ workers`.
  The PDF should print observations and the bracket, not a single
  "workers" number.
- `TrendsService.repeat_noncompliance_by_track`: tracks with ≥ N
  non-compliant observations for the same class inside the window, returning
  `track_id, camera_id, class, count, first_seen, last_seen`. Keep the
  existing camera-level flag.
- `LabelingService`: export at most one crop per `(track_id, day, class)`,
  choosing the highest-confidence person box.
- Optional, later: per-track compliance smoothing (majority over the last
  *k* observations) **as a reporting layer only**; per-frame flags stay as
  detected.

### 7.6 Renderer

Replace `P{person_id}` with a short tag such as `T-3f9a` (last four hex
characters of the track id). Same-spot links then become visible in the
processed images, which is also the cheapest way to eyeball wrong links.

---

## 8. Why this approach

- **It matches the physics of the input.** Position is the only cue we have
  that is both cheap and *correct when it fires*. A worker whose foot point
  is within a couple of metres of where a worker stood 20–60 minutes ago,
  with the same apparent height, is very probably the same worker at the
  same station. Beyond that radius we genuinely do not know, and the method
  says so by starting a new track.
- **It is honest in what it claims.** Naming it *same-spot association* and
  reporting a bracket avoids the trap of shipping a "unique workers" number
  that stakeholders will treat as headcount.
- **It creates the plumbing once.** `track_id` on rows, an associator seam in
  `ProcessingService`, per-track analytics. Whatever later replaces the
  matcher (ReID, ByteTrack on a faster feed) drops into the same seam.
- **It is testable without a GPU or a model.** Synthetic boxes cover every
  branch: stationary → same id; moved beyond gate → new id; gap too long →
  new id; two neighbours shifting → Hungarian keeps them straight; restart →
  reload from DB.
- **It adds no personal data.** A random UUID linking two boxes carries no
  appearance information.
- **Hungarian instead of greedy** because the failure mode that matters
  (two workers a few metres apart both drift) is exactly where greedy
  nearest-neighbour cross-links and optimal assignment does not. The cost is
  one function call.

---

## 9. Why these libraries (and why not the others)

| Need | Chosen | Why | Rejected alternatives and why |
|---|---|---|---|
| Optimal one-to-one matching | `scipy.optimize.linear_sum_assignment` (scipy 1.14.1, already installed as a scikit-learn dependency) | Exact Hungarian in one call; deterministic; no native build step; handles rectangular matrices | `lap` / `lapx` — extra compiled dependency for the same result at a scale where speed is irrelevant. Greedy nearest neighbour — cross-links neighbours (see §8). |
| Geometry (foot points, heights, distances) | `numpy` | Already a core dependency; vectorises trivially | `shapely` — polygon library, overkill for boxes. |
| Tracker framework | **None** for the current cadence | Video assumptions (Kalman, per-frame buffers) are violated at 20–60 min; see §4 and Option D | `supervision.ByteTrack` (0.24.0, installed) — reserved as the drop-in when the interval becomes video-like; `frame_rate`/`lost_track_buffer` are frame-based. ultralytics `model.track()` — unreachable because the person model runs through SAHI sliced inference, and same video assumptions. `boxmot` (BoT-SORT, StrongSORT, DeepOCSORT) — heavy dependency and weights, Kalman again, and its ReID part triggers the §9-privacy conditions. |
| Appearance re-identification | **Deferred** (candidate: OSNet via `torchreid`, or `boxmot`'s ReID module) | Unvalidated on uniform PPE, CPU-only host, new personal-data category | Not chosen until the experiment in §10 passes and privacy signs off. |
| Identity type | `uuid4()` as `String(36)` | Matches every other id in the schema; globally unique across cameras and days; no sequence contention; safe to generate in the processing step | Integer sequence — collisions across cameras, needs DB round trips or a central counter. |
| Persistence | Existing SQLAlchemy models + Alembic migration | Same mechanism as `person_id` and compliance columns were added | A separate `tracks` table — adds a join for no query we need today; revisit if tracks gain attributes (first/last seen are derivable). |

---

## 10. Deferred: appearance re-identification experiment

Only worth running if stakeholders confirm they need moving-worker identity
at the current camera interval.

**Data.** Five working days of raw frames already stored for camera 12875
(≈ 180 frames at 20 min) plus one hourly camera. Manually label person
identities across frames for that subset (an afternoon's work with the
processed images as a guide).

**Candidates.** (i) same-spot association alone (baseline), (ii) ReID-only
(cosine ≥ τ on OSNet embeddings), (iii) fused (ReID with the position prior
as a soft cost).

**Metrics.** IDF1 and ID switches per frame, per camera. Also count the
false links that matter most: two different people merged into one track.

**Go criteria (proposal).** Fused IDF1 ≥ 0.75 on the 20 min camera *and*
ID switches at least 30 % lower than the baseline. Below that, the extra
model and the privacy cost buy nothing over Option B.

**Privacy conditions, all mandatory before shipping.** Embeddings live in
process memory for the working day and are never written to the database,
disk or logs; the associator only stores the resulting `track_id`; a privacy
assessment is completed and worker information (site signage / notices)
covers re-identification; a retention statement is added to this ADR.

---

## 11. Validation and test plan

1. **Unit tests** (`tests/test_position_associator.py`, synthetic boxes, no
   model): stationary → same id; moved > gate → new id; Δt > max gap → new
   id; near/far height-ratio guard; two neighbours shifting → no cross-link;
   restart reload from repository; empty previous / empty current frame.
2. **Repository test**: `create_many` writes `track_id` on the person row and
   all its PPE rows.
3. **Offline replay** with `scripts/run_folder.py` on a stored day per
   camera. Note that `run_folder.py` stamps `downloaded_at = now` for every
   file; for replay the associator must take `captured_at` parsed from the
   filename (`{camera}_{YYYY-MM-DD}T{HH-MM-SS}Z.jpg`) so gap gating works.
   Inspect processed images for wrong links; tune the three thresholds per
   camera.
4. **Analytics check**: for the replayed day, verify
   `max_concurrent ≤ unique_tracks ≤ workers` and that the label queue
   exports one crop per track-violation.

---

## 12. Consequences

**Positive**

- Worker counts become defensible (a bracket instead of an inflated count).
- Repeat non-compliance can point at a specific stationary worker and time
  span, not just a camera.
- Label queue shrinks by roughly the number of frames a stationary worker
  stays in view.
- `captured_at` is finally populated, fixing the timestamp basis for every
  analytics query that already prefers it.
- Future trackers plug into an existing seam.

**Negative / risks**

- Tracks split whenever workers move, so `unique_tracks` over-counts. Must
  be communicated as an upper bound in reports and the PDF.
- Two people at the same station an hour apart can be merged. Tight gates
  reduce but cannot remove this; per-track repeat flags should therefore
  require ≥ 3 observations, not 2.
- Three new thresholds to tune per camera; defaults may need adjustment
  after the first replay.
- A migration that recreates the materialized views; downtime is seconds,
  but it must run before the new code deploys.

---

## 13. Revisit this decision when

- Panomax changes any camera's interval (especially to ≤ 1 min), or a video
  camera is added → evaluate Option D (`supervision.ByteTrack`) behind the
  same interface.
- Stakeholders require violation duration or moving-worker identity → run
  the §10 experiment and the privacy assessment.
- A camera is moved or re-aimed → the distance gate, being body-height
  normalised, should survive, but re-run the replay in §11.
- The §10 experiment reports its numbers, whichever way they fall.

---

## 14. Open questions for the deciders

1. Can the Panomax interval on 12846 and 12990 be shortened to match 12875
   (20 min) or lower? What does it cost?
2. Who owns the privacy assessment if ReID is pursued, and what jurisdiction
   applies to the sites (the cameras appear to be in Québec; the provider is
   in the EU)?
3. In the PDF report, do we show the bracket, or observations plus
   `max_concurrent` only?
