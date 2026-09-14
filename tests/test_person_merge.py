"""
Unit tests for PPEVisionEngine._merge_persons with synthetic SAHI-like
predictions. No model is loaded: the merge is a classmethod.
"""

from app.processing.cv.ppe_vision_engine import PPEVisionEngine


THRESHOLD = 0.7


class _Box:
    def __init__(self, xyxy):
        self._xyxy = list(xyxy)

    def to_xyxy(self):
        return list(self._xyxy)


class _Score:
    def __init__(self, value):
        self.value = value


class _Mask:
    def __init__(self, segmentation):
        self.segmentation = segmentation


class _Prediction:
    def __init__(self, xyxy, score, polygons=None):
        self.bbox = _Box(xyxy)
        self.score = _Score(score)
        self.mask = _Mask(polygons) if polygons is not None else None


def _rect(x1, y1, x2, y2):
    """Axis-aligned rectangle as one COCO flat polygon."""
    return [[x1, y1, x2, y1, x2, y2, x1, y2]]


def _merge(predictions):
    return PPEVisionEngine._merge_persons(predictions, THRESHOLD)


def test_torso_fragment_inside_full_body_is_absorbed():
    full_body = _Prediction((100, 100, 200, 400), 0.80, _rect(100, 100, 200, 400))
    torso = _Prediction((105, 110, 195, 250), 0.93, _rect(105, 110, 195, 250))

    merged = _merge([torso, full_body])

    assert len(merged) == 1

    bbox, score, polygons = merged[0]
    assert bbox == [100, 100, 200, 400]
    assert score == 0.93
    assert len(polygons) == 1


def test_fragment_chain_collapses_across_passes():
    # Legs and torso do not touch each other; only the full body
    # overlaps both. With legs scoring highest, the first pass merges
    # legs + body, and the torso is only caught on the second pass.
    legs = _Prediction((100, 250, 200, 400), 0.90, _rect(100, 250, 200, 400))
    torso = _Prediction((100, 100, 200, 240), 0.85, _rect(100, 100, 200, 240))
    body = _Prediction((100, 100, 200, 400), 0.80, _rect(100, 100, 200, 400))

    merged = _merge([legs, torso, body])

    assert len(merged) == 1
    assert merged[0][0] == [100, 100, 200, 400]


def test_neighbours_with_overlapping_boxes_but_disjoint_masks_stay_apart():
    # Worker A's silhouette is a triangle whose box is the whole
    # 0..200 square. Worker B is a small square inside that box but
    # outside the triangle. Box overlap-over-smaller would be 1.0;
    # silhouette overlap is 0.
    triangle = [[0, 0, 200, 0, 0, 200]]
    worker_a = _Prediction((0, 0, 200, 200), 0.9, triangle)
    worker_b = _Prediction((150, 150, 190, 190), 0.8, _rect(150, 150, 190, 190))

    merged = _merge([worker_a, worker_b])

    assert len(merged) == 2


def test_box_fallback_when_no_masks():
    # Detect-only weights: no mask, so boxes decide. A fragment box
    # inside the full box has overlap-over-smaller 1.0 (IoU would be
    # 0.35 and never merge).
    full_body = _Prediction((100, 100, 200, 400), 0.80)
    torso = _Prediction((105, 110, 195, 250), 0.93)

    merged = _merge([torso, full_body])

    assert len(merged) == 1
    assert merged[0][0] == [100, 100, 200, 400]
    assert merged[0][2] == []


def test_separate_workers_without_masks_stay_apart():
    left = _Prediction((0, 0, 100, 300), 0.9)
    right = _Prediction((90, 0, 190, 300), 0.8)

    merged = _merge([left, right])

    assert len(merged) == 2


def test_merged_polygons_come_from_union_not_concatenation():
    # Two rectangles overlapping by 75 % of the smaller one union into
    # a single polygon; the result must not contain two overlapping
    # outlines.
    upper = _Prediction((100, 100, 200, 300), 0.9, _rect(100, 100, 200, 300))
    lower = _Prediction((100, 150, 200, 400), 0.8, _rect(100, 150, 200, 400))

    merged = _merge([upper, lower])

    assert len(merged) == 1
    polygons = merged[0][2]
    assert len(polygons) == 1

    xs = polygons[0][0::2]
    ys = polygons[0][1::2]
    assert min(xs) == 100 and max(xs) == 200
    assert min(ys) == 100 and max(ys) == 400
