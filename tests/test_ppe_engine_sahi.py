"""The camera settings panel reaches the engine: per-camera SAHI settings, reload, merge passes."""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from app.processing.cv import ppe_vision_engine
from app.processing.cv.ppe_vision_engine import PPEVisionEngine


def _bump_mtime(path: Path, after: float | None) -> None:
    stat = path.stat()
    os.utime(path, (stat.st_atime, max(stat.st_mtime, after or 0.0) + 2))


def _write_config(path: Path, cameras: dict, default: dict | None = None) -> None:
    path.write_text(yaml.safe_dump({
        "models": {"person": {"confidence": 0.15}},
        "sahi": {"default": default or {"target_size": 640, "overlap": 0.2}, "cameras": cameras},
    }))


@pytest.fixture()
def engine(tmp_path: Path, monkeypatch) -> PPEVisionEngine:
    config = tmp_path / "vision_config.yaml"
    _write_config(config, {"12846": {"target_size": 640, "person_confidence": 0.34}})
    monkeypatch.setattr(ppe_vision_engine, "VISION_CONFIG_PATH", config)

    # Skip __init__: it loads the real models.
    engine = PPEVisionEngine.__new__(PPEVisionEngine)
    engine._person_confidence = 0.15
    engine._sahi_default, engine._sahi_cameras = engine._sahi_tables(yaml.safe_load(config.read_text()))
    engine._vision_config_mtime = engine._vision_config_mtime_on_disk()
    engine.config_path = config
    return engine


def test_entries_are_completed_from_the_default(engine: PPEVisionEngine) -> None:
    settings = engine._resolve_sahi_settings("12846")

    assert settings["person_confidence"] == 0.34
    assert settings["overlap"] == 0.2 and settings["merge_iou"] == 0.45  # fallback
    assert settings["second_stage_merge_enabled"] is False

    unknown = engine._resolve_sahi_settings("99999")
    assert unknown["target_size"] == 640
    assert unknown["person_confidence"] == 0.15  # the person model's own threshold


def test_saved_settings_apply_without_restart(engine: PPEVisionEngine) -> None:
    _write_config(engine.config_path, {
        "12846": {"target_size": 1200, "person_confidence": 0.41, "second_stage_merge_enabled": True},
        "12999": {"target_size": 800, "person_confidence": 0.5},
    })
    _bump_mtime(engine.config_path, engine._vision_config_mtime)

    engine._reload_sahi_settings_if_changed()

    assert engine._resolve_sahi_settings("12846")["target_size"] == 1200
    assert engine._resolve_sahi_settings("12846")["second_stage_merge_enabled"] is True
    assert engine._resolve_sahi_settings("12999")["person_confidence"] == 0.5


def test_unchanged_or_broken_file_keeps_the_settings(engine: PPEVisionEngine) -> None:
    before = engine._sahi_cameras
    engine._reload_sahi_settings_if_changed()
    assert engine._sahi_cameras is before

    engine.config_path.write_text("sahi: [not: a: mapping")
    _bump_mtime(engine.config_path, engine._vision_config_mtime)
    engine._reload_sahi_settings_if_changed()
    assert engine._sahi_cameras is before


def test_detection_uses_the_camera_overlaps(engine: PPEVisionEngine, monkeypatch) -> None:
    calls = []

    class Result:
        object_prediction_list = []

    def fake_sliced_prediction(image, model, **kwargs):
        calls.append(kwargs)
        return Result()

    monkeypatch.setattr(ppe_vision_engine, "get_sliced_prediction", fake_sliced_prediction)
    engine._person_model = object()

    settings = {
        "target_size": 640, "overlap": 0.2, "overlap_width": 0.3, "overlap_height": 0.1,
        "merge_iou": 0.4, "person_confidence": 0.3,
    }
    assert engine._detect_persons(np.zeros((400, 800, 3), dtype=np.uint8), settings) == []

    kwargs = calls[0]
    assert kwargs["overlap_width_ratio"] == 0.3 and kwargs["overlap_height_ratio"] == 0.1
    # tile stitching keeps its tuned threshold; merge_iou is the engine's own pass
    assert kwargs["postprocess_match_threshold"] == 0.15
    assert kwargs["postprocess_match_metric"] == "IOS"


def test_merge_threshold_pass_uses_iou(engine: PPEVisionEngine) -> None:
    overlapping = [([0, 0, 100, 100], 0.9), ([10, 10, 110, 110], 0.7)]  # IoU 0.68
    apart = [([0, 0, 100, 100], 0.9), ([60, 0, 160, 100], 0.7)]  # IoU 0.25

    assert engine._merge_boxes(overlapping, 0.45) == [([0, 0, 110, 110], 0.9)]
    assert engine._merge_boxes(apart, 0.45) == apart
    assert engine._merge_boxes(apart, 0.2) == [([0, 0, 160, 100], 0.9)]


def test_second_merge_pass_folds_nested_boxes(engine: PPEVisionEngine) -> None:
    persons = [([0, 0, 100, 200], 0.9), ([10, 20, 90, 180], 0.6), ([300, 0, 400, 200], 0.8)]

    merged = engine._merge_boxes(persons, 0.7, metric=engine._ios)

    assert merged == [([0, 0, 100, 200], 0.9), ([300, 0, 400, 200], 0.8)]
    assert engine._ios([0, 0, 100, 200], [10, 20, 90, 180]) == 1.0
    assert engine._ios([0, 0, 100, 200], [300, 0, 400, 200]) == 0.0
    # side by side with little overlap: kept apart
    assert engine._merge_boxes([([0, 0, 100, 100], 0.9), ([90, 0, 190, 100], 0.8)], 0.7, engine._ios) == [
        ([0, 0, 100, 100], 0.9), ([90, 0, 190, 100], 0.8)
    ]
