import os
from pathlib import Path

import pytest

from app.processing.cv import ppe_vision_engine
from app.processing.cv.ppe_vision_engine import PPEVisionEngine


def _bump_mtime(path: Path, after: float | None) -> None:
    """Make the change visible even within the same filesystem tick."""

    stat = path.stat()
    os.utime(path, (stat.st_atime, max(stat.st_mtime, after or 0.0) + 2))


class FakeModel:
    def __init__(self, path: str) -> None:
        self.path = path
        self.device = None

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self


@pytest.fixture()
def engine(tmp_path: Path, monkeypatch) -> PPEVisionEngine:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"deployed weights")

    loads: list[str] = []

    def fake_yolo(path):
        loads.append(path)
        return FakeModel(path)

    monkeypatch.setattr(ppe_vision_engine, "YOLO", fake_yolo)

    # Skip __init__: it loads the real person and PPE models.
    engine = PPEVisionEngine.__new__(PPEVisionEngine)
    engine._ppe_model_path = str(weights)
    engine._device = "cpu"
    engine._ppe_model = FakeModel("initial")
    engine._ppe_weights_mtime = engine._ppe_weights_mtime_on_disk()
    engine.loads = loads

    return engine


def _install(engine: PPEVisionEngine, content: bytes) -> Path:
    weights = Path(engine._ppe_model_path)
    weights.write_bytes(content)
    _bump_mtime(weights, engine._ppe_weights_mtime)
    return weights


def test_unchanged_file_is_not_reloaded(engine: PPEVisionEngine) -> None:
    before = engine._ppe_model

    engine._reload_ppe_model_if_changed()

    assert engine._ppe_model is before
    assert engine.loads == []


def test_changed_file_is_loaded_once(engine: PPEVisionEngine) -> None:
    before = engine._ppe_model

    weights = _install(engine, b"new weights")

    engine._reload_ppe_model_if_changed()

    assert engine._ppe_model is not before
    assert engine._ppe_model.path == str(weights)
    assert engine._ppe_model.device == "cpu"
    assert engine.loads == [str(weights)]

    # Same file again: nothing to do.
    engine._reload_ppe_model_if_changed()

    assert engine.loads == [str(weights)]


def test_broken_file_keeps_the_previous_model(
    engine: PPEVisionEngine, monkeypatch
) -> None:
    before = engine._ppe_model

    def broken_yolo(path):
        engine.loads.append(path)
        raise RuntimeError("not a checkpoint")

    monkeypatch.setattr(ppe_vision_engine, "YOLO", broken_yolo)

    _install(engine, b"garbage")

    engine._reload_ppe_model_if_changed()

    assert engine._ppe_model is before
    assert len(engine.loads) == 1

    # A broken file is tried once, not on every frame.
    engine._reload_ppe_model_if_changed()

    assert len(engine.loads) == 1

    # Once the file changes again it is retried.
    def fixed_yolo(path):
        engine.loads.append(path)
        return FakeModel(path)

    monkeypatch.setattr(ppe_vision_engine, "YOLO", fixed_yolo)

    _install(engine, b"fixed weights")

    engine._reload_ppe_model_if_changed()

    assert engine._ppe_model is not before
    assert len(engine.loads) == 2


def test_missing_file_keeps_the_previous_model(engine: PPEVisionEngine) -> None:
    before = engine._ppe_model

    Path(engine._ppe_model_path).unlink()

    engine._reload_ppe_model_if_changed()

    assert engine._ppe_model is before
    assert engine.loads == []
