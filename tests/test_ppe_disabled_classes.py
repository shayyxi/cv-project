import os
import shutil
from pathlib import Path

import pytest
import yaml

from app.processing.cv.ppe_vision_engine import PPEVisionEngine
from app.retraining.regions import (
    CROP_CONFIG_PATH,
    load_disabled_ppe,
    load_regions,
    save_disabled_ppe,
    save_region,
)


@pytest.fixture()
def config(tmp_path: Path) -> Path:
    path = tmp_path / "image_crop_config.yaml"
    shutil.copy(CROP_CONFIG_PATH, path)
    return path


def test_save_disabled_ppe_keeps_crops_and_other_cameras(config: Path) -> None:
    before = load_regions(config)

    save_disabled_ppe("12846", {"boots"}, config)

    assert load_disabled_ppe(config) == {"12846": {"boots"}}
    assert load_regions(config) == before

    # A crop save keeps the switches, and a second camera's switch keeps
    # the first camera's.
    save_region("12875", [(0, 0), (10, 10)], config, image_size=(100, 100))
    save_disabled_ppe("12875", ["Helmet", "vest"], config)

    assert load_disabled_ppe(config) == {
        "12846": {"boots"},
        "12875": {"helmet", "vest"},
    }

    # Nothing switched off removes the camera, then the key itself.
    save_disabled_ppe("12846", set(), config)
    assert load_disabled_ppe(config) == {"12875": {"helmet", "vest"}}

    save_disabled_ppe("12875", set(), config)
    assert "disabled_ppe" not in yaml.safe_load(config.read_text(encoding="utf-8"))


def test_save_disabled_ppe_rejects_an_unknown_class(config: Path) -> None:
    with pytest.raises(ValueError):
        save_disabled_ppe("12846", {"gloves"}, config)


def _engine(tmp_path: Path) -> PPEVisionEngine:
    # Skip __init__: it loads the real person and PPE models.
    engine = PPEVisionEngine.__new__(PPEVisionEngine)
    engine._helmet_class = "helmet"
    engine._vest_class = "vest"
    engine._boots_class = "boots"
    engine._disabled_ppe_path = tmp_path / "image_crop_config.yaml"
    engine._disabled_ppe = {}
    engine._disabled_ppe_mtime = None
    return engine


HELMET_AND_VEST = [{"label": "helmet"}, {"label": "vest"}]


def test_a_switched_off_class_counts_as_present(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._disabled_ppe = {"12846": {"boots"}}

    verdict = engine._compliance("12846", HELMET_AND_VEST)

    assert (verdict.helmet, verdict.vest, verdict.boots) == (True, True, True)
    assert verdict.compliant is True

    # The same person on a camera that still checks boots.
    verdict = engine._compliance("12875", HELMET_AND_VEST)

    assert verdict.boots is False
    assert verdict.compliant is False

    # Switching one class off does not excuse the others.
    assert engine._compliance("12846", [{"label": "helmet"}]).compliant is False


def test_switches_are_reloaded_when_the_file_changes(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    path = engine._disabled_ppe_path

    # No file: nothing switched off.
    engine._reload_disabled_ppe_if_changed()
    assert engine._disabled_ppe == {}

    path.write_text(
        "regions: {}\ndisabled_ppe:\n  '12846': [boots]\n", encoding="utf-8"
    )
    engine._reload_disabled_ppe_if_changed()
    assert engine._disabled_ppe == {"12846": {"boots"}}

    # A broken file keeps the previous table.
    path.write_text("disabled_ppe: [", encoding="utf-8")
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 2))
    engine._reload_disabled_ppe_if_changed()
    assert engine._disabled_ppe == {"12846": {"boots"}}

    # A removed file checks everything again.
    path.unlink()
    engine._reload_disabled_ppe_if_changed()
    assert engine._disabled_ppe == {}
