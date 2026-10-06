import os
import shutil
from pathlib import Path

import pytest

from app.processing.Image_cropper.image_cropper import CONFIG_PATH, ImageCropper
from app.retraining.regions import render_regions_yaml, save_region


def _bump_mtime(path: Path) -> None:
    """Make sure the change is visible even within the same second."""

    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 2))


@pytest.fixture()
def cropper(tmp_path: Path) -> ImageCropper:
    config = tmp_path / "image_crop_config.yaml"
    shutil.copy(CONFIG_PATH, config)

    cropper = ImageCropper()
    # Point the instance at the scratch copy, as if it were the real file.
    cropper._config_path = config
    cropper._config_mtime = None  # force one reload from the scratch copy

    return cropper


def test_polygon_follows_changes_to_the_config_file(cropper: ImageCropper) -> None:
    before = cropper._get_polygon("12846")
    assert before[0] == (7, 1649)

    save_region("12846", [(100, 200), (3000, 2500)], path=cropper._config_path)
    _bump_mtime(cropper._config_path)

    after = cropper._get_polygon("12846")
    assert after == [(100, 200), (3000, 200), (3000, 2500), (100, 2500)]

    # Other cameras are still there, and a new one appears without a restart.
    assert cropper._get_polygon("12990")[0] == (24, 2487)

    save_region("55555", [(0, 0), (10, 10)], path=cropper._config_path)
    _bump_mtime(cropper._config_path)

    assert cropper._get_polygon("55555") == [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_unchanged_file_is_not_reread(cropper: ImageCropper, monkeypatch) -> None:
    cropper._get_polygon("12846")

    calls = []
    monkeypatch.setattr(
        ImageCropper,
        "_load_config",
        staticmethod(lambda path=None: calls.append(path) or {"regions": {}}),
    )

    cropper._get_polygon("12846")
    assert calls == []


def test_broken_file_keeps_the_last_good_config(cropper: ImageCropper) -> None:
    good = cropper._get_polygon("12875")

    cropper._config_path.write_text("regions: [not: valid: yaml", encoding="utf-8")
    _bump_mtime(cropper._config_path)

    assert cropper._get_polygon("12875") == good

    # Once the file is valid again it is picked up.
    cropper._config_path.write_text(
        render_regions_yaml({"12875": [(1, 1), (9, 1), (9, 9), (1, 9)]}),
        encoding="utf-8",
    )
    _bump_mtime(cropper._config_path)

    assert cropper._get_polygon("12875") == [(1, 1), (9, 1), (9, 9), (1, 9)]
