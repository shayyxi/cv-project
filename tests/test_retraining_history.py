"""Run history, weight backups and per-camera SAHI settings behind the retraining page."""

import json
import os
from datetime import datetime

import pytest
import yaml

from app.retraining import history as h


def test_list_runs_reads_results_and_comparison(tmp_path):
    run = tmp_path / "ppe-20261003-141223"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"w")
    (run / "results.csv").write_text("epoch,loss\n1,0.5\n2,0.4\n3,0.3\n")
    (run / "comparison.json").write_text(json.dumps({
        "timestamp": "2026-10-03 14:12:23",
        "installed": False,
        "better": True,
        "new": {"mAP50-95": 0.21},
        "deployed": {"mAP50-95": 0.20},
    }))
    older = tmp_path / "ppe-20260920-114207"
    older.mkdir()
    stamp = datetime(2026, 9, 20, 11, 42, 7).timestamp()  # no comparison: the folder's own time
    os.utime(older, (stamp, stamp))
    (tmp_path / "latest_comparison.json").write_text("{}")  # a file, not a run

    runs = h.list_runs(tmp_path)

    assert [run.name for run in runs] == ["ppe-20261003-141223", "ppe-20260920-114207"]
    assert runs[0].epochs == 3 and runs[0].has_weights
    assert runs[0].map5095 == (0.21, 0.20)
    assert runs[0].started == datetime(2026, 10, 3, 14, 12, 23)
    assert runs[1].epochs is None and runs[1].comparison is None and not runs[1].has_weights
    assert runs[1].map5095 == (None, None)
    assert h.list_runs(tmp_path / "missing") == []


def test_deployed_weights_and_backups_newest_first(tmp_path):
    target = tmp_path / "best.pt"
    target.write_bytes(b"x" * 10)
    for stamp in ("20260816T215700", "20260920T114207"):
        (tmp_path / f"best_backup_{stamp}.pt").write_bytes(b"y")
    (tmp_path / "other.pt").write_bytes(b"z")

    backups = h.list_backups(target)

    assert [backup.modified for backup in backups] == [
        datetime(2026, 9, 20, 11, 42, 7),
        datetime(2026, 8, 16, 21, 57, 0),
    ]
    assert h.deployed_weights(target).size_bytes == 10
    assert h.deployed_weights(tmp_path / "none.pt") is None
    assert len(h.file_sha256(target)) == 12


def test_sahi_settings_roundtrip(tmp_path):
    path = tmp_path / "vision_config.yaml"
    path.write_text(yaml.safe_dump({
        "models": {"person": {"confidence": 0.15}},
        "sahi": {
            "default": {"target_size": 640, "overlap": 0.2},
            "cameras": {"12846": {"target_size": 640, "person_confidence": 0.34}},
        },
    }))

    defaults, cameras = h.load_sahi_settings(path)
    assert defaults["target_size"] == 640
    assert defaults["merge_iou"] == 0.45  # filled from DEFAULT_SAHI
    assert defaults["person_confidence"] == 0.15  # the person model's confidence
    assert cameras["12846"]["person_confidence"] == 0.34

    values = {**defaults, "target_size": 1200, "person_confidence": 0.41, h.SAHI_FLAG: True}
    assert h.save_sahi_settings("12990", values, path) == path

    content = yaml.safe_load(path.read_text())
    assert content["models"]["person"]["confidence"] == 0.15  # untouched
    entry = content["sahi"]["cameras"]["12990"]
    assert entry["target_size"] == 1200 and entry["person_confidence"] == 0.41
    assert entry[h.SAHI_FLAG] is True
    assert list(entry) == [key for key, *_ in h.SAHI_FIELDS] + [h.SAHI_FLAG]
    assert h.load_sahi_settings(path)[1]["12846"]["person_confidence"] == 0.34  # other camera kept

    with pytest.raises(ValueError):
        h.save_sahi_settings("", values, path)
    with pytest.raises(ValueError):
        h.save_sahi_settings("12990", {**values, "target_size": 5}, path)
    with pytest.raises(ValueError):
        h.save_sahi_settings("12990", {k: v for k, v in values.items() if k != "overlap"}, path)
