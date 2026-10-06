from pathlib import Path

import pytest

from app.retraining.train import (
    CV_DIR,
    TrainingResult,
    install_weights,
    ppe_class_names,
    ppe_weights_path,
)


def test_ppe_weights_path_resolves_relative_to_cv_package() -> None:
    config = {"models": {"ppe": {"path": "weights/best.pt"}}}

    assert ppe_weights_path(config) == (CV_DIR / "weights" / "best.pt").resolve()


def test_ppe_weights_path_keeps_absolute_paths(tmp_path) -> None:
    config = {"models": {"ppe": {"path": str(tmp_path / "custom.pt")}}}

    assert ppe_weights_path(config) == tmp_path / "custom.pt"


def test_ppe_class_names_are_lower_cased() -> None:
    config = {
        "classes": {"ppe": {"helmet": "Helmet", "vest": "vest", "boots": "BOOTS"}}
    }

    assert ppe_class_names(config) == ["helmet", "vest", "boots"]


def test_real_vision_config_is_readable() -> None:
    assert ppe_class_names() == ["helmet", "vest", "boots"]
    assert ppe_weights_path().name == "best.pt"


def test_install_weights_backs_up_the_previous_file(tmp_path) -> None:
    best = tmp_path / "run" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"new weights")

    target = tmp_path / "weights" / "best.pt"
    target.parent.mkdir()
    target.write_bytes(b"old weights")

    installed, backup = install_weights(best, target)

    assert installed == target
    assert target.read_bytes() == b"new weights"

    assert backup is not None
    assert backup.parent == target.parent
    assert backup.name.startswith("best_backup_")
    assert backup.suffix == ".pt"
    assert backup.read_bytes() == b"old weights"


def test_install_weights_without_a_previous_file(tmp_path) -> None:
    best = tmp_path / "best.pt"
    best.write_bytes(b"new weights")

    target = tmp_path / "weights" / "best.pt"

    installed, backup = install_weights(best, target)

    assert installed == target
    assert backup is None
    assert target.read_bytes() == b"new weights"


def test_install_weights_needs_a_source_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        install_weights(tmp_path / "missing.pt", tmp_path / "best.pt")


def test_training_result_summary_picks_headline_metrics() -> None:
    result = TrainingResult(
        best=Path("best.pt"),
        last=Path("last.pt"),
        save_dir=Path("run"),
        metrics={
            "metrics/precision(B)": 0.812345,
            "metrics/mAP50(B)": 0.91234,
            "metrics/mAP50-95(B)": 0.6,
            "fitness": 0.7,
        },
    )

    assert result.summary == {
        "precision": 0.8123,
        "mAP50": 0.9123,
        "mAP50-95": 0.6,
    }


# --- install gate -----------------------------------------------------------

from app.retraining.train import (  # noqa: E402
    ModelComparison,
    comparison_split,
    metrics_summary,
)


def _summary(map5095: float, per_class: dict | None = None) -> dict:
    return {
        "precision": 0.8,
        "recall": 0.7,
        "mAP50": 0.9,
        "mAP50-95": map5095,
        "per_class": per_class or {},
    }


def test_comparison_better_when_new_is_equal_or_higher() -> None:
    assert ModelComparison("val", _summary(0.60), _summary(0.60)).better is True
    assert ModelComparison("val", _summary(0.60), _summary(0.61)).better is True
    assert ModelComparison("val", _summary(0.60), _summary(0.59)).better is False


def test_comparison_reports_classes_that_regressed() -> None:
    old = _summary(0.6, {"helmet": 0.70, "vest": 0.60, "boots": 0.50})
    new = _summary(0.6, {"helmet": 0.62, "vest": 0.58, "boots": 0.55})

    comparison = ModelComparison("test", old, new)

    # helmet dropped by 0.08 (> 0.05); vest by 0.02 (ignored); boots improved.
    assert comparison.regressed_classes == ["helmet"]


def test_comparison_ignores_classes_missing_on_one_side() -> None:
    old = _summary(0.6, {"helmet": 0.70, "gloves": 0.90})
    new = _summary(0.6, {"helmet": 0.70})

    assert ModelComparison("val", old, new).regressed_classes == []


def test_comparison_rows_cover_headline_and_per_class() -> None:
    old = _summary(0.60, {"helmet": 0.70})
    new = _summary(0.65, {"helmet": 0.72, "vest": 0.40})

    rows = ModelComparison("val", old, new).rows()

    assert [row[0] for row in rows] == [
        "precision", "recall", "mAP50", "mAP50-95",
        "mAP50-95 helmet", "mAP50-95 vest",
    ]
    assert rows[3] == ("mAP50-95", 0.60, 0.65)
    assert rows[5] == ("mAP50-95 vest", 0.0, 0.40)


def test_comparison_split_prefers_test(tmp_path) -> None:
    with_test = tmp_path / "with_test.yaml"
    with_test.write_text(
        "train: /d/train\nval: /d/valid\ntest: /d/test\n", encoding="utf-8"
    )

    without_test = tmp_path / "without_test.yaml"
    without_test.write_text("train: /d/train\nval: /d/valid\n", encoding="utf-8")

    assert comparison_split(with_test) == "test"
    assert comparison_split(without_test) == "val"


class FakeMetrics:
    """The parts of ultralytics DetMetrics the summary reads."""

    results_dict = {
        "metrics/precision(B)": 0.81,
        "metrics/recall(B)": 0.72,
        "metrics/mAP50(B)": 0.90,
        "metrics/mAP50-95(B)": 0.61,
        "fitness": 0.64,
    }
    names = {0: "boots", 1: "helmet", 2: "vest"}
    maps = [0.50, 0.70, 0.63]


def test_metrics_summary_from_ultralytics_metrics() -> None:
    summary = metrics_summary(FakeMetrics())

    assert summary["precision"] == 0.81
    assert summary["recall"] == 0.72
    assert summary["mAP50"] == 0.90
    assert summary["mAP50-95"] == 0.61
    assert summary["per_class"] == {"boots": 0.50, "helmet": 0.70, "vest": 0.63}


def test_metrics_summary_tolerates_missing_fields() -> None:
    summary = metrics_summary(object())

    assert summary["mAP50-95"] == 0.0
    assert summary["per_class"] == {}


# --- comparison record for the page ------------------------------------------

import json  # noqa: E402

from app.retraining.train import comparison_to_dict, write_comparison  # noqa: E402


def test_comparison_to_dict_has_overall_and_per_class(tmp_path) -> None:
    comparison = ModelComparison(
        "test",
        _summary(0.4852, {"boots": 0.492, "helmet": 0.5138, "vest": 0.4498}),
        _summary(0.4328, {"boots": 0.371, "helmet": 0.4964, "vest": 0.4311}),
    )

    record = comparison_to_dict(
        comparison,
        installed=False,
        run_name="ppe-x",
        new_weights=tmp_path / "run" / "weights" / "best.pt",
        deployed_weights=tmp_path / "deployed" / "best.pt",
    )

    assert record["run"] == "ppe-x"
    assert record["split"] == "test"
    assert record["better"] is False
    assert record["installed"] is False
    assert record["installed_at"] is None
    assert record["new_weights"] == str((tmp_path / "run" / "weights" / "best.pt").resolve())
    assert record["deployed_weights"] == str((tmp_path / "deployed" / "best.pt").resolve())
    assert record["regressed_classes"] == ["boots"]
    assert record["deployed"]["mAP50-95"] == 0.4852
    assert record["new"]["mAP50-95"] == 0.4328
    assert record["per_class"]["boots"] == {
        "deployed": 0.492, "new": 0.371, "delta": -0.121,
    }

    latest = write_comparison(
        comparison,
        installed=False,
        run_name="ppe-x",
        run_dir=tmp_path / "run",
        latest_path=tmp_path / "latest.json",
    )

    assert latest == tmp_path / "latest.json"
    assert json.loads(latest.read_text(encoding="utf-8"))["run"] == "ppe-x"
    assert (tmp_path / "run" / "comparison.json").exists()


def test_mark_installed_updates_latest_and_run_copy(tmp_path) -> None:
    from app.retraining.train import mark_installed

    comparison = ModelComparison("test", _summary(0.5), _summary(0.6))
    run_dir = tmp_path / "run"
    latest = tmp_path / "latest.json"

    write_comparison(
        comparison,
        installed=False,
        run_name="ppe-y",
        run_dir=run_dir,
        latest_path=latest,
        new_weights=run_dir / "weights" / "best.pt",
        deployed_weights=tmp_path / "deployed.pt",
    )

    record = mark_installed(latest)

    assert record["installed"] is True
    assert record["installed_at"] is not None

    assert json.loads(latest.read_text(encoding="utf-8"))["installed"] is True
    run_copy = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    assert run_copy["installed"] is True
    assert run_copy["installed_at"] == record["installed_at"]
