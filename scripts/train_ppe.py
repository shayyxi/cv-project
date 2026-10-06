"""
Retrain the PPE detector (weights/best.pt) from a Roboflow dataset.

Usage:
    python -m scripts.train_ppe                        # latest version
    python -m scripts.train_ppe --generate             # new version from the current labels
    python -m scripts.train_ppe --version 3
    python -m scripts.train_ppe --data path/to/data.yaml   # local YOLO dataset, no Roboflow
    python -m scripts.train_ppe --generate --install   # ...and replace weights/best.pt

Options:
    --epochs N  --imgsz N  --batch N  --device cpu|0|0,1
    --base-model PATH        weights to fine-tune from
                             (default: the deployed best.pt, else yolov8n.pt)
    --name NAME              run folder under data/analytics/training/
    --patience N  --workers N
    --redownload             fetch a version again even if already on disk
    --version-settings PATH  JSON with Roboflow preprocessing/augmentation
                             settings, used with --generate
    --skip-class-check       train even if the dataset's class names differ
                             from classes.ppe in vision_config.yaml
    --install                for scripted runs: after the comparison, copy
                             the new best.pt over the engine's weights if its
                             mAP50-95 is not lower (previous file kept as a
                             backup). Without it nothing is installed; the
                             retraining page offers a replace button instead.
    --weights PATH           skip training; compare (and with --install,
                             install) this best.pt

After training, the new model is always evaluated against the deployed
one on the same split (test if the dataset has one, else valid). The
table is printed and written to data/analytics/training/
latest_comparison.json for the page.

Needs ROBOFLOW_API_KEY and ROBOFLOW_PROJECT in .env unless --data is
given. See docs/retraining.md for the full loop.
"""

import argparse
import json
import sys
from pathlib import Path

from app.retraining.config import (
    HOW_TO_CONFIGURE,
    RoboflowNotConfigured,
    RoboflowSettings,
)
from app.retraining.dataset import (
    RoboflowDatasetService,
    check_class_names,
    dataset_class_names,
    prepare_data_yaml,
)
from app.retraining.train import (
    REGRESSION_TOLERANCE,
    ModelComparison,
    compare_models,
    default_base_model,
    install_weights,
    ppe_class_names,
    ppe_weights_path,
    train_ppe_model,
    write_comparison,
)
from app.utils.logging import configure_logging


def _resolve_dataset(args) -> Path:
    """
    The prepared data.yaml: a local one, or a Roboflow version
    downloaded (and optionally generated) first.
    """

    if args.data:
        return prepare_data_yaml(Path(args.data))

    config = RoboflowSettings.from_env()

    if not config.configured:
        raise RoboflowNotConfigured(
            f"{HOW_TO_CONFIGURE}, or pass --data with a local YOLO "
            f"data.yaml"
        )

    version_settings = None

    if args.version_settings:
        version_settings = json.loads(
            Path(args.version_settings).read_text(encoding="utf-8")
        )

    service = RoboflowDatasetService()

    return service.download(
        version_number=args.version,
        generate=args.generate,
        version_settings=version_settings,
        redownload=args.redownload,
    )


def _check_classes(data_yaml: Path, skip: bool) -> None:
    """
    The engine finds helmet / vest / boots by class name, so a model
    trained with other names would silently detect nothing.
    """

    names = dataset_class_names(data_yaml)
    expected = ppe_class_names()

    missing, extra = check_class_names(names, expected)

    print(f"Dataset classes: {', '.join(names) or '(none)'}")
    print(f"Engine expects:  {', '.join(expected)}")

    if extra:
        print(f"Note: the engine ignores extra classes: {', '.join(extra)}")

    if not missing:
        return

    message = (
        f"the dataset is missing classes the engine needs: "
        f"{', '.join(missing)}. Rename them in Roboflow, or change "
        f"classes.ppe in vision_config.yaml."
    )

    if skip:
        print(f"Warning: {message}")
        return

    sys.exit(f"Error: {message} Pass --skip-class-check to train anyway.")


def _print_comparison(comparison: ModelComparison) -> None:
    print(f"\nDeployed vs new on the {comparison.split} split:")
    print(f"  {'metric':<20}{'deployed':>10}{'new':>10}")

    for label, old, new in comparison.rows():
        print(f"  {label:<20}{old:>10.4f}{new:>10.4f}")

    regressed = comparison.regressed_classes

    if regressed:
        print(
            f"  Warning: mAP50-95 dropped by more than "
            f"{REGRESSION_TOLERANCE} on: {', '.join(regressed)}"
        )


def _install(best: Path) -> None:
    target, backup = install_weights(best)

    print()
    print(f"Installed -> {target}")

    if backup:
        print(f"Previous weights kept at {backup}")

    print(
        "Restart the pipeline (or rebuild the Docker image) to load "
        "the new weights."
    )


def _compare_and_maybe_install(
    best: Path,
    data_yaml: Path,
    args,
    run_name: str,
) -> None:
    """
    Always compare the new weights with the deployed ones on the same
    split and record the result. Install only with --install, and then
    only when the new model is not worse.
    """

    deployed = ppe_weights_path()

    if not deployed.exists():
        print(f"\nNo deployed weights at {deployed}; nothing to compare against.")

        if args.install:
            _install(best)
        else:
            print(f"Copy {best} there by hand, or re-run with --install.")

        return

    comparison = compare_models(
        deployed,
        best,
        data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        run_name=run_name,
    )

    _print_comparison(comparison)

    install_now = bool(args.install) and comparison.better

    # Record for the Streamlit page (and in the run folder when best.pt
    # sits in one).
    run_dir = best.parent.parent if best.parent.name == "weights" else None

    write_comparison(
        comparison,
        installed=install_now,
        run_name=run_name,
        run_dir=run_dir,
        new_weights=best,
        deployed_weights=deployed,
    )

    if install_now:
        _install(best)
        return

    if args.install:
        print(
            f"\nNot installed: new mAP50-95 {comparison.new['mAP50-95']:.4f} "
            f"is below the deployed {comparison.old['mAP50-95']:.4f} on the "
            f"{comparison.split} split."
        )
        print(f"Deployed weights kept at {deployed}. The new model stays at {best}.")
        return

    print(
        f"\nNot installed. The new model stays at {best}. Replace the "
        f"deployed model from the retraining page, or re-run with "
        f"--install to install automatically when not worse."
    )


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Retrain the PPE detector from a Roboflow dataset"
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--version",
        type=int,
        default=None,
        help="Roboflow version number (default: the latest one)",
    )
    source.add_argument(
        "--generate",
        action="store_true",
        help="Generate a new Roboflow version from the current labels first",
    )
    source.add_argument(
        "--data",
        default=None,
        help="Local YOLO data.yaml to train on instead of Roboflow",
    )

    parser.add_argument(
        "--version-settings",
        default=None,
        help="JSON file with Roboflow preprocessing/augmentation for --generate",
    )
    parser.add_argument("--redownload", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, 0, or 0,1 (default: auto-detect)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=f"Weights to fine-tune from (default: {default_base_model()})",
    )
    parser.add_argument("--name", default=None, help="Run name")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-class-check", action="store_true")
    parser.add_argument(
        "--install",
        action="store_true",
        help=(
            "After the comparison, replace the engine's PPE weights with "
            "the new best.pt if it scores at least as well as the deployed "
            "one (for scripted runs; the page has a replace button)"
        ),
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Skip training and compare / install this best.pt instead",
    )

    args = parser.parse_args()

    try:
        data_yaml = _resolve_dataset(args)
    except RoboflowNotConfigured as error:
        sys.exit(f"Error: {error}")

    print(f"Dataset: {data_yaml}")

    _check_classes(data_yaml, skip=args.skip_class_check)

    if args.weights:
        best = Path(args.weights).resolve()

        if not best.is_file():
            sys.exit(f"Error: no weights file at {best}")

        print(f"Skipping training; using {best}")

        # <run>/weights/best.pt -> <run>; anything else -> file stem.
        run_name = (
            best.parent.parent.name
            if best.parent.name == "weights"
            else best.stem
        )
    else:
        result = train_ppe_model(
            data_yaml=data_yaml,
            base_model=args.base_model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            run_name=args.name,
            patience=args.patience,
            workers=args.workers,
        )

        print()
        print(f"Run folder: {result.save_dir}")
        print(f"best.pt:    {result.best}")

        for label, value in result.summary.items():
            print(f"  {label:<9} {value}")

        best = result.best
        run_name = result.save_dir.name

    _compare_and_maybe_install(best, data_yaml, args, run_name)


if __name__ == "__main__":
    main()
