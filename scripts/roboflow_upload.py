"""
Upload label-queue crops to Roboflow, with the model's PPE boxes as
pre-labels. Each crop is sent once; later runs only send new ones.

Usage:
    python -m scripts.roboflow_upload                  # upload what is new, once
    python -m scripts.roboflow_upload --dry-run        # only count pending crops
    python -m scripts.roboflow_upload --export         # run the label-queue export first
    python -m scripts.roboflow_upload --watch          # keep running, check every 30 min
    python -m scripts.roboflow_upload --watch --interval 600 --export

Options:
    --export            export violation crops first, like
                        `scripts.analytics export-crops` (--days, --limit, --all)
    --upload-limit N    cap the number of crops sent per run
    --watch             loop forever: (export,) upload, sleep --interval seconds
    --interval SECONDS  watch cadence (default 1800)

Automatic uploads: the pipeline's daily job already fills the label
queue at REPORT_HOUR. Run this script with --watch next to the
pipeline (or without --watch from cron / Task Scheduler) and every
new crop reaches Roboflow without anyone touching it.

Needs ROBOFLOW_API_KEY and ROBOFLOW_PROJECT in .env. See
docs/retraining.md.
"""

import argparse
import logging
import sys
import time

from app.analytics import AnalyticsRepository, LabelingService
from app.retraining.config import (
    HOW_TO_CONFIGURE,
    LABEL_QUEUE_DIR,
    RoboflowSettings,
)
from app.retraining.upload import RoboflowUploadService
from app.storage.database import SessionLocal
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def export_crops(args) -> None:
    session = SessionLocal()

    try:
        result = LabelingService(
            AnalyticsRepository(session)
        ).export_violation_crops(
            days=args.days,
            limit=args.limit,
            include_compliant=args.all,
        )
    finally:
        session.close()

    print(
        f"Exported {result['saved']} crops, "
        f"{result['annotations']} PPE pre-labels -> "
        f"{result['output_dir']} ({result['skipped']} skipped)"
    )


def upload_crops(args, config: RoboflowSettings) -> dict:
    service = RoboflowUploadService(config=config)

    result = service.upload_new_crops(
        limit=args.upload_limit,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(
            f"{result['pending']} crops pending upload "
            f"(dry run; manifest: {result['manifest']})"
        )
    else:
        print(
            f"Roboflow: {result['uploaded']} uploaded, "
            f"{result['duplicates']} duplicates, "
            f"{result['failed']} failed of {result['pending']} pending "
            f"-> batch {result['batch']} ({config.project_name})"
        )

    return result


def run_once(args, config: RoboflowSettings) -> dict:
    if args.export:
        export_crops(args)

    return upload_crops(args, config)


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Upload label-queue crops to Roboflow"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upload-limit", type=int, default=None)

    parser.add_argument(
        "--export",
        action="store_true",
        help="Run the label-queue export before uploading",
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--all",
        action="store_true",
        help="With --export: every detected person, not only violations",
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep running and repeat every --interval seconds",
    )
    parser.add_argument("--interval", type=int, default=1800)

    args = parser.parse_args()

    config = RoboflowSettings.from_env()

    if not config.configured and not args.dry_run:
        sys.exit(f"Error: Roboflow is not configured; {HOW_TO_CONFIGURE}.")

    if not args.watch:
        run_once(args, config)
        return

    logger.info(
        "Watching %s every %d s for new crops to upload.",
        LABEL_QUEUE_DIR,
        args.interval,
    )

    while True:
        try:
            run_once(args, config)
        except Exception:
            # Never let one bad cycle stop the watcher.
            logger.exception("Roboflow upload cycle failed")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
