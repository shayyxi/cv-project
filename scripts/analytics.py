"""
Phase 2 analytics CLI.

Usage:
    python -m scripts.analytics score [--day YYYY-MM-DD]
    python -m scripts.analytics trends [--days N] [--weekly] [--camera ID]
    python -m scripts.analytics repeat [--days N] [--min N]
    python -m scripts.analytics heatmap --camera ID [--days N] [--json]
        [--image PATH]
    python -m scripts.analytics report [--days N] [--webhook] [--email]
    python -m scripts.analytics risk [--train] [--min-days N]
    python -m scripts.analytics export-crops [--days N] [--limit N] [--all]
"""

import argparse
from datetime import date

from app.analytics import (
    AnalyticsRepository,
    HeatmapService,
    LabelingService,
    ReportService,
    RiskService,
    ScoringService,
    TrendsService,
)
from app.delivery import ReportDeliveryService
from app.storage.database import SessionLocal
from app.utils.logging import configure_logging


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def cmd_score(args, repository: AnalyticsRepository) -> None:
    service = ScoringService(repository)

    result = service.overall_score(day=args.day)

    if result is None:
        print("No detections for that day.")
        return

    print(
        f"Overall safety score {result['day']}: "
        f"{result['score']}/100 over {result['workers']} workers"
    )

    print(f"\n{'camera':<10} {'workers':>8} {'score':>8}")

    for row in result["cameras"]:
        print(
            f"{row['camera_id']:<10} "
            f"{row['workers']:>8} "
            f"{row['score']:>8}"
        )


def cmd_trends(args, repository: AnalyticsRepository) -> None:
    service = TrendsService(repository)

    rows = service.frequency_trends(
        days=args.days,
        weekly=args.weekly,
        camera_id=args.camera,
    )

    if not rows:
        print("No data in window.")
        return

    period_label = "week" if args.weekly else "day"

    header = f"{period_label:<12} "

    if not args.weekly:
        header += f"{'camera':<10} "

    header += f"{'workers':>8} {'helmet':>8} {'vest':>8} {'boots':>8}"

    print(header)

    for row in rows:
        period = row["period"] if args.weekly else row["day"]

        line = f"{str(period):<12} "

        if not args.weekly:
            line += f"{row['camera_id']:<10} "

        line += (
            f"{row['workers']:>8} "
            f"{row['helmet_viol']:>8} "
            f"{row['vest_viol']:>8} "
            f"{row['boots_viol']:>8}"
        )

        print(line)


def cmd_repeat(args, repository: AnalyticsRepository) -> None:
    service = TrendsService(repository)

    flags = service.repeat_noncompliance(
        days=args.days,
        min_violations=args.min,
    )

    if not flags:
        print("No repeated non-compliance flags.")
        return

    for flag in flags:
        print(
            f"[FLAG] camera {flag['camera_id']}: "
            f"{flag['violations']}x missing {flag['class']} "
            f"in last {flag['window_days']}d"
        )


def cmd_heatmap(args, repository: AnalyticsRepository) -> None:
    service = HeatmapService(repository)

    output_path = service.violation_heatmap(
        camera_id=args.camera,
        days=args.days,
        reference_image=args.image,
        as_json=args.json,
    )

    if output_path is None:
        print("No violations logged for that camera/window.")
        return

    print(f"Heatmap written to {output_path}")


def cmd_report(args, repository: AnalyticsRepository) -> None:
    service = ReportService(
        repository=repository,
        scoring_service=ScoringService(repository),
        trends_service=TrendsService(repository),
        risk_service=RiskService(repository),
        heatmap_service=HeatmapService(repository),
    )

    report_path = service.generate_report(days=args.days)

    print(f"Report written to {report_path}")

    delivery = ReportDeliveryService()

    if args.webhook:
        status = delivery.deliver_webhook(report_path)
        print(f"Webhook delivery: HTTP {status}")

    if args.email:
        delivery.deliver_email(report_path)
        print("Emailed.")


def cmd_risk(args, repository: AnalyticsRepository) -> None:
    service = RiskService(repository)

    if args.train:
        model = service.train(min_days=args.min_days)

        if model is None:
            print(
                f"Not enough history to train "
                f"(needs >= {args.min_days} days) - keep logging."
            )
        else:
            print("Risk model trained and saved.")
        return

    result = service.score(min_days=args.min_days)

    if result is None:
        print(
            f"Not enough history to score "
            f"(needs >= {args.min_days} days) - keep logging."
        )
        return

    print(
        f"Predicted violations tomorrow: "
        f"{result['predicted_violations']} "
        f"-> daily risk score {result['risk_score']}/100"
    )


def cmd_export_crops(args, repository: AnalyticsRepository) -> None:
    service = LabelingService(repository)

    result = service.export_violation_crops(
        days=args.days,
        limit=args.limit,
        include_compliant=args.all,
    )

    print(
        f"{result['saved']} crops, "
        f"{result['annotations']} PPE pre-labels -> "
        f"{result['output_dir']} "
        f"({result['skipped']} skipped, "
        f"COCO: {result['coco_path']})"
    )


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Phase 2 analytics")

    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Safety scores for one day")
    score.add_argument("--day", type=_parse_day, default=None)

    trends = subparsers.add_parser("trends", help="Violation trends")
    trends.add_argument("--days", type=int, default=None)
    trends.add_argument("--weekly", action="store_true")
    trends.add_argument("--camera", default=None)

    repeat = subparsers.add_parser("repeat", help="Repeated non-compliance")
    repeat.add_argument("--days", type=int, default=None)
    repeat.add_argument("--min", type=int, default=None)

    heatmap = subparsers.add_parser("heatmap", help="Violation heatmap")
    heatmap.add_argument("--camera", required=True)
    heatmap.add_argument("--days", type=int, default=None)
    heatmap.add_argument("--json", action="store_true")
    heatmap.add_argument("--image", default=None)

    report = subparsers.add_parser("report", help="Compliance report PDF")
    report.add_argument("--days", type=int, default=7)
    report.add_argument("--webhook", action="store_true")
    report.add_argument("--email", action="store_true")

    risk = subparsers.add_parser("risk", help="Predictive risk score")
    risk.add_argument("--train", action="store_true")
    risk.add_argument("--min-days", type=int, default=14)

    export_crops = subparsers.add_parser(
        "export-crops", help="Export violation crops for labelling"
    )
    export_crops.add_argument("--days", type=int, default=7)
    export_crops.add_argument("--limit", type=int, default=200)
    export_crops.add_argument(
        "--all",
        action="store_true",
        help="Export every detected person, not only violations.",
    )

    args = parser.parse_args()

    session = SessionLocal()

    try:
        repository = AnalyticsRepository(session)

        # Materialized views only update on refresh; make sure the
        # read commands see the latest detections.
        if args.command in (
            "score", "trends", "repeat", "heatmap", "report", "risk",
        ):
            repository.refresh_views()

        if args.command == "score":
            cmd_score(args, repository)
        elif args.command == "trends":
            cmd_trends(args, repository)
        elif args.command == "repeat":
            cmd_repeat(args, repository)
        elif args.command == "heatmap":
            cmd_heatmap(args, repository)
        elif args.command == "report":
            cmd_report(args, repository)
        elif args.command == "risk":
            cmd_risk(args, repository)
        elif args.command == "export-crops":
            cmd_export_crops(args, repository)

    finally:
        session.close()


if __name__ == "__main__":
    main()
