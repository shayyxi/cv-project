import calendar
import logging
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.analytics.analytics_config import load_analytics_config
from app.analytics.analytics_repository import AnalyticsRepository
from app.analytics.heatmap_service import HeatmapService
from app.analytics.report_analysis import (
    MIN_DAYS_FOR_WEEKDAY_PATTERN,
    RISK_MODEL_MIN_DAYS,
    analyze,
)
from app.analytics.report_charts import build_charts
from app.analytics.risk_service import RiskService
from app.analytics.scoring_service import ScoringService, weighted_score
from app.analytics.trends_service import TrendsService
from app.config import settings

logger = logging.getLogger(__name__)

INK = colors.HexColor("#0b0b0b")
INK_SECONDARY = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
RULE = colors.HexColor("#c3c2b7")
HAIRLINE = colors.HexColor("#e1e0d9")
TILE = colors.HexColor("#f4f4f1")
HEADER_BG = colors.HexColor("#f0efec")
ZEBRA = colors.HexColor("#fafaf8")
GOOD = colors.HexColor("#006300")
BAD = colors.HexColor("#d03b3b")

# Longest side of an embedded heatmap photo; half a page wide at
# 200 dpi needs about 700 px, so this leaves headroom for zooming.
HEATMAP_MAX_PIXELS = 1400


class ReportService:
    """
    PPE compliance report over a rolling window, written as PDF:
    headline figures against the prior period, plain-language
    findings, trend charts (daily violations, workers seen, safety
    score), breakdowns by PPE class and camera, weekday pattern,
    repeated non-compliance flags, violation heatmaps and the
    per-day per-camera table.

    trends, risk and heatmap services are optional: without them the
    report simply omits the flags, the forecast and the heatmaps.
    """

    def __init__(
        self,
        repository: AnalyticsRepository,
        scoring_service: ScoringService,
        trends_service: TrendsService | None = None,
        risk_service: RiskService | None = None,
        heatmap_service: HeatmapService | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self._repository = repository
        self._scoring_service = scoring_service
        self._trends_service = trends_service
        self._risk_service = risk_service
        self._heatmap_service = heatmap_service

        config = load_analytics_config()

        self._severity = config["severity"]
        self._trend_window_days = int(config["trends"]["window_days"])
        self._repeat_threshold = {
            "window_days": int(config["repeat_noncompliance"]["window_days"]),
            "min_violations": int(
                config["repeat_noncompliance"]["min_violations"]
            ),
        }

        self._output_dir = (
            Path(output_dir)
            if output_dir is not None
            else settings.local_analytics_dir
        )

    def generate_report(
        self,
        days: int = 7,
        today: date | None = None,
    ) -> Path:
        today = today or date.today()

        window_start = today - timedelta(days=days)
        prior_start = today - timedelta(days=2 * days)
        history_start = min(
            prior_start,
            today - timedelta(days=self._trend_window_days),
        )

        rows = self._repository.daily_stats(since=history_start)

        today_score = self._scoring_service.overall_score(day=today)

        analysis = analyze(
            rows=rows,
            window_start=window_start,
            prior_start=prior_start,
            history_start=history_start,
            today=today,
            days=days,
            severity=self._severity,
            repeat_flags=self._repeat_flags(days),
            repeat_threshold=self._repeat_threshold,
            risk=self._risk_forecast(),
            today_score=today_score,
        )

        heatmaps = self._heatmaps(analysis["cameras"], days)

        charts = build_charts(analysis)

        self._output_dir.mkdir(parents=True, exist_ok=True)

        output_path = self._output_dir / f"report_{today.isoformat()}.pdf"

        _PdfBuilder(
            output_path=output_path,
            analysis=analysis,
            charts=charts,
            heatmaps=heatmaps,
            severity=self._severity,
        ).build()

        logger.info("Report -> %s", output_path)

        return output_path

    # ==================================================================
    # Optional inputs: a failure here must never block the report.
    # ==================================================================

    def _repeat_flags(self, days: int) -> list[dict]:
        if self._trends_service is None:
            return []

        try:
            return self._trends_service.repeat_noncompliance(days=days)
        except Exception:
            logger.exception("Repeated non-compliance flags unavailable")
            return []

    def _risk_forecast(self) -> dict | None:
        if self._risk_service is None:
            return None

        try:
            return self._risk_service.details()
        except Exception:
            logger.exception("Risk forecast unavailable")
            return None

    def _heatmaps(
        self,
        cameras: list[dict],
        days: int,
    ) -> list[tuple[str, Path]]:
        if self._heatmap_service is None:
            return []

        heatmaps = []

        for camera in cameras:
            if not camera["violations"]:
                continue

            camera_id = camera["camera_id"]

            try:
                path = self._heatmap_service.violation_heatmap(
                    camera_id=camera_id,
                    days=days,
                )
            except Exception as error:
                # e.g. no raw frame on disk yet for this camera
                logger.info(
                    "Heatmap skipped for camera_id=%s: %s",
                    camera_id,
                    error,
                )
                continue

            if path is not None and Path(path).exists():
                heatmaps.append((camera_id, Path(path)))

        return heatmaps


# ======================================================================
# PDF layout
# ======================================================================


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "rpt_title",
            parent=base["Title"],
            fontSize=20,
            leading=24,
            alignment=TA_LEFT,
            textColor=INK,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "rpt_subtitle",
            parent=base["Normal"],
            fontSize=9.5,
            leading=12,
            textColor=INK_SECONDARY,
        ),
        "h2": ParagraphStyle(
            "rpt_h2",
            parent=base["Heading2"],
            fontSize=13,
            leading=16,
            textColor=INK,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "rpt_h3",
            parent=base["Heading3"],
            fontSize=10.5,
            leading=13,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "rpt_body",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=INK,
        ),
        "bullet": ParagraphStyle(
            "rpt_bullet",
            parent=base["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=INK,
            leftIndent=12,
            bulletIndent=0,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "rpt_caption",
            parent=base["Normal"],
            fontSize=8,
            leading=10.5,
            textColor=INK_SECONDARY,
            spaceBefore=2,
            spaceAfter=6,
        ),
        "tile_label": ParagraphStyle(
            "rpt_tile_label",
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
            textColor=INK_SECONDARY,
        ),
        "tile_value": ParagraphStyle(
            "rpt_tile_value",
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=INK,
            spaceBefore=3,
        ),
        "tile_delta": ParagraphStyle(
            "rpt_tile_delta",
            fontName="Helvetica",
            fontSize=7.5,
            leading=9,
            textColor=INK_SECONDARY,
            spaceBefore=2,
        ),
        "cell": ParagraphStyle(
            "rpt_cell",
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=INK,
        ),
    }


def _fmt_pct(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{100.0 * value:.{digits}f}%"


def _fmt_signed_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:+.0f}%"


def _fmt_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


class _PdfBuilder:
    def __init__(
        self,
        output_path: Path,
        analysis: dict,
        charts: dict[str, bytes],
        heatmaps: list[tuple[str, Path]],
        severity: dict[str, float],
    ) -> None:
        self._analysis = analysis
        self._charts = charts
        self._heatmaps = heatmaps
        self._severity = severity
        self._styles = _styles()

        self._doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=1.8 * cm,
            bottomMargin=2.0 * cm,
            title="PPE Compliance Report",
            author="AI vision pipeline",
        )

        self._width = self._doc.width

        self._footer_text = (
            "PPE Compliance Report  |  "
            f"{analysis['window_start'].isoformat()} to "
            f"{analysis['today'].isoformat()}"
        )

    # ------------------------------------------------------------------

    def build(self) -> None:
        elements: list = []

        elements += self._header()
        elements += self._kpi_row()
        elements += self._trends()
        elements += self._risk()
        elements += self._classes()
        elements += self._cameras()
        elements += self._weekdays()
        elements += self._heatmap_section()
        elements += self._daily_table()
        elements += self._findings()

        self._doc.build(
            elements,
            onFirstPage=self._footer,
            onLaterPages=self._footer,
        )

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _footer(self, canvas, doc) -> None:
        canvas.saveState()

        right = doc.pagesize[0] - doc.rightMargin

        canvas.setStrokeColor(HAIRLINE)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, 1.5 * cm, right, 1.5 * cm)

        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 1.1 * cm, self._footer_text)
        canvas.drawRightString(right, 1.1 * cm, f"Page {doc.page}")

        canvas.restoreState()

    def _p(self, text: str, style: str = "body") -> Paragraph:
        return Paragraph(text, self._styles[style])

    def _bullet(self, text: str) -> Paragraph:
        return Paragraph(
            f"<bullet>&bull;</bullet>{escape(text)}",
            self._styles["bullet"],
        )

    def _figure(self, png: bytes, width: float | None = None) -> Image:
        width = width or self._width

        image_width, image_height = ImageReader(BytesIO(png)).getSize()

        return Image(
            BytesIO(png),
            width=width,
            height=width * image_height / image_width,
        )

    def _image_file(self, path: Path, width: float) -> Image:
        """
        Embed a photo (heatmap overlay) downscaled to print size, so a
        full-resolution camera frame does not add megabytes to the PDF.
        """

        with PILImage.open(path) as source:
            photo = source.convert("RGB")
            photo.thumbnail((HEATMAP_MAX_PIXELS, HEATMAP_MAX_PIXELS))

            buffer = BytesIO()
            photo.save(buffer, format="JPEG", quality=82, optimize=True)

            image_width, image_height = photo.size

        buffer.seek(0)

        return Image(
            buffer,
            width=width,
            height=width * image_height / image_width,
        )

    def _table(
        self,
        header: list[str],
        rows: list[list],
        col_widths: list[float] | None = None,
        numeric_from: int = 1,
    ) -> Table:
        data = [header] + rows

        table = Table(
            data,
            colWidths=col_widths,
            repeatRows=1,
            hAlign="LEFT",
        )

        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("LEADING", (0, 0), (-1, -1), 10),
                    ("TEXTCOLOR", (0, 0), (-1, 0), INK_SECONDARY),
                    ("TEXTCOLOR", (0, 1), (-1, -1), INK),
                    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.3, HAIRLINE),
                    ("ALIGN", (numeric_from, 0), (-1, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, ZEBRA],
                    ),
                ]
            )
        )

        return table

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _header(self) -> list:
        a = self._analysis

        camera_count = len(a["cameras"])
        camera_text = (
            f"{camera_count} camera{'s' if camera_count != 1 else ''} "
            "with detections"
        )

        return [
            self._p("PPE Compliance Report", "title"),
            self._p(
                f"{a['window_start'].isoformat()} to "
                f"{a['today'].isoformat()} ({a['days']} days) "
                f"&nbsp;|&nbsp; {camera_text} &nbsp;|&nbsp; "
                f"generated {datetime.now():%Y-%m-%d %H:%M}",
                "subtitle",
            ),
            Spacer(1, 0.5 * cm),
        ]

    def _kpi_row(self) -> list:
        a = self._analysis
        current = a["current"]
        today_score = a["today_score"]
        risk = a["risk"] if a["risk_available"] else None

        violations_change = a["violations_change_pct"]
        score_change = a["score_change"]

        def delta_color(value: float | None, up_is_good: bool):
            if value is None or abs(value) < 1e-9:
                return INK_SECONDARY
            return GOOD if (value > 0) == up_is_good else BAD

        tiles = [
            (
                "Workers seen",
                f"{current['workers']}",
                (
                    f"{_fmt_signed_pct(a['workers_change_pct'])} vs prior"
                    if a["workers_change_pct"] is not None
                    else (
                        f"{current['days_active']} day"
                        f"{'s' if current['days_active'] != 1 else ''} "
                        "with detections"
                    )
                ),
                INK_SECONDARY,
            ),
            (
                "Violations",
                f"{current['violations']}",
                (
                    f"{_fmt_signed_pct(violations_change)} vs prior"
                    if violations_change is not None
                    else "no prior data"
                ),
                delta_color(violations_change, up_is_good=False),
            ),
            (
                "Fully compliant",
                _fmt_pct(current["compliance_rate"]),
                f"{current['compliant']} of {current['workers']}",
                INK_SECONDARY,
            ),
            (
                "Period score",
                _fmt_score(current["score"]),
                (
                    f"{score_change:+.1f} pts vs prior"
                    if score_change is not None
                    else "out of 100"
                ),
                delta_color(score_change, up_is_good=True),
            ),
            (
                "Score today",
                (
                    _fmt_score(today_score["score"])
                    if today_score
                    else "-"
                ),
                (
                    f"{today_score['workers']} workers today"
                    if today_score
                    else "no detections"
                ),
                INK_SECONDARY,
            ),
            (
                "Risk tomorrow",
                f"{risk['risk_score']:.0f}" if risk else "n/a",
                (
                    f"~{risk['predicted_violations']:.0f} violations"
                    if risk
                    else (
                        f"needs {(a['risk'] or {}).get('min_days', RISK_MODEL_MIN_DAYS)}d "
                        "history"
                    )
                ),
                INK_SECONDARY,
            ),
        ]

        gap = 0.2 * cm
        tile_width = (self._width - gap * (len(tiles) - 1)) / len(tiles)

        cells = []
        widths = []

        for index, (label, value, delta, color) in enumerate(tiles):
            delta_style = ParagraphStyle(
                f"rpt_tile_delta_{index}",
                parent=self._styles["tile_delta"],
                textColor=color,
            )

            cells.append(
                [
                    Paragraph(label, self._styles["tile_label"]),
                    Paragraph(value, self._styles["tile_value"]),
                    Paragraph(delta, delta_style),
                ]
            )
            widths.append(tile_width)

            if index < len(tiles) - 1:
                cells.append("")
                widths.append(gap)

        table = Table([cells], colWidths=widths, hAlign="LEFT")

        style = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]

        for column in range(0, len(cells), 2):
            style.append(("BACKGROUND", (column, 0), (column, 0), TILE))

        table.setStyle(TableStyle(style))

        return [table, Spacer(1, 0.3 * cm)]

    def _findings(self) -> list:
        return [
            KeepTogether(
                [self._p("Key findings", "h2")]
                + [
                    self._bullet(finding)
                    for finding in self._analysis["findings"]
                ]
            )
        ]

    def _trends(self) -> list:
        a = self._analysis
        heading = self._p("Trends", "h2")

        if "daily" not in self._charts:
            return [
                heading,
                self._p(
                    "No detections have been logged in the trend window, "
                    "so there is nothing to chart yet."
                ),
            ]

        elements = [
            KeepTogether(
                [
                    heading,
                    self._figure(self._charts["daily"]),
                    self._p(
                        "Each column stacks the day's violations by PPE "
                        "class; faded columns are the prior period. The "
                        "lower panel shows how many workers were seen, so "
                        "a rise in violations can be read against crew "
                        "size. A missing column means no detections were "
                        "logged that day.",
                        "caption",
                    ),
                ]
            )
        ]

        history = a["per_day_history"]

        elements.append(
            KeepTogether(
                [
                    Spacer(1, 0.2 * cm),
                    self._figure(self._charts["score"]),
                    self._p(
                        "Severity-weighted safety score per day from "
                        f"{history[0]['day'].isoformat()} to "
                        f"{history[-1]['day'].isoformat()}; the shaded "
                        "band is this report's period and the horizontal "
                        "line its worker-weighted average. Earlier days "
                        "are grey context.",
                        "caption",
                    ),
                ]
            )
        )

        return elements

    def _risk(self) -> list:
        a = self._analysis
        risk = a["risk"] or {}
        heading = self._p("Risk forecast", "h2")

        description = self._p(
            "The forecast is a gradient-boosted regression that predicts "
            "tomorrow's violation count from today's workers seen, "
            "violations, violations per worker, day of week, mean capture "
            "hour and the trailing 7-day rate. The risk score scales that "
            "prediction to 100 at the worst day on record.",
            "caption",
        )

        if not a["risk_available"]:
            min_days = risk.get("min_days", RISK_MODEL_MIN_DAYS)
            logged = risk.get("training_days")

            text = (
                f"Not available yet. The model needs at least {min_days} "
                "days of history with a following day logged"
            )

            if logged is not None:
                text += (
                    f"; {logged} {'is' if logged == 1 else 'are'} logged "
                    "so far"
                )

            text += (
                ". It trains automatically in the daily analytics job "
                "once enough history exists."
            )

            return [KeepTogether([heading, self._p(text), description])]

        tomorrow = a["today"] + timedelta(days=1)
        worst = risk.get("worst_day")

        headline = (
            f"Risk score for {tomorrow.isoformat()}: "
            f"<b>{risk['risk_score']:.0f}/100</b>. The model expects about "
            f"{risk['predicted_violations']:.0f} violations"
        )

        if worst:
            headline += (
                f"; the worst day on record is {worst['violations']} on "
                f"{worst['day'].isoformat()}"
            )

        lead = [heading, self._p(headline + ".")]

        if "risk" in self._charts:
            lead += [
                self._figure(self._charts["risk"]),
                self._p(
                    "Grey is what was logged each day; blue is what the "
                    "model predicts for that day from the previous logged "
                    "day, with tomorrow's forecast at the end. This is an "
                    "in-sample fit (the model was trained on these days), "
                    "so it shows what the model has learned rather than "
                    "how well it will generalise.",
                    "caption",
                ),
            ]

        elements: list = [KeepTogether(lead)]

        latest = risk.get("latest")
        labels = risk.get("feature_labels") or {}

        if latest:
            names = [
                name
                for name in (risk.get("feature_names") or latest)
                if name in latest and name != "day"
            ]

            rows = [
                [labels.get(name, name), self._fmt_feature(name, latest[name])]
                for name in names
            ]

            elements.append(
                KeepTogether(
                    [
                        self._p("What the model saw today", "h3"),
                        self._table(
                            [
                                f"Model input ({latest['day'].isoformat()})",
                                "Value",
                            ],
                            rows,
                            col_widths=[7 * cm, 3 * cm],
                        ),
                    ]
                )
            )

        if "importance" in self._charts:
            elements.append(
                KeepTogether(
                    [
                        Spacer(1, 0.2 * cm),
                        self._figure(self._charts["importance"]),
                    ]
                )
            )

        notes = []

        if risk.get("training_days"):
            notes.append(f"trained on {risk['training_days']} days")

        if risk.get("trained_at"):
            notes.append(f"last trained {risk['trained_at']:%Y-%m-%d %H:%M}")

        if risk.get("mean_absolute_error") is not None:
            notes.append(
                "in-sample mean absolute error "
                f"{risk['mean_absolute_error']:.1f} violations per day"
            )

        if notes:
            elements.append(
                self._p(
                    "Model: gradient-boosted regressor, "
                    + ", ".join(notes)
                    + ".",
                    "caption",
                )
            )

        elements.append(description)

        return elements

    @staticmethod
    def _fmt_feature(name: str, value) -> str:
        if name == "dow":
            return calendar.day_name[int(value)]

        if name in ("crew", "viol"):
            return f"{int(value)}"

        if name == "hour":
            hours = int(value)
            minutes = int(round((float(value) - hours) * 60))
            return f"{hours:02d}:{minutes:02d}"

        return f"{float(value):.2f}"

    def _classes(self) -> list:
        a = self._analysis
        heading = self._p("Violations by PPE class", "h2")

        if "classes" not in self._charts:
            return [
                heading,
                self._p("No violations were logged in this period."),
            ]

        rows = []

        for entry in a["classes"]:
            rows.append(
                [
                    f"missing {entry['name']}",
                    f"{entry['count']}",
                    _fmt_pct(entry["share"]),
                    _fmt_pct(entry["missing_rate"]),
                    (
                        f"{entry['prior_count']}"
                        if a["prior_available"]
                        else "-"
                    ),
                    _fmt_pct(entry["prior_missing_rate"]),
                    _fmt_signed_pct(entry["change_pct"]),
                    f"{self._severity[entry['name']]:.0f}",
                ]
            )

        current = a["current"]
        prior = a["prior"]

        rows.append(
            [
                "all violations",
                f"{current['violations']}",
                "100%",
                f"{current['rate']:.2f} per worker",
                f"{prior['violations']}" if a["prior_available"] else "-",
                (
                    f"{prior['rate']:.2f} per worker"
                    if a["prior_available"] and prior["workers"]
                    else "-"
                ),
                _fmt_signed_pct(a["violations_change_pct"]),
                "",
            ]
        )

        table = self._table(
            [
                "PPE item",
                "Violations",
                "Share",
                "Sightings missing it",
                "Prior violations",
                "Prior sightings",
                "Change",
                "Weight",
            ],
            rows,
        )

        return [
            KeepTogether(
                [
                    heading,
                    self._figure(self._charts["classes"]),
                    self._p(
                        "How often each item was missing from a worker "
                        "sighting this period (bar) and in the prior "
                        "period (grey marker). A sighting can carry up to "
                        "three violations, one per item.",
                        "caption",
                    ),
                    table,
                ]
            )
        ]

    def _cameras(self) -> list:
        a = self._analysis
        heading = self._p("Cameras", "h2")
        elements: list = []

        if "cameras" in self._charts:
            rows = [
                [
                    f"camera {camera['camera_id']}",
                    f"{camera['days_active']}",
                    f"{camera['workers']}",
                    f"{camera['violations']}",
                    f"{camera['rate']:.2f}",
                    _fmt_pct(camera["share"]),
                    _fmt_pct(camera["compliance_rate"]),
                    _fmt_score(camera["score"]),
                    _fmt_score(camera["score_today"]),
                ]
                for camera in a["cameras"]
            ]

            table = self._table(
                [
                    "Camera",
                    "Days active",
                    "Workers",
                    "Violations",
                    "Per worker",
                    "Share",
                    "Fully compliant",
                    "Score, period",
                    "Score, today",
                ],
                rows,
            )

            elements.append(
                KeepTogether(
                    [
                        heading,
                        self._figure(self._charts["cameras"]),
                        self._p(
                            "Each camera is scored as its own project. "
                            "'Per worker' is violations divided by worker "
                            "sightings, which makes cameras with different "
                            "crew sizes comparable.",
                            "caption",
                        ),
                        table,
                    ]
                )
            )
        else:
            elements += [
                heading,
                self._p("No camera logged detections in this period."),
            ]

        elements.append(KeepTogether(self._repeat_flags()))

        return elements

    def _repeat_flags(self) -> list:
        a = self._analysis
        flags = a["repeat_flags"]
        threshold = a["repeat_threshold"]

        elements = [self._p("Repeated non-compliance", "h3")]

        threshold_text = (
            f"{threshold['min_violations']} or more violations of the same "
            f"item on the same camera within {threshold['window_days']} "
            "days"
            if threshold
            else "the configured threshold"
        )

        if not flags:
            elements.append(
                self._p(
                    f"No camera reached the flag threshold "
                    f"({threshold_text})."
                )
            )
            return elements

        rows = [
            [
                f"camera {flag['camera_id']}",
                f"missing {flag['class']}",
                f"{flag['violations']}",
                f"last {flag['window_days']} days",
            ]
            for flag in flags
        ]

        elements.append(
            self._p(
                f"Flagged where {threshold_text}. These are the "
                "camera/PPE combinations to follow up on first.",
                "caption",
            )
        )
        elements.append(
            self._table(
                ["Camera", "PPE item", "Violations", "Window"],
                rows,
                numeric_from=2,
            )
        )

        return elements

    def _weekdays(self) -> list:
        a = self._analysis

        if "weekdays" not in self._charts:
            return [
                self._p("Weekday pattern", "h2"),
                self._p(
                    "The weekday pattern appears once "
                    f"{MIN_DAYS_FOR_WEEKDAY_PATTERN} days of history are "
                    f"logged (currently {a['history_days']})."
                ),
            ]

        rows = [
            [
                entry["name"],
                f"{entry['days']}",
                f"{entry['workers']}",
                f"{entry['violations']}",
                f"{entry['rate']:.2f}" if entry["rate"] is not None else "-",
            ]
            for entry in a["weekdays"]
        ]

        return [
            KeepTogether(
                [
                    self._p("Weekday pattern", "h2"),
                    self._figure(self._charts["weekdays"]),
                    self._p(
                        f"Violations per worker by weekday over the last "
                        f"{a['history_days']} logged days. Read weekdays "
                        "with only one or two logged days as indicative.",
                        "caption",
                    ),
                    self._table(
                        [
                            "Weekday",
                            "Days logged",
                            "Workers",
                            "Violations",
                            "Per worker",
                        ],
                        rows,
                    ),
                ]
            ),
        ]

    def _heatmap_section(self) -> list:
        if not self._heatmaps:
            return []

        a = self._analysis
        image_width = (self._width - 0.4 * cm) / 2

        cells = []

        for camera_id, path in self._heatmaps:
            cells.append(
                [
                    self._image_file(path, image_width),
                    self._p(
                        f"camera {camera_id}, last {a['days']} days",
                        "caption",
                    ),
                ]
            )

        rows = [cells[i : i + 2] for i in range(0, len(cells), 2)]

        if len(rows[-1]) == 1:
            rows[-1].append("")

        table = Table(
            rows,
            colWidths=[image_width, image_width],
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0.4 * cm),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        return [
            KeepTogether(
                [
                    self._p("Where violations happen", "h2"),
                    self._p(
                        "Density of non-compliant workers over the period, "
                        "drawn on the latest frame from each camera. Hot "
                        "spots are where workers without full PPE stood "
                        "most often.",
                        "caption",
                    ),
                    table,
                ]
            )
        ]

    def _daily_table(self) -> list:
        a = self._analysis

        rows = []

        for row in a["rows"]:
            workers = int(row["workers"] or 0)

            rows.append(
                [
                    str(row["day"]),
                    str(row["camera_id"]),
                    f"{workers}",
                    f"{int(row.get('compliant') or 0)}",
                    f"{int(row['helmet_viol'] or 0)}",
                    f"{int(row['vest_viol'] or 0)}",
                    f"{int(row['boots_viol'] or 0)}",
                    _fmt_score(
                        weighted_score(
                            workers,
                            int(row["helmet_viol"] or 0),
                            int(row["vest_viol"] or 0),
                            int(row["boots_viol"] or 0),
                            self._severity,
                        )
                        if workers
                        else None
                    ),
                ]
            )

        if not rows:
            rows.append(["-", "-", "0", "0", "0", "0", "0", "-"])

        table = self._table(
            [
                "Day",
                "Camera",
                "Workers",
                "Compliant",
                "Missing helmet",
                "Missing vest",
                "Missing boots",
                "Score",
            ],
            rows,
            numeric_from=2,
        )

        caption = self._p(
            "One row per day and camera. 'Compliant' counts sightings "
            "with helmet, vest and boots all present.",
            "caption",
        )

        # The heading moves to a fresh page with its table rather than
        # sitting orphaned; a table longer than a page still splits,
        # repeating its header row.
        return [
            KeepTogether([self._p("Daily breakdown", "h2"), caption, table]),
        ]
