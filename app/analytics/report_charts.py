"""
Chart rendering for the PPE compliance report.

matplotlib on the Agg backend, returning PNG bytes for embedding in
the PDF. Colour roles are fixed: one hue per PPE class (helmet blue,
vest orange, boots aqua), faded versions of the same hues for the
prior period, recessive hairline grid and axes. Text never wears a
series colour.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from app.analytics.report_analysis import (  # noqa: E402
    MIN_DAYS_FOR_WEEKDAY_PATTERN,
    PPE_CLASSES,
)

CLASS_COLORS = {
    "helmet": "#2a78d6",
    "vest": "#eb6834",
    "boots": "#1baf7a",
}

ACCENT = "#2a78d6"
CONTEXT = "#b9b8b1"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"

# Figure width in inches: the PDF text column is 17 cm.
FIGURE_WIDTH = 6.7
DPI = 200

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": 7.5,
    "axes.titlecolor": INK,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "legend.handlelength": 1.0,
    "legend.handleheight": 0.8,
    "legend.labelcolor": INK_SECONDARY,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
}


def _fade(color: str, keep: float = 0.4) -> tuple[float, float, float]:
    """Mix a colour towards the surface; keep=1 is the full colour."""

    r, g, b = to_rgb(color)

    return (
        1.0 - keep * (1.0 - r),
        1.0 - keep * (1.0 - g),
        1.0 - keep * (1.0 - b),
    )


def _render(fig) -> bytes:
    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.06,
    )

    plt.close(fig)

    return buffer.getvalue()


def _date_axis(ax, first: date, last: date) -> None:
    span = (last - first).days + 1

    ax.set_xlim(
        mdates.date2num(first) - 0.7,
        mdates.date2num(last) + 0.7,
    )

    if span <= 10:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    elif span <= 20:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    else:
        ax.xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=4, maxticks=10)
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def _integer_ticks(ax) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))


def _label(ax, x, y, text: str, dy: float = 3) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
        color=INK_SECONDARY,
    )


# ======================================================================
# Charts
# ======================================================================


def daily_violations_chart(
    prior: list[dict],
    current: list[dict],
    window_start: date,
) -> bytes:
    """
    Stacked columns of violations per day by PPE class (prior period
    faded), with workers seen per day in a small panel underneath so
    counts can be read against crew size.
    """

    series = list(prior) + list(current)
    n_prior = len(prior)

    with plt.rc_context(RC):
        fig, (ax_v, ax_w) = plt.subplots(
            2,
            1,
            figsize=(FIGURE_WIDTH, 3.9),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.22},
        )

        xs = [mdates.date2num(bucket["day"]) for bucket in series]
        width = 0.62

        totals_ = [0.0] * len(series)

        for ppe_class in PPE_CLASSES:
            for index, bucket in enumerate(series):
                value = bucket[ppe_class] or 0

                if value <= 0:
                    continue

                color = (
                    _fade(CLASS_COLORS[ppe_class])
                    if index < n_prior
                    else CLASS_COLORS[ppe_class]
                )

                ax_v.bar(
                    xs[index],
                    value,
                    width,
                    bottom=totals_[index],
                    color=color,
                    edgecolor=SURFACE,
                    linewidth=0.8,
                )

                totals_[index] += value

        peak = max(totals_) if totals_ else 0.0

        ax_v.set_ylim(0, (peak or 1.0) * 1.22)
        _integer_ticks(ax_v)

        # Label only the peak of the report period and its last
        # logged day; the table carries the rest.
        current_indices = [
            i for i in range(n_prior, len(series)) if totals_[i] > 0
        ]

        if current_indices:
            peak_index = max(current_indices, key=lambda i: totals_[i])
            last_index = current_indices[-1]

            for index in {peak_index, last_index}:
                _label(ax_v, xs[index], totals_[index], f"{int(totals_[index])}")

        if n_prior:
            boundary = mdates.date2num(window_start) - 0.5

            for ax in (ax_v, ax_w):
                ax.axvline(boundary, color=BASELINE, linewidth=0.8)

            ax_v.text(
                boundary - 0.15,
                0.97,
                "prior period",
                transform=ax_v.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=7,
                color=MUTED,
            )
            ax_v.text(
                boundary + 0.15,
                0.97,
                "report period",
                transform=ax_v.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=7,
                color=MUTED,
            )

        ax_v.set_title("Violations per day by PPE class", pad=20)

        ax_v.legend(
            handles=[
                Patch(color=CLASS_COLORS[c], label=f"missing {c}")
                for c in PPE_CLASSES
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=3,
            borderaxespad=0.0,
        )

        # Workers panel.
        worker_values = [bucket["workers"] or 0 for bucket in series]

        for index, value in enumerate(worker_values):
            if value <= 0:
                continue

            ax_w.bar(
                xs[index],
                value,
                width,
                color=_fade(MUTED, 0.55) if index < n_prior else MUTED,
                edgecolor=SURFACE,
                linewidth=0.8,
            )

        worker_peak = max(worker_values) if worker_values else 0

        ax_w.set_ylim(0, (worker_peak or 1) * 1.35)
        _integer_ticks(ax_w)
        ax_w.set_title("Workers seen per day", pad=4)

        if worker_peak:
            peak_index = max(
                range(len(worker_values)), key=lambda i: worker_values[i]
            )
            _label(ax_w, xs[peak_index], worker_peak, f"{worker_peak}")

        _date_axis(ax_w, series[0]["day"], series[-1]["day"])

        return _render(fig)


def score_trend_chart(
    history: list[dict],
    window_start: date,
    period_average: float | None = None,
) -> bytes:
    """
    Daily safety score across the long-range window. Days before the
    report period are grey context; the report period is the accent
    line inside a light wash.
    """

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.5))

        xs = [mdates.date2num(bucket["day"]) for bucket in history]
        scores = [
            bucket["score"] if bucket["workers"] else float("nan")
            for bucket in history
        ]

        first_current = next(
            (
                i
                for i, bucket in enumerate(history)
                if bucket["day"] >= window_start
            ),
            len(history),
        )

        ax.axvspan(
            mdates.date2num(window_start) - 0.5,
            xs[-1] + 0.7,
            color=ACCENT,
            alpha=0.06,
            linewidth=0,
        )

        marker_style = {
            "marker": "o",
            "markersize": 4.5,
            "markeredgecolor": SURFACE,
            "markeredgewidth": 1.0,
            "linewidth": 2,
            "solid_capstyle": "round",
            "solid_joinstyle": "round",
        }

        # The grey series includes the first report-period point so
        # the two segments join.
        grey = [
            score if i <= first_current else float("nan")
            for i, score in enumerate(scores)
        ]
        accent = [
            score if i >= first_current else float("nan")
            for i, score in enumerate(scores)
        ]

        ax.plot(xs, grey, color=CONTEXT, **marker_style)
        ax.plot(xs, accent, color=ACCENT, **marker_style)

        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])

        handles = [
            Line2D([], [], color=ACCENT, linewidth=2, label="report period"),
            Line2D([], [], color=CONTEXT, linewidth=2, label="earlier days"),
        ]

        if period_average is not None:
            ax.axhline(period_average, color=BASELINE, linewidth=0.8)
            handles.append(
                Line2D(
                    [],
                    [],
                    color=BASELINE,
                    linewidth=1,
                    label=f"period average {period_average:.0f}",
                )
            )

        logged = [i for i, score in enumerate(scores) if score == score]

        if logged:
            last = logged[-1]
            _label(ax, xs[last], scores[last], f"{scores[last]:.0f}", dy=6)

        ax.set_title(
            "Daily safety score (100 = no violations, 0 = every worker "
            "missing everything)",
            pad=16,
        )

        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=3,
            borderaxespad=0.0,
        )

        _date_axis(ax, history[0]["day"], history[-1]["day"])

        return _render(fig)


def class_chart(classes: list[dict]) -> bytes:
    """
    Share of worker sightings missing each PPE item this period
    (bars), with the prior period as a grey marker.
    """

    with plt.rc_context(RC):
        fig, ax = plt.subplots(
            figsize=(FIGURE_WIDTH, 0.75 + 0.5 * len(classes))
        )

        ax.grid(False, axis="y")
        ax.grid(True, axis="x")

        positions = list(range(len(classes)))[::-1]
        has_prior = False

        for position, entry in zip(positions, classes):
            rate = 100.0 * (entry["missing_rate"] or 0.0)

            ax.barh(
                position,
                rate,
                height=0.5,
                color=CLASS_COLORS[entry["name"]],
                edgecolor=SURFACE,
                linewidth=0.8,
            )

            ax.annotate(
                f"{rate:.0f}%  ({entry['count']})",
                (rate, position),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
                color=INK_SECONDARY,
            )

            if entry["prior_missing_rate"] is not None:
                has_prior = True

                ax.plot(
                    100.0 * entry["prior_missing_rate"],
                    position,
                    marker="o",
                    markersize=7,
                    color=CONTEXT,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.2,
                    linestyle="none",
                )

        ax.set_yticks(positions)
        ax.set_yticklabels([f"missing {e['name']}" for e in classes])
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

        ax.set_title("Share of worker sightings missing each item", pad=16)

        handles = [Patch(color=INK_SECONDARY, label="this period")]

        if has_prior:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    markersize=6,
                    color=CONTEXT,
                    linestyle="none",
                    label="prior period",
                )
            )

        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=2,
            borderaxespad=0.0,
        )

        return _render(fig)


def camera_chart(cameras: list[dict]) -> bytes:
    """Violations per camera, stacked by PPE class."""

    with plt.rc_context(RC):
        fig, ax = plt.subplots(
            figsize=(FIGURE_WIDTH, 0.85 + 0.45 * len(cameras))
        )

        ax.grid(False, axis="y")
        ax.grid(True, axis="x")

        positions = list(range(len(cameras)))[::-1]
        peak = max((c["violations"] for c in cameras), default=0)

        for position, camera in zip(positions, cameras):
            left = 0.0

            for ppe_class in PPE_CLASSES:
                value = camera[ppe_class]

                if value <= 0:
                    continue

                ax.barh(
                    position,
                    value,
                    height=0.5,
                    left=left,
                    color=CLASS_COLORS[ppe_class],
                    edgecolor=SURFACE,
                    linewidth=0.8,
                )

                left += value

            ax.annotate(
                f"{camera['violations']}  ({camera['rate']:.2f} per worker)",
                (left, position),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
                color=INK_SECONDARY,
            )

        ax.set_yticks(positions)
        ax.set_yticklabels([f"camera {c['camera_id']}" for c in cameras])
        ax.set_xlim(0, (peak or 1) * 1.35)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

        ax.set_title("Violations by camera and PPE class", pad=16)

        ax.legend(
            handles=[
                Patch(color=CLASS_COLORS[c], label=f"missing {c}")
                for c in PPE_CLASSES
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=3,
            borderaxespad=0.0,
        )

        return _render(fig)


def weekday_chart(weekdays: list[dict]) -> bytes:
    """Violations per worker by weekday over the logged history."""

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.2))

        rates = [w["rate"] or 0.0 for w in weekdays]
        peak = max(rates) if rates else 0.0

        for index, entry in enumerate(weekdays):
            if entry["rate"] is None:
                continue

            ax.bar(
                index,
                entry["rate"],
                0.55,
                color=ACCENT,
                edgecolor=SURFACE,
                linewidth=0.8,
            )

        if peak:
            peak_index = rates.index(peak)
            _label(ax, peak_index, peak, f"{peak:.2f}")

        ax.set_xticks(range(len(weekdays)))
        ax.set_xticklabels(
            [
                f"{w['name']}\n{w['days']} day{'s' if w['days'] != 1 else ''}"
                if w["days"]
                else f"{w['name']}\nno data"
                for w in weekdays
            ]
        )
        ax.set_ylim(0, (peak or 1.0) * 1.25)
        ax.set_ylabel("violations per worker")

        ax.set_title("Violations per worker by weekday (all logged history)")

        return _render(fig)


def risk_chart(
    backtest: list[dict],
    forecast_day: date,
    predicted: float,
    worst_day: dict | None,
) -> bytes:
    """
    Actual daily violations against the model's in-sample
    prediction for each day, extended with tomorrow's forecast.
    """

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.6))

        xs = [mdates.date2num(entry["day"]) for entry in backtest]
        actual = [entry["actual"] for entry in backtest]
        fitted = [entry["predicted"] for entry in backtest]

        marker_style = {
            "marker": "o",
            "markersize": 4,
            "markeredgecolor": SURFACE,
            "markeredgewidth": 1.0,
            "linewidth": 2,
            "solid_capstyle": "round",
            "solid_joinstyle": "round",
        }

        ax.plot(xs, actual, color=MUTED, **marker_style)
        ax.plot(xs, fitted, color=ACCENT, **marker_style)

        forecast_x = mdates.date2num(forecast_day)

        ax.plot(
            [xs[-1], forecast_x],
            [fitted[-1], predicted],
            color=ACCENT,
            linewidth=1.5,
            linestyle=(0, (3, 3)),
        )
        ax.plot(
            forecast_x,
            predicted,
            marker="o",
            markersize=7,
            color=SURFACE,
            markeredgecolor=ACCENT,
            markeredgewidth=2,
            linestyle="none",
        )
        _label(ax, forecast_x, predicted, f"{predicted:.0f}", dy=7)

        handles = [
            Line2D([], [], color=MUTED, linewidth=2, label="actual"),
            Line2D([], [], color=ACCENT, linewidth=2, label="model fit"),
            Line2D(
                [],
                [],
                marker="o",
                markersize=6,
                color=SURFACE,
                markeredgecolor=ACCENT,
                markeredgewidth=1.5,
                linestyle="none",
                label="forecast for tomorrow",
            ),
        ]

        ceiling = max(max(actual), max(fitted), predicted, 1.0)

        if worst_day:
            ax.axhline(worst_day["violations"], color=BASELINE, linewidth=0.8)
            ceiling = max(ceiling, worst_day["violations"])
            handles.append(
                Line2D(
                    [],
                    [],
                    color=BASELINE,
                    linewidth=1,
                    label=f"worst day on record ({worst_day['violations']})",
                )
            )

        ax.set_ylim(0, ceiling * 1.25)
        _integer_ticks(ax)

        ax.set_title("Model fit and forecast: violations per day", pad=16)

        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=4,
            borderaxespad=0.0,
        )

        _date_axis(ax, backtest[0]["day"], forecast_day)

        return _render(fig)


def feature_importance_chart(
    importances: dict[str, float],
    labels: dict[str, str],
) -> bytes:
    """Relative importance of each model input, largest first."""

    items = sorted(importances.items(), key=lambda item: item[1])

    with plt.rc_context(RC):
        fig, ax = plt.subplots(
            figsize=(FIGURE_WIDTH, 0.7 + 0.32 * len(items))
        )

        ax.grid(False, axis="y")
        ax.grid(True, axis="x")

        positions = list(range(len(items)))
        values = [100.0 * value for _, value in items]

        ax.barh(
            positions,
            values,
            height=0.5,
            color=ACCENT,
            edgecolor=SURFACE,
            linewidth=0.8,
        )

        for position, value in zip(positions, values):
            ax.annotate(
                f"{value:.0f}%",
                (value, position),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
                color=INK_SECONDARY,
            )

        ax.set_yticks(positions)
        ax.set_yticklabels([labels.get(name, name) for name, _ in items])
        ax.set_xlim(0, (max(values) if values else 1.0) * 1.25)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
        )

        ax.set_title("What drives the forecast (relative importance)")

        return _render(fig)


def build_charts(analysis: dict) -> dict[str, bytes]:
    """Render every chart the analysis has data for."""

    charts: dict[str, bytes] = {}

    risk = analysis.get("risk") or {}

    if analysis.get("risk_available"):
        # Cap the fit chart at two months so a long history stays legible.
        backtest = risk.get("backtest") or []

        if backtest:
            charts["risk"] = risk_chart(
                backtest[-60:],
                analysis["today"] + timedelta(days=1),
                risk["predicted_violations"],
                risk.get("worst_day"),
            )

        if risk.get("feature_importances"):
            charts["importance"] = feature_importance_chart(
                risk["feature_importances"],
                risk.get("feature_labels") or {},
            )

    any_history = any(
        bucket["workers"] for bucket in analysis["per_day_history"]
    )

    if any_history:
        charts["daily"] = daily_violations_chart(
            analysis["per_day_prior"],
            analysis["per_day"],
            analysis["window_start"],
        )
        charts["score"] = score_trend_chart(
            analysis["per_day_history"],
            analysis["window_start"],
            analysis["current"]["score"],
        )

    if analysis["current"]["workers"]:
        charts["classes"] = class_chart(analysis["classes"])
        charts["cameras"] = camera_chart(analysis["cameras"])

    if analysis["history_days"] >= MIN_DAYS_FOR_WEEKDAY_PATTERN:
        charts["weekdays"] = weekday_chart(analysis["weekdays"])

    return charts
