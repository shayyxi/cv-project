import logging
from datetime import date, datetime

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from app.analytics.analytics_repository import AnalyticsRepository
from app.config import settings

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["crew", "viol", "rate", "dow", "hour", "rate_7d"]

# Plain-language names for the report.
FEATURE_LABELS = {
    "crew": "workers seen",
    "viol": "violations",
    "rate": "violations per worker",
    "dow": "day of week",
    "hour": "mean capture hour",
    "rate_7d": "7-day mean violations per worker",
}

DEFAULT_MEAN_HOUR = 12.0


def build_daily_features(
    rows: list[dict],
    mean_hours: dict[date, float] | None = None,
) -> list[dict]:
    """
    Turn daily_stats rows (one per day per camera) into one feature
    row per day across all cameras:

        crew     - workers seen
        viol     - total violations
        rate     - violations per worker
        dow      - day of week (0 = Monday)
        hour     - mean capture hour of the day's detections
                   (time-of-day; noon when unknown)
        rate_7d  - mean rate over the trailing 7 logged days

    Every day except the last also gets next_viol, the following
    day's violation count (the training target).
    """

    mean_hours = mean_hours or {}

    per_day: dict[date, dict] = {}

    for row in rows:
        day = per_day.setdefault(
            row["day"],
            {"workers": 0, "viol": 0},
        )

        day["workers"] += row["workers"]
        day["viol"] += (
            row["helmet_viol"]
            + row["vest_viol"]
            + row["boots_viol"]
        )

    features = []
    rates: list[float] = []

    for day in sorted(per_day):
        info = per_day[day]

        rate = info["viol"] / max(info["workers"], 1)

        rates.append(rate)

        window = rates[-7:]

        features.append(
            {
                "day": day,
                "crew": info["workers"],
                "viol": info["viol"],
                "rate": rate,
                "dow": day.weekday(),
                "hour": mean_hours.get(day, DEFAULT_MEAN_HOUR),
                "rate_7d": sum(window) / len(window),
            }
        )

    for index in range(len(features) - 1):
        features[index]["next_viol"] = features[index + 1]["viol"]

    return features


class RiskService:
    """
    Predictive risk scoring: a gradient-boosted model trained on the
    detection history predicts tomorrow's violation count, scaled
    into a 0-100 daily risk score against the project's worst day.
    """

    def __init__(
        self,
        repository: AnalyticsRepository,
    ) -> None:
        self._repository = repository

        self._model_path = (
            settings.local_analytics_dir
            / "models"
            / "risk_model.joblib"
        )

    def train(
        self,
        min_days: int = 14,
    ) -> GradientBoostingRegressor | None:
        """
        Train and persist the model. Returns None (and logs) when
        there are fewer than min_days days of history with a known
        next-day target.
        """

        training = [f for f in self._features() if "next_viol" in f]

        if len(training) < min_days:
            logger.info(
                "Risk model needs >= %d days of history, have %d "
                "- keep logging.",
                min_days,
                len(training),
            )
            return None

        x = np.array(
            [[row[name] for name in FEATURE_NAMES] for row in training]
        )

        y = np.array(
            [row["next_viol"] for row in training]
        )

        model = GradientBoostingRegressor(random_state=0).fit(x, y)

        self._model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, self._model_path)

        logger.info(
            "Risk model trained on %d days -> %s",
            len(training),
            self._model_path,
        )

        return model

    def score(
        self,
        min_days: int = 14,
    ) -> dict | None:
        """
        Predict tomorrow's violations from the latest day's features.

        Returns {"predicted_violations", "risk_score"} or None when
        no model can be trained yet.
        """

        model = self._load_or_train(min_days)

        if model is None:
            return None

        features = self._features()

        if not features:
            return None

        predicted = self._predict(model, features)[-1]

        worst_day = max(
            (row["viol"] for row in features),
            default=1,
        )

        risk_score = self._risk_score(predicted, worst_day)

        logger.info(
            "Predicted violations tomorrow: %.1f -> risk score %s/100",
            predicted,
            risk_score,
        )

        return {
            "predicted_violations": round(predicted, 1),
            "risk_score": risk_score,
        }

    def details(
        self,
        min_days: int = 14,
    ) -> dict:
        """
        Everything the compliance report shows about the model:
        whether it is available, how much history it has, the
        forecast for tomorrow, the latest day's inputs, the relative
        feature importances and the in-sample fit over the history
        (each day's actual violations against what the model
        predicts from the previous logged day).

        Always returns a dict; "available" is False until enough
        history exists to train.
        """

        features = self._features()
        training = [f for f in features if "next_viol" in f]

        details: dict = {
            "available": False,
            "min_days": min_days,
            "history_days": len(features),
            "training_days": len(training),
            "feature_names": list(FEATURE_NAMES),
            "feature_labels": dict(FEATURE_LABELS),
            "trained_at": None,
            "feature_importances": None,
            "latest": None,
            "predicted_violations": None,
            "risk_score": None,
            "worst_day": None,
            "backtest": [],
            "mean_absolute_error": None,
        }

        model = self._load_or_train(min_days)

        if model is None or not features:
            return details

        predictions = self._predict(model, features)

        worst = max(features, key=lambda row: row["viol"])

        backtest = [
            {
                "day": features[index + 1]["day"],
                "actual": int(features[index + 1]["viol"]),
                "predicted": max(predictions[index], 0.0),
            }
            for index in range(len(features) - 1)
        ]

        errors = [
            abs(entry["predicted"] - entry["actual"]) for entry in backtest
        ]

        importances = getattr(model, "feature_importances_", None)

        predicted = max(predictions[-1], 0.0)

        details.update(
            {
                "available": True,
                "trained_at": (
                    datetime.fromtimestamp(self._model_path.stat().st_mtime)
                    if self._model_path.exists()
                    else None
                ),
                "feature_importances": (
                    {
                        name: float(value)
                        for name, value in zip(FEATURE_NAMES, importances)
                    }
                    if importances is not None
                    else None
                ),
                "latest": dict(features[-1]),
                "predicted_violations": round(predicted, 1),
                "risk_score": self._risk_score(predicted, worst["viol"]),
                "worst_day": {
                    "day": worst["day"],
                    "violations": int(worst["viol"]),
                },
                "backtest": backtest,
                "mean_absolute_error": (
                    round(sum(errors) / len(errors), 1) if errors else None
                ),
            }
        )

        return details

    # ==================================================================
    # Internals
    # ==================================================================

    def _features(self) -> list[dict]:
        return build_daily_features(
            self._repository.daily_stats(),
            mean_hours=self._repository.daily_mean_hours(),
        )

    def _load_or_train(
        self,
        min_days: int,
    ) -> GradientBoostingRegressor | None:
        if self._model_path.exists():
            return joblib.load(self._model_path)

        return self.train(min_days)

    @staticmethod
    def _predict(
        model: GradientBoostingRegressor,
        features: list[dict],
    ) -> list[float]:
        x = np.array(
            [[row[name] for name in FEATURE_NAMES] for row in features]
        )

        return [float(value) for value in model.predict(x)]

    @staticmethod
    def _risk_score(predicted: float, worst_violations: float) -> float:
        return round(
            min(
                100.0,
                100.0 * predicted / max(float(worst_violations), 1.0),
            ),
            1,
        )
