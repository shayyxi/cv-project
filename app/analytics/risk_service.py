import logging
from datetime import date

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from app.analytics.analytics_repository import AnalyticsRepository
from app.config import settings

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["crew", "viol", "rate", "dow", "hour", "rate_7d"]

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

        features = build_daily_features(
            self._repository.daily_stats(),
            mean_hours=self._repository.daily_mean_hours(),
        )

        training = [f for f in features if "next_viol" in f]

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

        if self._model_path.exists():
            model = joblib.load(self._model_path)
        else:
            model = self.train(min_days)

        if model is None:
            return None

        features = build_daily_features(
            self._repository.daily_stats(),
            mean_hours=self._repository.daily_mean_hours(),
        )

        if not features:
            return None

        latest = features[-1]

        x = np.array(
            [[latest[name] for name in FEATURE_NAMES]]
        )

        predicted = float(model.predict(x)[0])

        worst_day = max(
            (row["viol"] for row in features),
            default=1,
        )

        risk_score = round(
            min(
                100.0,
                100.0 * predicted / max(float(worst_day), 1.0),
            ),
            1,
        )

        logger.info(
            "Predicted violations tomorrow: %.1f -> risk score %s/100",
            predicted,
            risk_score,
        )

        return {
            "predicted_violations": round(predicted, 1),
            "risk_score": risk_score,
        }
