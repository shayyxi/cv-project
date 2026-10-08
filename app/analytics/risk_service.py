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

# Logged days a camera needs before it gets its own forecast. Matches
# the rate_7d window, so the trailing rate is a real seven-day mean.
CAMERA_MIN_DAYS = 7

# Bumped whenever a saved model is no longer comparable with what
# train() produces (version 2: per-camera rows instead of site-wide
# totals). _load_or_train retrains when the file carries anything else.
MODEL_VERSION = 2


def build_camera_features(
    rows: list[dict],
    mean_hours: dict[tuple[str, date], float] | None = None,
) -> dict[str, list[dict]]:
    """
    Turn daily_stats rows (one per day per camera) into one feature
    series per camera, keyed by camera_id and sorted by day:

        crew     - workers seen on that camera
        viol     - total violations on that camera
        rate     - violations per worker
        dow      - day of week (0 = Monday)
        hour     - mean capture hour of the camera's detections that
                   day (time-of-day; noon when unknown)
        rate_7d  - mean rate over the camera's trailing 7 logged days

    Every day except the camera's last also gets next_viol, the
    camera's violation count on its next logged day (the training
    target). mean_hours is keyed (camera_id, day).
    """

    mean_hours = mean_hours or {}

    per_camera: dict[str, dict[date, dict]] = {}

    for row in rows:
        camera_id = str(row["camera_id"])

        day = per_camera.setdefault(camera_id, {}).setdefault(
            row["day"],
            {"workers": 0, "viol": 0},
        )

        day["workers"] += row["workers"]
        day["viol"] += (
            row["helmet_viol"]
            + row["vest_viol"]
            + row["boots_viol"]
        )

    return {
        camera_id: _feature_series(camera_id, per_camera[camera_id], mean_hours)
        for camera_id in sorted(per_camera)
    }


def _feature_series(
    camera_id: str,
    per_day: dict[date, dict],
    mean_hours: dict[tuple[str, date], float],
) -> list[dict]:
    features = []
    rates: list[float] = []

    for day in sorted(per_day):
        info = per_day[day]

        rate = info["viol"] / max(info["workers"], 1)

        rates.append(rate)

        window = rates[-7:]

        features.append(
            {
                "camera_id": camera_id,
                "day": day,
                "crew": info["workers"],
                "viol": info["viol"],
                "rate": rate,
                "dow": day.weekday(),
                "hour": mean_hours.get((camera_id, day), DEFAULT_MEAN_HOUR),
                "rate_7d": sum(window) / len(window),
            }
        )

    for index in range(len(features) - 1):
        features[index]["next_viol"] = features[index + 1]["viol"]

    return features


class RiskService:
    """
    Predictive risk scoring per camera: one gradient-boosted model is
    trained on every camera's daily history and applied to each
    camera's latest logged day to predict its violations tomorrow,
    scaled into a 0-100 risk score against that camera's worst day.
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
        Train and persist the model on every camera's rows. Returns
        None (and logs) when fewer than min_days distinct days have a
        known next-day target.
        """

        training = self._training_rows(self._features())
        training_days = self._distinct_days(training)

        if training_days < min_days:
            logger.info(
                "Risk model needs >= %d days of history, have %d "
                "- keep logging.",
                min_days,
                training_days,
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

        joblib.dump(
            {"version": MODEL_VERSION, "model": model},
            self._model_path,
        )

        logger.info(
            "Risk model trained on %d days (%d camera-days) -> %s",
            training_days,
            len(training),
            self._model_path,
        )

        return model

    def score(
        self,
        min_days: int = 14,
    ) -> dict[str, dict] | None:
        """
        Predict tomorrow's violations for every camera with enough
        history.

        Returns {camera_id: {"predicted_violations", "risk_score"}}
        (empty when the model exists but no camera has CAMERA_MIN_DAYS
        logged days yet) or None when no model can be trained yet.
        """

        details = self.details(min_days)

        if not details["available"]:
            return None

        scores: dict[str, dict] = {}

        for camera in details["cameras"]:
            if not camera["available"]:
                logger.info(
                    "Camera %s has %d of the %d logged days needed for "
                    "a forecast.",
                    camera["camera_id"],
                    camera["history_days"],
                    CAMERA_MIN_DAYS,
                )
                continue

            logger.info(
                "Camera %s predicted violations tomorrow: %.1f "
                "-> risk score %s/100",
                camera["camera_id"],
                camera["predicted_violations"],
                camera["risk_score"],
            )

            scores[camera["camera_id"]] = {
                "predicted_violations": camera["predicted_violations"],
                "risk_score": camera["risk_score"],
            }

        return scores

    def details(
        self,
        min_days: int = 14,
    ) -> dict:
        """
        Everything the compliance report shows about the model:
        whether it is available, how much history it has, the
        relative feature importances, and per camera the forecast
        for tomorrow, the latest day's inputs and the in-sample fit
        over that camera's history (each day's actual violations
        against what the model predicts from the previous logged
        day).

        Always returns a dict; "available" is False until enough
        history exists to train. "cameras" lists every camera with
        history, sorted by camera_id, each flagged "available" only
        once it has CAMERA_MIN_DAYS logged days. "top_camera" is the
        available camera with the highest risk score, or None.

        Cameras come from the detection history, not from
        settings.camera_ids: a configured camera with no detections
        yet has nothing to forecast and does not appear.
        """

        features = self._features()
        training = self._training_rows(features)

        details: dict = {
            "available": False,
            "min_days": min_days,
            "camera_min_days": CAMERA_MIN_DAYS,
            "history_days": self._distinct_days(
                [row for series in features.values() for row in series]
            ),
            "training_days": self._distinct_days(training),
            "training_rows": len(training),
            "feature_names": list(FEATURE_NAMES),
            "feature_labels": dict(FEATURE_LABELS),
            "trained_at": None,
            "feature_importances": None,
            "mean_absolute_error": None,
            "cameras": [
                self._pending_camera(camera_id, series)
                for camera_id, series in features.items()
            ],
            "top_camera": None,
        }

        model = self._load_or_train(min_days)

        if model is None or not features:
            return details

        cameras = [
            self._camera_forecast(model, camera_id, series)
            for camera_id, series in features.items()
        ]

        ready = [camera for camera in cameras if camera["available"]]

        errors = [
            abs(entry["predicted"] - entry["actual"])
            for camera in ready
            for entry in camera["backtest"]
        ]

        importances = getattr(model, "feature_importances_", None)

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
                "mean_absolute_error": (
                    round(sum(errors) / len(errors), 1) if errors else None
                ),
                "cameras": cameras,
                "top_camera": max(
                    ready,
                    key=lambda camera: (
                        camera["risk_score"],
                        camera["predicted_violations"],
                    ),
                    default=None,
                ),
            }
        )

        return details

    # ==================================================================
    # Internals
    # ==================================================================

    def _features(self) -> dict[str, list[dict]]:
        return build_camera_features(
            self._repository.daily_stats(),
            mean_hours=self._repository.daily_mean_hours(),
        )

    @staticmethod
    def _training_rows(features: dict[str, list[dict]]) -> list[dict]:
        return [
            row
            for series in features.values()
            for row in series
            if "next_viol" in row
        ]

    @staticmethod
    def _distinct_days(rows: list[dict]) -> int:
        return len({row["day"] for row in rows})

    def _load_or_train(
        self,
        min_days: int,
    ) -> GradientBoostingRegressor | None:
        if self._model_path.exists():
            payload = joblib.load(self._model_path)

            if (
                isinstance(payload, dict)
                and payload.get("version") == MODEL_VERSION
            ):
                return payload["model"]

            logger.info(
                "Saved risk model at %s predates the per-camera format; "
                "retraining.",
                self._model_path,
            )

        return self.train(min_days)

    def _camera_forecast(
        self,
        model: GradientBoostingRegressor,
        camera_id: str,
        series: list[dict],
    ) -> dict:
        forecast = self._pending_camera(camera_id, series)

        if len(series) < CAMERA_MIN_DAYS:
            return forecast

        predictions = self._predict(model, series)

        worst = max(series, key=lambda row: row["viol"])

        backtest = [
            {
                "day": series[index + 1]["day"],
                "actual": int(series[index + 1]["viol"]),
                "predicted": max(predictions[index], 0.0),
            }
            for index in range(len(series) - 1)
        ]

        errors = [
            abs(entry["predicted"] - entry["actual"]) for entry in backtest
        ]

        predicted = max(predictions[-1], 0.0)

        forecast.update(
            {
                "available": True,
                "latest": dict(series[-1]),
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

        return forecast

    @staticmethod
    def _pending_camera(camera_id: str, series: list[dict]) -> dict:
        return {
            "camera_id": camera_id,
            "history_days": len(series),
            "last_day": series[-1]["day"] if series else None,
            "available": False,
            "latest": None,
            "predicted_violations": None,
            "risk_score": None,
            "worst_day": None,
            "backtest": [],
            "mean_absolute_error": None,
        }

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
