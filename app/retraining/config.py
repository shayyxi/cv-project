"""
Roboflow settings for the retraining loop.

Read from real environment variables first, then the repo's .env (the
same file app.config reads). Kept out of app.config so the pipeline's
settings stay untouched.

    ROBOFLOW_API_KEY               required
    ROBOFLOW_PROJECT               required; project slug (last part of the project URL)
    ROBOFLOW_WORKSPACE             optional; workspace slug (default: the key's workspace)
    ROBOFLOW_UPLOAD_AS_PREDICTION  default true; pre-labels are stored as model
                                   predictions to review in Roboflow. false makes
                                   them count as ground truth right away.
    ROBOFLOW_VALID_SPLIT_PERCENT   default 20; share of uploads sent to the "valid"
                                   split (0-100), the rest go to "train"
    ROBOFLOW_BATCH_PREFIX          default "label-queue"; uploads are batched as
                                   "<prefix>-YYYY-MM-DD"
"""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.config import BASE_DIR, settings

logger = logging.getLogger(__name__)

# Real env vars win over .env, like pydantic-settings in app.config.
load_dotenv(BASE_DIR / ".env")

# Same folder LabelingService exports to.
LABEL_QUEUE_DIR = settings.local_analytics_dir / "label_queue"

DATASETS_DIR = settings.local_analytics_dir / "datasets"

TRAINING_DIR = settings.local_analytics_dir / "training"

HOW_TO_CONFIGURE = (
    "set ROBOFLOW_API_KEY and ROBOFLOW_PROJECT (optionally "
    "ROBOFLOW_WORKSPACE) in .env"
)


class RoboflowNotConfigured(RuntimeError):
    """Roboflow credentials are missing or the SDK is not installed."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)

    if value is None or value.strip() == "":
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)

    if value is None or value.strip() == "":
        return default

    try:
        return int(value)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer - using %d", name, value, default
        )
        return default


@dataclass(frozen=True)
class RoboflowSettings:
    api_key: str = ""
    project: str = ""
    workspace: str = ""
    upload_as_prediction: bool = True
    valid_split_percent: int = 20
    batch_prefix: str = "label-queue"

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.project)

    @property
    def project_name(self) -> str:
        """"<workspace>/<project>", or just the project slug."""

        if self.workspace:
            return f"{self.workspace}/{self.project}"

        return self.project

    @classmethod
    def from_env(cls) -> "RoboflowSettings":
        return cls(
            api_key=os.environ.get("ROBOFLOW_API_KEY", "").strip(),
            project=os.environ.get("ROBOFLOW_PROJECT", "").strip(),
            workspace=os.environ.get("ROBOFLOW_WORKSPACE", "").strip(),
            upload_as_prediction=_env_bool(
                "ROBOFLOW_UPLOAD_AS_PREDICTION", True
            ),
            valid_split_percent=_env_int(
                "ROBOFLOW_VALID_SPLIT_PERCENT", 20
            ),
            batch_prefix=(
                os.environ.get("ROBOFLOW_BATCH_PREFIX", "").strip()
                or "label-queue"
            ),
        )


def connect_project(config: RoboflowSettings | None = None):
    """
    The Roboflow Project object for the configured workspace and
    project. The SDK is imported lazily so nothing else needs it.
    """

    config = config or RoboflowSettings.from_env()

    if not config.configured:
        raise RoboflowNotConfigured(
            f"Roboflow is not configured; {HOW_TO_CONFIGURE}"
        )

    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise RoboflowNotConfigured(
            "The roboflow package is not installed; run "
            "`pip install roboflow`"
        ) from error

    client = Roboflow(api_key=config.api_key)

    # None selects the API key's default workspace.
    workspace = client.workspace(config.workspace or None)

    project = workspace.project(config.project)

    logger.info(
        "Connected to Roboflow project %s",
        getattr(project, "id", config.project_name),
    )

    return project
