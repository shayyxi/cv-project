from pathlib import Path

import yaml


def load_analytics_config() -> dict:
    """
    Load the Phase 2 analytics configuration
    (severity weights, flagging and window defaults).
    """

    config_path = (
        Path(__file__).resolve().parent
        / "config"
        / "analytics_config.yaml"
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)
