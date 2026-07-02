from pathlib import Path

from app.config import Settings


def test_settings_load_from_values() -> None:
    settings = Settings(
        database_url="postgresql+psycopg://user:pass@localhost:5432/test",
        wordpress_webhook_url="https://example.com/webhook",
        wordpress_api_key="secret",
    )

    assert settings.app_env == "development"
    assert settings.ftp_poll_interval_seconds == 60
    assert isinstance(settings.local_raw_dir, Path)