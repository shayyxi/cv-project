from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    database_url: str = Field(validation_alias="DATABASE_URL")

    storage_provider: str = Field(default="local")

    camera_ids: list[str] = Field(default_factory=list)

    local_raw_dir: Path = Field(default=BASE_DIR / "data" / "raw")
    local_processed_dir: Path = Field(default=BASE_DIR / "data" / "processed")
    local_failed_dir: Path = Field(default=BASE_DIR / "data" / "failed")

    ftp_poll_interval_seconds: int = Field(default=60)
    processor_sleep_seconds: int = Field(default=5)

    wordpress_webhook_url: str = Field(validation_alias="WORDPRESS_WEBHOOK_URL")
    wordpress_api_key: str = Field(validation_alias="WORDPRESS_API_KEY")

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()   # type: ignore[call-arg]