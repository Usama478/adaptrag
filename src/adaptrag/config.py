from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ADAPTRAG_", extra="ignore")

    app_name: str = "AdaptRAG"
    environment: str = "dev"
    log_level: str = "INFO"
    max_upload_mb: int = 20
    data_dir: Path = Path("data")
    database_url: str = "postgresql+psycopg://adaptrag:change-me@localhost:5432/adaptrag"


@lru_cache
def get_settings() -> Settings:
    return Settings()
