from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_parse_none_str: the value "None" in an env var or .env means Python None
    # (used by ADAPTRAG_LLM_TEMPERATURE=None; an empty value would not parse as a float).
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ADAPTRAG_", extra="ignore", env_parse_none_str="None"
    )

    app_name: str = "AdaptRAG"
    environment: str = "dev"
    log_level: str = "INFO"
    max_upload_mb: int = 20
    data_dir: Path = Path("data")
    database_url: str = "postgresql+psycopg://adaptrag:change-me@localhost:5432/adaptrag"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    # LLM answers (note 0015). SecretStr hides the key in repr() and logs; read it only
    # with get_secret_value() where the client is built. None or empty: answering is off.
    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    # None means "do not send temperature" (for models that do not accept it).
    llm_temperature: float | None = 0.0
    llm_timeout_seconds: float = 30.0
    llm_max_output_tokens: int = 500
    # Provisional: from 9 measured dev questions (note 0015); recalibrate in Stage 4.
    retrieval_min_score: float = 0.55


@lru_cache
def get_settings() -> Settings:
    return Settings()
