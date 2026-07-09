"""
Centralized application configuration.

All environment-dependent values are loaded here via pydantic-settings so the
rest of the codebase never touches os.environ directly. This keeps
configuration testable, typed, and validated at startup (fail fast if a
required variable like GOOGLE_API_KEY is missing).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    google_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "models/text-embedding-004"

    # Storage
    database_url: str = "sqlite:///./data/app.db"
    chroma_persist_dir: str = "./data/chroma"

    # App behaviour
    log_level: str = "INFO"
    llm_max_retries: int = 3
    rag_top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton so .env is parsed only once."""
    return Settings()
