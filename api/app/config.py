"""
config.py — Application settings, read exclusively from environment variables.

We use pydantic-settings so every value is type-checked at startup.
Missing required values surface as a loud ValidationError, not a silent None.

ANTHROPIC_API_KEY is Optional because the app deliberately keeps working
without it — the AI feature gracefully returns an error on that endpoint
while everything else continues normally.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database — empty string means SQLite (local dev, zero setup).
    # Set to postgresql://user:pass@host:5432/taskflow for deployment.
    database_url: str = ""

    # Anthropic key is optional. If absent, /ai/suggest-dependencies returns
    # a clear 503 and all other endpoints continue normally.
    anthropic_api_key: str = ""

    # Allowed frontend origin for CORS. In dev, localhost:3000 is always added.
    allowed_origin: str = "http://localhost:3000"

    @property
    def effective_database_url(self) -> str:
        """Return the SQLite URL when DATABASE_URL is not set."""
        if not self.database_url:
            return "sqlite:///./taskflow.db"
        return self.database_url

    @property
    def is_sqlite(self) -> bool:
        return self.effective_database_url.startswith("sqlite")

    @property
    def anthropic_available(self) -> bool:
        return bool(self.anthropic_api_key)


# Module-level singleton — imported everywhere.
settings = Settings()
