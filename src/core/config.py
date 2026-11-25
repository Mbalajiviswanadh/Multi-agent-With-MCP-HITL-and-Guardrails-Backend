"""
src/core/config.py
------------------
Pydantic BaseSettings — validates all required environment variables at
application startup.  Any missing or malformed key raises a clear error
before the server binds to a port.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for all configuration knobs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Providers (Groq only — both primary and fallback use the same key) ---
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"        # Primary: 120B high quality
    groq_fallback_model: str = "openai/gpt-oss-20b"  # Fallback: 20B higher TPM quota
    groq_request_timeout: float = 15.0

    # --- Database ---
    database_url: str
    db_pool_min: int = 2
    db_pool_max: int = 10

    # --- Tool API Keys ---
    tavily_api_key: str = ""
    aviation_stack_api_key: str = ""
    openweather_api_key: str = ""
    huggingface_api_key: str = ""

    # --- Qdrant Cloud (Mem0 Vector Store) ---
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # --- Security & JWT Authentication ---
    jwt_secret_key: str = "tripmate_production_jwt_secret_key_change_in_env_super_secure_99"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60 * 24 * 7  # 7 days expiration

    # --- CORS ---
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    # --- LangSmith Observability ---
    langsmith_tracing: str = "false"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "TripMate-AI"

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str = "agent_runs.log"

    @property
    def db_url_with_ssl(self) -> str:
        url = self.database_url
        if "sslmode=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}sslmode=require"
        return url

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


# Module-level singleton — import this everywhere
settings = Settings()
