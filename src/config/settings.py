# src/config/settings.py
"""Centralized configuration using Pydantic Settings.

Supports environment-based configuration (dev / staging / production)
via the APP_ENV environment variable.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ─── General ──────────────────────────────────────────────────────────────
    app_env: str = Field(default="development", alias="APP_ENV")
    app_name: str = "SocrAItes"
    debug: bool = Field(default=True, alias="DEBUG")

    # ─── Server ───────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # ─── Database ─────────────────────────────────────────────────────────────
    # SQLite (legacy/dev)
    sqlite_db_path: Optional[str] = Field(default=None, alias="SOCRAITES_DB_PATH")
    # PostgreSQL (production)
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # ─── Elasticsearch ────────────────────────────────────────────────────────
    es_url: str = Field(default="http://localhost:9200", alias="ES_URL")

    # ─── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # ─── Auth / JWT ───────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(
        default="CHANGE-ME-IN-PRODUCTION-USE-RANDOM-SECRET",
        alias="JWT_SECRET_KEY",
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = Field(default=60, alias="JWT_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(default=7, alias="JWT_REFRESH_EXPIRE_DAYS")

    # ─── OAuth (Google) ───────────────────────────────────────────────────────
    google_client_id: Optional[str] = Field(default=None, alias="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[str] = Field(default=None, alias="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field(
        default="http://localhost:8000/auth/google/callback",
        alias="GOOGLE_REDIRECT_URI",
    )

    # ─── OAuth (Kakao) ────────────────────────────────────────────────────────
    kakao_client_id: Optional[str] = Field(default=None, alias="KAKAO_CLIENT_ID")
    kakao_client_secret: Optional[str] = Field(default=None, alias="KAKAO_CLIENT_SECRET")
    kakao_redirect_uri: str = Field(
        default="http://localhost:8000/auth/kakao/callback",
        alias="KAKAO_REDIRECT_URI",
    )

    # ─── LLM / OpenAI ────────────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # ─── Rate Limiting ────────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(default=30, alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_daily_tokens: int = Field(default=100000, alias="RATE_LIMIT_DAILY_TOKENS")

    # ─── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: List[str] = Field(
        default=["http://localhost:8000", "http://localhost:3000"],
        alias="CORS_ORIGINS",
    )

    # ─── File Upload ──────────────────────────────────────────────────────────
    max_upload_size_mb: int = Field(default=50, alias="MAX_UPLOAD_SIZE_MB")

    # ─── Monitoring ───────────────────────────────────────────────────────────
    sentry_dsn: Optional[str] = Field(default=None, alias="SENTRY_DSN")
    langsmith_api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def effective_cors_origins(self) -> List[str]:
        """In development, allow all origins. In production, use configured list."""
        if self.is_development:
            return ["*"]
        return self.cors_origins


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
