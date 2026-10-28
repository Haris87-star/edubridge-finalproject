"""Environment-driven configuration for the API service.

All settings can be overridden with ``SPAM_*`` environment variables, which is
what the Docker image and docker-compose file do.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.config import MODEL_FILE, PROJECT_ROOT


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "SMS Spam Detector API"
    version: str = "1.0.0"
    model_path: Path = MODEL_FILE
    threshold: float = 0.5
    max_batch_size: int = 500
    max_text_length: int = 5000
    admin_token: str | None = None
    cors_origins: tuple[str, ...] = ("*",)
    log_level: str = "INFO"
    feedback_path: Path = PROJECT_ROOT / "feedback" / "feedback.jsonl"
    autotrain: bool = False
    explain_top_k: int = 8


def load_settings() -> Settings:
    """Read settings from the environment (with safe defaults)."""
    origins = os.getenv("SPAM_CORS_ORIGINS", "*")
    return Settings(
        model_path=Path(os.getenv("SPAM_MODEL_PATH", str(MODEL_FILE))),
        threshold=min(max(_env_float("SPAM_THRESHOLD", 0.5), 0.01), 0.99),
        max_batch_size=max(_env_int("SPAM_MAX_BATCH_SIZE", 500), 1),
        max_text_length=max(_env_int("SPAM_MAX_TEXT_LENGTH", 5000), 1),
        admin_token=os.getenv("SPAM_ADMIN_TOKEN") or None,
        cors_origins=tuple(
            origin.strip() for origin in origins.split(",") if origin.strip()
        )
        or ("*",),
        log_level=os.getenv("SPAM_LOG_LEVEL", "INFO").upper(),
        feedback_path=Path(
            os.getenv("SPAM_FEEDBACK_PATH", str(PROJECT_ROOT / "feedback" / "feedback.jsonl"))
        ),
        autotrain=_env_bool("SPAM_AUTOTRAIN", False),
        explain_top_k=max(_env_int("SPAM_EXPLAIN_TOP_K", 8), 1),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
