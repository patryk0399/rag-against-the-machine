"""Application configuration.

Maps environment variables and .env values into a typed AppConfig object.
This module is used by all layers (data, RAG, agents, UI) as a single
source of truth for runtime configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import os

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


class AppConfig(BaseModel):
    """Typed application configuration.
        Define existing fields + defaults/fallbacks.
        Will be used by all modules to handle paths, variables, etc.
    """

    env: Literal["dev", "prod"] = "dev"
    data_dir: Path = Path("data")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    llm_backend: str = "dummy"


def load_config() -> AppConfig:
    """Load application configuration from environment and .env file.

    Load .env first, then read environment variables, and finally
    validate them into an AppConfig instance.
    """
    # Load .env from the current working directory if present.
    load_dotenv()

    # Collect raw values from environment with explicit defaults.
    raw_config = {
    "env": os.getenv("ENV", "dev"),
    "data_dir": os.getenv("DATA_DIR", "data"),
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "llm_backend": os.getenv("LLM_BACKEND", "dummy"),
    "embedding_model_name": os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    ),
}

    try:
        config = AppConfig(**raw_config)
    except ValidationError as exc:
        # Fail fast and show what went wrong with configuration.
        print("[config] Failed to validate AppConfig:")
        print(exc)
        raise

    print("[config] Loaded AppConfig:", config)
    return config


__all__ = ["AppConfig", "load_config"]