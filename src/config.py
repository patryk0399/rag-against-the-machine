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
    llm_backend: str = "llama_cpp"
    #NOTE: for llama.cpp specific (fir now)
    # llm_model_path: Path = Path("models") / "llamacpp" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    # llm_model_path: Path = Path("models") / "llamacpp" / "mistral-7b-instruct-v0.2.Q8_0.gguf"
    llm_model_path: Path = Path("models") / "llamacpp" / "Llama-3-SauerkrautLM-8b-Instruct-Q4_K_L.gguf"
    llm_context_window: int = 2048
    llm_n_gpu_layers: int = 10      #  0=CPU only, >0=some layers on GPU
    llm_n_threads: int = 4          #  threading hint
    max_messages: int = 20          #  max number of displayed messages (not used for context)
    
    llm_n_batch: int = 8
    llm_use_mmap: bool = False
    llm_use_mlock: bool = False
    
    #todo  prompt limit for context / "short-term-memory"


def load_config() -> AppConfig:
    """Load application configuration from environment and .env file.

    Load .env first, then read environment variables, and finally
    validate them into an AppConfig instance.
    """
    # Load .env from the current working directory if present.
    load_dotenv(override=True)
    defaults = AppConfig()

    raw_config = {
        "env": os.getenv("ENV", defaults.env),
        "data_dir": os.getenv("DATA_DIR", str(defaults.data_dir)),
        "log_level": os.getenv("LOG_LEVEL", defaults.log_level),
        "embedding_model_name": os.getenv(
            "EMBEDDING_MODEL_NAME",
            defaults.embedding_model_name,
        ),
        "llm_backend": os.getenv("LLM_BACKEND", defaults.llm_backend),
        "llm_model_path": os.getenv("LLM_MODEL_PATH", str(defaults.llm_model_path)),
        "llm_context_window": int(
            os.getenv("LLM_CONTEXT_WINDOW", str(defaults.llm_context_window))
        ),
        "llm_n_gpu_layers": int(
            os.getenv("LLM_N_GPU_LAYERS", str(defaults.llm_n_gpu_layers))
        ),
        "llm_n_threads": int(os.getenv("LLM_N_THREADS", str(defaults.llm_n_threads))),
        "llm_n_batch": int(os.getenv("LLM_N_BATCH", str(defaults.llm_n_batch))),
        "llm_use_mmap": os.getenv("LLM_USE_MMAP", str(defaults.llm_use_mmap)).lower()
        in {"1", "true", "yes", "on"},
        "llm_use_mlock": os.getenv("LLM_USE_MLOCK", str(defaults.llm_use_mlock)).lower()
        in {"1", "true", "yes", "on"},
        "max_messages": int(os.getenv("MAX_MESSAGES", str(defaults.max_messages))),
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