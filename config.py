"""
Central configuration for the TravelSATHI recommendation service.

Everything is read from environment variables so the service is configurable
without code changes. Sensible hackathon-safe defaults are provided so the
service runs in "demo mode" (mock Google Places + rule-based intent
extraction) even with zero API keys configured. This lets your teammates
run the service immediately, and lets you flip in real keys later without
touching a single line of code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- External API keys (optional — service falls back to mock/rule-based
    #     behaviour when absent, so it is always demoable offline) ---
    GEMINI_API_KEY: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    GOOGLE_PLACES_API_KEY: str | None = field(default_factory=lambda: os.getenv("GOOGLE_PLACES_API_KEY"))

    # --- Mumbai dataset source ---
    # "json"  -> read data/mumbai_places.json (default, zero setup)
    # "mysql" -> read from the shared MySQL database via retrieval/mysql_retriever.py
    MUMBAI_DATA_SOURCE: str = field(default_factory=lambda: os.getenv("MUMBAI_DATA_SOURCE", "json"))
    MUMBAI_DATA_PATH: str = field(
        default_factory=lambda: os.getenv(
            "MUMBAI_DATA_PATH",
            os.path.join(os.path.dirname(__file__), "data", "mumbai_places.json"),
        )
    )

    # --- MySQL (only used when MUMBAI_DATA_SOURCE=mysql or for feedback storage) ---
    MYSQL_HOST: str = field(default_factory=lambda: os.getenv("MYSQL_HOST", "localhost"))
    MYSQL_PORT: int = field(default_factory=lambda: _int("MYSQL_PORT", 3306))
    MYSQL_USER: str = field(default_factory=lambda: os.getenv("MYSQL_USER", "travelsathi"))
    MYSQL_PASSWORD: str = field(default_factory=lambda: os.getenv("MYSQL_PASSWORD", ""))
    MYSQL_DATABASE: str = field(default_factory=lambda: os.getenv("MYSQL_DATABASE", "travelsathi"))

    # --- Feedback storage ---
    # "json"  -> append-only local file (default, zero setup, fine for a hackathon demo)
    # "mysql" -> writes into the `recommendation_feedback` table (see sql/schema.sql)
    FEEDBACK_STORE: str = field(default_factory=lambda: os.getenv("FEEDBACK_STORE", "json"))
    FEEDBACK_LOG_PATH: str = field(
        default_factory=lambda: os.getenv(
            "FEEDBACK_LOG_PATH",
            os.path.join(os.path.dirname(__file__), "data", "feedback_log.jsonl"),
        )
    )

    # --- Embeddings ---
    # "auto"      -> try sentence-transformers, fall back to TF-IDF if unavailable/offline
    # "st"        -> force sentence-transformers (raises if not installed)
    # "hashing"   -> force the offline TF-IDF/SVD fallback (no downloads, deterministic)
    EMBEDDING_BACKEND: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "auto"))
    EMBEDDING_MODEL_NAME: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    )
    EMBEDDING_CACHE_PATH: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_CACHE_PATH",
            os.path.join(os.path.dirname(__file__), "data", "embedding_cache.json"),
        )
    )

    # --- Google Places retrieval ---
    GOOGLE_PLACES_CANDIDATE_LIMIT: int = field(default_factory=lambda: _int("GOOGLE_PLACES_CANDIDATE_LIMIT", 60))
    GOOGLE_PLACES_DEFAULT_RADIUS_M: int = field(default_factory=lambda: _int("GOOGLE_PLACES_DEFAULT_RADIUS_M", 3000))
    GOOGLE_PLACES_RESULT_CACHE_TTL_SECONDS: int = field(
        default_factory=lambda: _int("GOOGLE_PLACES_RESULT_CACHE_TTL_SECONDS", 3600)  # 1 hour, in-memory only
    )

    # --- Ranking weights (Section 15 of the spec — starting weights, see DESIGN.md) ---
    WEIGHT_SEMANTIC: float = field(default_factory=lambda: _float("WEIGHT_SEMANTIC", 0.40))
    WEIGHT_DISTANCE: float = field(default_factory=lambda: _float("WEIGHT_DISTANCE", 0.20))
    WEIGHT_BUDGET: float = field(default_factory=lambda: _float("WEIGHT_BUDGET", 0.15))
    WEIGHT_TIME: float = field(default_factory=lambda: _float("WEIGHT_TIME", 0.10))
    WEIGHT_GROUP: float = field(default_factory=lambda: _float("WEIGHT_GROUP", 0.05))
    WEIGHT_QUALITY: float = field(default_factory=lambda: _float("WEIGHT_QUALITY", 0.05))
    WEIGHT_LOCAL: float = field(default_factory=lambda: _float("WEIGHT_LOCAL", 0.05))

    DEFAULT_TOP_K: int = field(default_factory=lambda: _int("DEFAULT_TOP_K", 10))
    MAX_TOP_K: int = field(default_factory=lambda: _int("MAX_TOP_K", 50))

    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()
