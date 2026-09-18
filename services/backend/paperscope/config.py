from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_DOTENV_PATH = find_dotenv(usecwd=True)
load_dotenv(_DOTENV_PATH)
_CONFIG_ROOT = Path(_DOTENV_PATH).resolve().parent if _DOTENV_PATH else Path.cwd()

DEFAULT_ANALYSIS_MODEL = "gemini-3.8-flash"
ANALYSIS_MODEL_FALLBACKS: dict[str, tuple[str, ...]] = {
    "gemini-3.8-flash": ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"),
    "gemini-3.7-flash": ("gemini-3.7-flash", "gemini-3.6-flash"),
    "gemini-3.6-flash": ("gemini-3.6-flash",),
}


def analysis_model_chain(preferred: str | None = None) -> tuple[str, ...]:
    """Return the supported model order for a requested indexing model."""
    return ANALYSIS_MODEL_FALLBACKS.get(preferred or DEFAULT_ANALYSIS_MODEL, ANALYSIS_MODEL_FALLBACKS[DEFAULT_ANALYSIS_MODEL])


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://paperscope:paperscope@127.0.0.1:5432/paperscope",
    )
    papers_root: Path = Path(os.getenv("PAPERS_ROOT", "./data/papers"))
    google_ai_api_key: str | None = os.getenv("GOOGLE_AI_API_KEY")
    paper_analysis_model: str = os.getenv("PAPER_ANALYSIS_MODEL", DEFAULT_ANALYSIS_MODEL)
    nim_base_url: str = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nvidia_api_key: str | None = os.getenv("NVIDIA_API_KEY")
    rag_primary_model: str = os.getenv("RAG_PRIMARY_MODEL", "meta/muse-glimmer-30b")
    rag_fallback_model: str = os.getenv(
        "RAG_FALLBACK_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    )
    rag_primary_max_retries: int = _int("RAG_PRIMARY_MAX_RETRIES", 2)
    rag_fallback_max_retries: int = _int("RAG_FALLBACK_MAX_RETRIES", 2)
    rag_timeout_seconds: int = _int("RAG_TIMEOUT_SECONDS", 60)
    # Keep each upstream call bounded so a slow Muse request cannot consume the
    # entire request budget and prevent the Nemotron fallback from running.
    rag_provider_timeout_seconds: int = _int("RAG_PROVIDER_TIMEOUT_SECONDS", 18)
    circuit_breaker_failures: int = _int("RAG_CIRCUIT_BREAKER_FAILURES", 3)
    circuit_breaker_cooldown_seconds: int = _int("RAG_CIRCUIT_BREAKER_COOLDOWN_SECONDS", 90)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    embedding_dimensions: int = _int("EMBEDDING_DIMENSIONS", 1024)
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "auto")
    semantic_scholar_api_key: str | None = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    discovery_timeout_seconds: int = _int("DISCOVERY_TIMEOUT_SECONDS", 20)
    discovery_provider_timeout_seconds: int = _int("DISCOVERY_PROVIDER_TIMEOUT_SECONDS", 8)
    discovery_http_timeout_seconds: int = _int("DISCOVERY_HTTP_TIMEOUT_SECONDS", 6)
    max_pdf_bytes: int = _int("MAX_PDF_BYTES", 100 * 1024 * 1024)
    max_pdf_pages: int = _int("MAX_PDF_PAGES", 200)
    render_dpi: int = _int("PDF_RENDER_DPI", 200)
    analysis_prompt_version: str = os.getenv("ANALYSIS_PROMPT_VERSION", "analysis-v1")
    gemini_model_max_retries: int = _int("GEMINI_MODEL_MAX_RETRIES", 2)
    rag_prompt_version: str = os.getenv("RAG_PROMPT_VERSION", "rag-v1")
    rag_rate_limit_per_minute: int = _int("RAG_RATE_LIMIT_PER_MINUTE", 30)
    ingestion_max_attempts: int = _int("INGESTION_MAX_ATTEMPTS", 3)
    ingestion_lease_seconds: int = _int("INGESTION_LEASE_SECONDS", 600)
    ingestion_retry_backoff_seconds: int = _int("INGESTION_RETRY_BACKOFF_SECONDS", 5)

    def __post_init__(self) -> None:
        if not self.papers_root.is_absolute():
            object.__setattr__(self, "papers_root", (_CONFIG_ROOT / self.papers_root).resolve())


settings = Settings()
