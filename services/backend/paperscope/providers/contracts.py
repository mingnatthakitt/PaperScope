from __future__ import annotations

from pathlib import Path
from typing import Protocol

from paperscope.schemas import DeepPaperAnalysis, RAGAnswer, RAGModelInput


class ProviderConfigurationError(RuntimeError):
    pass


class RetryableProviderError(RuntimeError):
    pass


class ProviderAttemptsExhaustedError(RetryableProviderError):
    """A provider completed its configured internal retry/fallback sequence."""


class DocumentAnalysisProvider(Protocol):
    async def analyze_pdf(self, pdf_path: Path) -> DeepPaperAnalysis: ...


class EmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class MultimodalRAGProvider(Protocol):
    async def answer(self, request: RAGModelInput) -> RAGAnswer: ...
