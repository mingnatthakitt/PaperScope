from __future__ import annotations

from functools import lru_cache

from paperscope.config import settings
from paperscope.providers.contracts import ProviderConfigurationError

QUERY_INSTRUCTION = (
    "Given a research question, retrieve passages, visual descriptions, or graph nodes "
    "from a scientific paper that directly answer the question."
)


@lru_cache(maxsize=4)
def _load_model(model_name: str, requested_device: str):
    """Load one model instance per model/device pair within a process."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ProviderConfigurationError(
            "Install the PaperScope conda environment before using local embeddings"
        ) from exc

    device = requested_device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return SentenceTransformer(model_name, device=device)


class QwenEmbeddingProvider:
    """Lazy, process-wide cache around the local Qwen embedding model."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.requested_device = device or settings.embedding_device

    @property
    def model(self):
        return _load_model(self.model_name, self.requested_device)

    def embed_query(self, text: str) -> list[float]:
        encoded = self.model.encode(
            [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vector = encoded[0].tolist()
        if len(vector) != settings.embedding_dimensions:
            raise ProviderConfigurationError(
                f"Embedding dimension {len(vector)} does not match {settings.embedding_dimensions}"
            )
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=False,
        )
        vectors = [row.tolist() for row in encoded]
        if any(len(vector) != settings.embedding_dimensions for vector in vectors):
            raise ProviderConfigurationError("Embedding provider returned an unexpected dimension")
        return vectors
