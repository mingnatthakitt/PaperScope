from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.orm import Session

from paperscope.config import GEMMA_MODEL, settings
from paperscope.models.tables import PaperPage, PaperVisual
from paperscope.providers.contracts import ProviderConfigurationError, RetryableProviderError
from paperscope.providers.embeddings import QwenEmbeddingProvider
from paperscope.providers.gemma import Gemma4RAGProvider
from paperscope.providers.nim import NimMuseProvider, NimNemotronProvider, citation_map
from paperscope.rag.context import build_multimodal_context
from paperscope.rag.retrieval import retrieve_all
from paperscope.schemas import RAGAnswer, RAGModelInput, RAGRequest, RetrievedContext

logger = logging.getLogger("paperscope.rag")
DEFAULT_PROVIDER_ORDER = ("muse", "nemotron", "gemma")


class CircuitBreaker:
    def __init__(self, failure_limit: int, window_seconds: int = 120, cooldown_seconds: int = 90) -> None:
        self.failure_limit = failure_limit
        self.window = timedelta(seconds=window_seconds)
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.failures: deque[datetime] = deque()
        self.opened_at: datetime | None = None
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            now = datetime.now(UTC)
            while self.failures and now - self.failures[0] > self.window:
                self.failures.popleft()
            if self.opened_at and now - self.opened_at < self.cooldown:
                return False
            if self.opened_at and now - self.opened_at >= self.cooldown:
                self.opened_at = None
            return True

    async def success(self) -> None:
        async with self._lock:
            self.failures.clear()
            self.opened_at = None

    async def failure(self) -> None:
        async with self._lock:
            now = datetime.now(UTC)
            self.failures.append(now)
            while self.failures and now - self.failures[0] > self.window:
                self.failures.popleft()
            if len(self.failures) >= self.failure_limit:
                self.opened_at = now


class RagService:
    def __init__(self) -> None:
        self.embeddings = QwenEmbeddingProvider()
        self.embedding_warmed = False
        self.breaker = CircuitBreaker(
            settings.circuit_breaker_failures,
            cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
        )
        self.breakers = {
            "muse": self.breaker,
            "nemotron": CircuitBreaker(
                settings.circuit_breaker_failures,
                cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
            ),
            "gemma": CircuitBreaker(
                settings.circuit_breaker_failures,
                cooldown_seconds=settings.circuit_breaker_cooldown_seconds,
            ),
        }

    def warm_embeddings(self) -> None:
        """Load Qwen in the API process before the first user query."""
        _ = self.embeddings.model
        self.embedding_warmed = True

    async def answer(self, session: Session, request: RAGRequest) -> RAGAnswer:
        started = time.perf_counter()
        # Leave a small amount of headroom for FastAPI to serialize the result
        # and return it before the endpoint's hard timeout fires.
        deadline = time.monotonic() + max(settings.rag_timeout_seconds - 1, 0.1)
        retrieval_query = request.question
        if request.history:
            recent_history = "\n".join(turn.content for turn in request.history[-4:])
            retrieval_query = f"{recent_history}\nCurrent question: {request.question}"[-8000:]
        query_vector = await asyncio.to_thread(self.embeddings.embed_query, retrieval_query)
        self.embedding_warmed = True
        # SQLAlchemy sessions are not thread-safe. Keep the database work on the
        # request thread and move only the CPU/model call off the event loop.
        evidence = retrieve_all(session, request.paper_ids, query_vector)
        evidence = self._with_citation_ids(evidence)
        context = build_multimodal_context(
            session,
            request.question,
            evidence,
            request.explanation_level,
            request.history,
        )
        records = self._citation_records(session, context)
        configured_order = [request.answer_model, *(key for key in DEFAULT_PROVIDER_ORDER if key != request.answer_model)]
        provider_order = [
            key
            for key in configured_order
            if (settings.nvidia_api_key and key in {"muse", "nemotron"})
            or (settings.google_ai_api_key and key == "gemma")
        ]
        if not provider_order:
            raise ProviderConfigurationError(
                "Configure NVIDIA_API_KEY or GOOGLE_AI_API_KEY to enable paper answers"
            )

        available_order = []
        for key in provider_order:
            if await self.breakers[key].allow():
                available_order.append(key)
            else:
                logger.warning("RAG provider %s circuit is open", key)
        if not available_order:
            raise RetryableProviderError("All configured RAG provider circuits are open")

        retry_limits = {
            "muse": settings.rag_primary_max_retries,
            "nemotron": settings.rag_fallback_max_retries,
            "gemma": settings.rag_gemma_max_retries,
        }
        last_error: Exception | None = None
        provider_timeout = float(settings.rag_provider_timeout_seconds)

        for index, provider_key in enumerate(available_order):
            reserve = provider_timeout * (len(available_order) - index - 1) + 1.0
            provider_deadline = deadline - reserve
            if provider_deadline - time.monotonic() <= 0.25:
                break
            try:
                provider = self._provider(provider_key, provider_timeout)
            except ProviderConfigurationError as exc:
                last_error = exc
                continue

            for attempt in range(retry_limits[provider_key] + 1):
                if provider_deadline - time.monotonic() <= 0.25:
                    break
                try:
                    result = await self._answer_with_deadline(provider, context, provider_deadline)
                    await self.breakers[provider_key].success()
                    return self._finalize(
                        result.answer,
                        result.model,
                        provider_key != request.answer_model,
                        records,
                        started,
                    )
                except ProviderConfigurationError as exc:
                    last_error = exc
                    logger.warning("RAG provider %s is not configured", provider_key)
                    break
                except RetryableProviderError as exc:
                    last_error = exc
                    await self.breakers[provider_key].failure()
                    logger.warning("RAG provider %s attempt %s failed", provider_key, attempt + 1)
                    if attempt < retry_limits[provider_key]:
                        backoff = min(0.6 * (2**attempt), max(provider_deadline - time.monotonic(), 0))
                        if backoff > 0:
                            await asyncio.sleep(backoff)
                except Exception as exc:
                    last_error = exc
                    await self.breakers[provider_key].failure()
                    logger.warning(
                        "RAG provider %s rejected the request: %s",
                        provider_key,
                        type(exc).__name__,
                    )
                    break

        if last_error:
            raise RetryableProviderError("All configured RAG providers failed after their retries") from last_error
        raise RetryableProviderError("The RAG request ran out of time before a provider could answer")

    @staticmethod
    def _provider(provider_key: str, timeout_seconds: float):
        if provider_key == "muse":
            return NimMuseProvider(timeout_seconds=timeout_seconds)
        if provider_key == "nemotron":
            return NimNemotronProvider(timeout_seconds=timeout_seconds)
        if provider_key == "gemma":
            return Gemma4RAGProvider(model=settings.rag_gemma_model or GEMMA_MODEL, timeout_seconds=timeout_seconds)
        raise ProviderConfigurationError("Unknown RAG provider selection")

    @staticmethod
    async def _answer_with_deadline(provider, context: RAGModelInput, deadline: float) -> RAGAnswer:
        """Run a provider call within its reserved slice of the request deadline."""
        timeout = min(float(settings.rag_provider_timeout_seconds), deadline - time.monotonic())
        if timeout <= 0.25:
            raise RetryableProviderError(f"RAG time budget exhausted before calling {provider.model}")
        if hasattr(provider, "timeout_seconds"):
            provider.timeout_seconds = timeout
        try:
            # The OpenAI-compatible client owns the socket timeout. Do not wrap
            # its thread-backed call in wait_for at the same deadline: cancelling
            # asyncio.to_thread leaves the underlying HTTP request alive and can
            # consume NIM worker capacity after the user-facing request ends.
            return await provider.answer(context)
        except TimeoutError as exc:
            raise RetryableProviderError(f"{provider.model} timed out after {timeout:.1f}s") from exc

    @staticmethod
    def _citation_records(session: Session, context: RAGModelInput) -> list[dict]:
        records: list[dict] = []
        for item in context.evidence:
            page = item.page_start
            asset_id = item.visual_id
            if item.source_type == "graph_node":
                for raw_evidence_id in item.metadata.get("evidenceIds", []):
                    try:
                        evidence_id = UUID(str(raw_evidence_id))
                    except (TypeError, ValueError):
                        continue
                    visual = session.get(PaperVisual, evidence_id)
                    if visual:
                        asset_id = visual.id
                        page = visual.page_number
                        break
                    paper_page = session.get(PaperPage, evidence_id)
                    if paper_page:
                        asset_id = paper_page.id
                        page = paper_page.page_number
                        break
            records.append(
                {
                    "id": item.id,
                    "paper_id": item.paper_id,
                    "source_type": item.source_type,
                    "page": page,
                    "section": item.section,
                    "label": item.metadata.get("label") or item.source_type,
                    "excerpt": item.content[:500],
                    "asset_id": asset_id,
                }
            )
        return records

    @staticmethod
    def _with_citation_ids(evidence: list[RetrievedContext]) -> list[RetrievedContext]:
        counters = {"E": 0, "V": 0, "G": 0}
        labeled: list[RetrievedContext] = []
        for item in evidence:
            prefix = "V" if item.source_type in {"visual", "table"} else "G" if item.source_type == "graph_node" else "E"
            counters[prefix] += 1
            labeled.append(item.model_copy(update={"id": f"{prefix}{counters[prefix]}"}))
        return labeled

    @staticmethod
    def _finalize(answer: str, model: str, fallback: bool, records: list[dict], started: float) -> RAGAnswer:
        citations = citation_map(answer, records)
        # The IDs in the prompt are the public citation contract. A response without
        # a recognized token is still returned, but the UI can warn that it is not grounded.
        return RAGAnswer(
            answer=answer,
            citations=citations,
            model=model,
            fallback_used=fallback,
            grounded=bool(citations),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
