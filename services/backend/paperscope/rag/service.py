from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.orm import Session

from paperscope.config import settings
from paperscope.models.tables import PaperPage, PaperVisual
from paperscope.providers.contracts import ProviderConfigurationError, RetryableProviderError
from paperscope.providers.embeddings import QwenEmbeddingProvider
from paperscope.providers.nim import NimMuseProvider, NimNemotronProvider, citation_map
from paperscope.rag.context import build_multimodal_context
from paperscope.rag.retrieval import retrieve_all
from paperscope.schemas import RAGAnswer, RAGModelInput, RAGRequest, RetrievedContext


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
        primary_error: Exception | None = None
        provider_timeout = float(settings.rag_provider_timeout_seconds)
        # Keep a complete provider window available for Nemotron. Without this
        # guard, the configured Muse retry count can spend all 60 seconds before
        # the fallback is even constructed.
        fallback_reserve = provider_timeout + 1.0

        if await self.breaker.allow():
            try:
                provider = NimMuseProvider()
                for attempt in range(settings.rag_primary_max_retries + 1):
                    remaining = deadline - time.monotonic()
                    if remaining <= provider_timeout + fallback_reserve:
                        break
                    try:
                        result = await self._answer_with_deadline(provider, context, deadline)
                        await self.breaker.success()
                        return self._finalize(result.answer, result.model, False, records, started)
                    except RetryableProviderError as exc:
                        primary_error = exc
                        await self.breaker.failure()
                        if attempt < settings.rag_primary_max_retries:
                            await asyncio.sleep(0.6 * (2**attempt))
                        else:
                            break
            except ProviderConfigurationError as exc:
                primary_error = exc
        else:
            primary_error = RetryableProviderError("Muse circuit is open")

        fallback_error: RetryableProviderError | None = None
        try:
            fallback_provider = NimNemotronProvider()
            for fallback_attempt in range(settings.rag_fallback_max_retries + 1):
                try:
                    fallback = await self._answer_with_deadline(fallback_provider, context, deadline)
                    return self._finalize(fallback.answer, fallback.model, True, records, started)
                except RetryableProviderError as exc:
                    fallback_error = exc
                    remaining = deadline - time.monotonic()
                    if fallback_attempt < settings.rag_fallback_max_retries and remaining > provider_timeout + 1:
                        await asyncio.sleep(0.8)
                        continue
                    break
        except ProviderConfigurationError:
            raise primary_error or ProviderConfigurationError("No RAG provider configured")
        if fallback_error:
            raise fallback_error
        raise primary_error or RetryableProviderError("No RAG provider returned an answer")

    @staticmethod
    async def _answer_with_deadline(provider, context: RAGModelInput, deadline: float) -> RAGAnswer:
        """Run one provider call without starving the configured fallback."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RetryableProviderError(f"RAG time budget exhausted before calling {provider.model}")

        timeout = min(float(settings.rag_provider_timeout_seconds), remaining)
        if remaining < float(settings.rag_provider_timeout_seconds):
            raise RetryableProviderError(f"RAG time budget is too small to call {provider.model}")
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
