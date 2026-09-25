from __future__ import annotations

import asyncio
import base64
import time

from paperscope.config import settings
from paperscope.providers.contracts import ProviderConfigurationError, RetryableProviderError
from paperscope.schemas import EvidenceCitation, RAGAnswer, RAGModelInput

RAG_SYSTEM_PROMPT = """
You answer questions about research papers using only the supplied evidence. Cite claims
with the exact evidence tokens included in the context, such as [E1], [V1], or [G1].
If the evidence is insufficient, say that clearly. Never invent a result, figure detail,
or comparison. For visual questions, inspect the attached page or figure image and use
it together with the text. Prior conversation is context for a follow-up, not evidence;
re-ground every claim in the supplied evidence. Keep the answer concise but explain the reasoning.
""".strip()

NIM_CONCISE_RESPONSE_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
MAX_EVIDENCE_ITEMS = 8
MAX_EVIDENCE_CHARS = 2800
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 1200


def _status_code(error: Exception) -> str:
    return str(getattr(error, "status_code", getattr(error, "response", "")))


def _retryable_nim_error(error: Exception) -> bool:
    status = _status_code(error)
    message = str(error).lower()
    return status in {"408", "429", "500", "502", "503", "504"} or any(
        marker in message
        for marker in (
            "timeout",
            "timed out",
            "readtimeout",
            "connection reset",
            "server disconnected",
            "temporarily unavailable",
            "resourceexhausted",
        )
    )


class NimProvider:
    def __init__(self, model: str, timeout_seconds: float | None = None) -> None:
        if not settings.nvidia_api_key:
            raise ProviderConfigurationError("NVIDIA_API_KEY is not configured")
        self.model = model
        self.timeout_seconds = (
            settings.rag_provider_timeout_seconds if timeout_seconds is None else timeout_seconds
        )

    @property
    def client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderConfigurationError("Install openai in the PaperScope environment") from exc
        return OpenAI(
            base_url=settings.nim_base_url,
            api_key=settings.nvidia_api_key,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    def _content_parts(self, request: RAGModelInput) -> list[dict[str, object]]:
        evidence_text = "\n\n".join(
            f"[{item.id}] ({item.source_type}, pages {item.page_start or '?'}-{item.page_end or item.page_start or '?'})\n{item.content[:MAX_EVIDENCE_CHARS]}"
            for item in request.evidence[:MAX_EVIDENCE_ITEMS]
        )
        history_text = "\n".join(
            f"{turn.role.title()}: {turn.content[:MAX_HISTORY_CHARS]}"
            for turn in request.history[-MAX_HISTORY_TURNS:]
        )
        conversation_context = f"\nConversation so far:\n{history_text}\n" if history_text else ""
        parts: list[dict[str, object]] = [
            {
                "type": "text",
                "text": f"Answer level: {request.explanation_level}{conversation_context}\nCurrent question: {request.question}\n\nEvidence:\n{evidence_text}",
            }
        ]
        for image in request.images:
            url = image.url
            if not url and image.base64:
                url = image.base64 if image.base64.startswith("data:") else f"data:image/webp;base64,{image.base64}"
            if url:
                parts.append({"type": "text", "text": f"Image {image.id}, page {image.page_number}, {image.label or image.type}:"})
                parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts

    def _answer_sync(self, request: RAGModelInput) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": self._content_parts(request)},
                ],
                temperature=0.2,
                max_tokens=1200,
                extra_body=NIM_CONCISE_RESPONSE_BODY,
            )
        except Exception as exc:
            if _retryable_nim_error(exc):
                raise RetryableProviderError(str(exc)) from exc
            raise
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise RetryableProviderError("NIM returned an empty response")
        return str(content)

    async def answer_text(self, request: RAGModelInput) -> str:
        return await asyncio.to_thread(self._answer_sync, request)

    def _complete_sync(self, prompt: str, system: str | None = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system or "Return only the requested JSON or text."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2200,
                extra_body=NIM_CONCISE_RESPONSE_BODY,
            )
        except Exception as exc:
            if _retryable_nim_error(exc):
                raise RetryableProviderError(str(exc)) from exc
            raise
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise RetryableProviderError("NIM returned an empty response")
        return str(content)

    async def complete(self, prompt: str, system: str | None = None) -> str:
        return await asyncio.to_thread(self._complete_sync, prompt, system)


class NimMuseProvider(NimProvider):
    def __init__(self, timeout_seconds: float | None = None) -> None:
        super().__init__(settings.rag_primary_model, timeout_seconds)

    async def answer(self, request: RAGModelInput) -> RAGAnswer:
        started = time.perf_counter()
        content = await self.answer_text(request)
        return RAGAnswer(
            answer=content,
            citations=[],
            model=self.model,
            fallback_used=False,
            grounded=False,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )


class NimNemotronProvider(NimProvider):
    def __init__(self, timeout_seconds: float | None = None) -> None:
        super().__init__(settings.rag_fallback_model, timeout_seconds)

    async def answer(self, request: RAGModelInput) -> RAGAnswer:
        started = time.perf_counter()
        content = await self.answer_text(request)
        return RAGAnswer(
            answer=content,
            citations=[],
            model=self.model,
            fallback_used=True,
            grounded=False,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )


def image_data_url(path_bytes: bytes, mime_type: str = "image/webp") -> str:
    return f"data:{mime_type};base64,{base64.b64encode(path_bytes).decode('ascii')}"


def citation_map(answer: str, evidence: list[dict[str, object]]) -> list[EvidenceCitation]:
    """Resolve citation tokens without allowing the model to invent source records."""
    tokens = set()
    for token in answer.replace("[", " [").split():
        clean = token.strip(".,:;()")
        if clean.startswith("[") and clean.endswith("]"):
            tokens.add(clean[1:-1])
    citations: list[EvidenceCitation] = []
    for item in evidence:
        if item["id"] in tokens:
            citations.append(EvidenceCitation.model_validate(item))
    return citations
