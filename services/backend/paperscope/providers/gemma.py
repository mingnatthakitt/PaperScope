from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from PIL import Image
from pydantic import Field

from paperscope.config import GEMMA_MODEL, settings
from paperscope.providers.contracts import (
    ProviderAttemptsExhaustedError,
    ProviderConfigurationError,
    RetryableProviderError,
)
from paperscope.providers.nim import (
    MAX_EVIDENCE_CHARS,
    MAX_EVIDENCE_ITEMS,
    MAX_HISTORY_CHARS,
    MAX_HISTORY_TURNS,
    RAG_SYSTEM_PROMPT,
)
from paperscope.schemas import (
    CamelModel,
    DeepPaperAnalysis,
    ImportantVisual,
    RAGAnswer,
    RAGModelInput,
)

logger = logging.getLogger("paperscope.gemma")

MAX_BATCH_PAGES = 10
MAX_BATCH_TEXT_CHARS = 50_000
MAX_PAGE_TEXT_CHARS = 20_000
MAX_SYNTHESIS_NOTE_CHARS = 500
MAX_SYNTHESIS_VISUAL_TEXT_CHARS = 400
MAX_IMAGE_EDGE = 1280
IMAGE_JPEG_QUALITY = 78


class PageLike(Protocol):
    page_number: int
    text: str


class GemmaPageNote(CamelModel):
    page_number: int
    notes: str = Field(max_length=800)


class GemmaPageBatch(CamelModel):
    page_notes: list[GemmaPageNote]
    important_visuals: list[ImportantVisual] = Field(default_factory=list, max_length=3)


def _status_code(error: Exception) -> str:
    for attribute in ("code", "status_code"):
        value = getattr(error, attribute, None)
        if value is not None:
            return str(value)
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return str(value) if value is not None else ""


def _retryable_error(error: Exception) -> RetryableProviderError | None:
    message = str(error).lower()
    if _status_code(error) in {"408", "429", "500", "502", "503", "504"} or any(
        marker in message
        for marker in (
            "timeout",
            "timed out",
            "server disconnected",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "resourceexhausted",
        )
    ):
        return RetryableProviderError(str(error))
    return None


def _parse_json(text: str):
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else value
        value = value.rsplit("```", 1)[0].strip()
    return json.loads(value)


def _client(timeout_seconds: float | None = None):
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ProviderConfigurationError("Install google-genai in the PaperScope environment") from exc
    if not settings.google_ai_api_key:
        raise ProviderConfigurationError("GOOGLE_AI_API_KEY is not configured")
    options = None
    if timeout_seconds is not None:
        options = types.HttpOptions(timeout=max(1, round(timeout_seconds * 1000)))
    client = genai.Client(api_key=settings.google_ai_api_key, http_options=options) if options else genai.Client(api_key=settings.google_ai_api_key)
    return client, types


def _page_batches(pages: list[PageLike]) -> list[list[PageLike]]:
    batches: list[list[PageLike]] = []
    current: list[PageLike] = []
    current_chars = 0
    for page in pages:
        text = GemmaDocumentAnalysisProvider._page_text(page)
        page_chars = len(text)
        if current and (
            len(current) >= MAX_BATCH_PAGES
            or current_chars + page_chars > MAX_BATCH_TEXT_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(page)
        current_chars += page_chars
    if current:
        batches.append(current)
    return batches


def _page_image_part(types, path: Path):
    try:
        from PIL import Image

        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not prepare a rendered page image: {path.name}") from exc
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


class GemmaDocumentAnalysisProvider:
    """Page-batched Gemma analysis used after the Gemini Flash models fail."""

    def __init__(self, model: str = GEMMA_MODEL) -> None:
        self.model = model

    @staticmethod
    def _page_text(page: PageLike) -> str:
        text = str(getattr(page, "text", "") or "")
        if len(text) > MAX_PAGE_TEXT_CHARS:
            return f"{text[:MAX_PAGE_TEXT_CHARS]}\n[Page text truncated; inspect the rendered image for remaining content.]"
        return text

    @staticmethod
    def _batch_prompt(pages: list[PageLike]) -> str:
        schema = GemmaPageBatch.model_json_schema(by_alias=True)
        numbers = [int(page.page_number) for page in pages]
        return (
            "Analyze the provided research-paper pages. Each page is represented by its extracted text "
            "and its rendered page image. Use both; scanned or image-heavy pages may have little extracted text. "
            "Return one concise note for every page number listed, including blank notes only when the page "
            "contains no useful information. Identify figures, plots, diagrams, equations, and tables that "
            "matter to the paper. For each visual, use its 1-based page number, an accurate caption and "
            "description, and a normalized [x, y, width, height] bounding box only when the location is clear. "
            "Do not invent details. Return JSON only, with this schema:\n"
            f"{json.dumps(schema, separators=(',', ':'))}\n"
            f"Pages to cover exactly: {numbers}. Keep each page note under 500 characters; return at most three "
            "high-signal visual records for the batch, with concise visual text under 400 characters per field. "
            "Visual keys should be stable within the page batch."
        )

    def _analyze_batch_sync(
        self,
        client,
        types,
        pages: list[PageLike],
        page_images: dict[int, Path],
    ) -> GemmaPageBatch:
        parts = [types.Part.from_text(text=self._batch_prompt(pages))]
        for page in pages:
            page_number = int(page.page_number)
            page_text = self._page_text(page)
            parts.append(types.Part.from_text(text=f"\n--- PAGE {page_number} EXTRACTED TEXT ---\n{page_text or '[No extractable text on this page.]'}"))
            image_path = page_images.get(page_number)
            if image_path is None or not image_path.is_file():
                raise ValueError(f"Rendered image is missing for page {page_number}")
            parts.append(types.Part.from_text(text=f"Rendered image for page {page_number}:"))
            parts.append(_page_image_part(types, image_path))
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=parts,
                config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=6000),
            )
        except Exception as exc:
            if retryable := _retryable_error(exc):
                raise retryable from exc
            raise
        text = getattr(response, "text", None)
        if not text:
            raise RetryableProviderError("Gemma returned an empty page-analysis response")
        try:
            result = GemmaPageBatch.model_validate(_parse_json(text))
        except (json.JSONDecodeError, ValueError) as exc:
            raise RetryableProviderError("Gemma returned invalid page-analysis JSON") from exc
        self._validate_batch_result(result, pages)
        return result

    @staticmethod
    def _validate_batch_result(result: GemmaPageBatch, pages: list[PageLike]) -> None:
        expected_pages = {int(page.page_number) for page in pages}
        returned_pages = [note.page_number for note in result.page_notes]
        if set(returned_pages) != expected_pages or len(returned_pages) != len(expected_pages):
            raise RetryableProviderError("Gemma did not return exactly one note for every requested page")
        if any(visual.page not in expected_pages for visual in result.important_visuals):
            raise RetryableProviderError("Gemma returned a visual for a page outside the current batch")
        visual_keys = [visual.key.strip() for visual in result.important_visuals]
        if any(not key for key in visual_keys) or len(visual_keys) != len(set(visual_keys)):
            raise RetryableProviderError("Gemma returned an empty or duplicate visual key")

    def _final_prompt(self, page_notes: list[GemmaPageNote], visuals: list[ImportantVisual]) -> str:
        schema = DeepPaperAnalysis.model_json_schema(by_alias=True)
        bounded_page_notes = [
            {"pageNumber": note.page_number, "notes": note.notes[:MAX_SYNTHESIS_NOTE_CHARS]}
            for note in page_notes
        ]
        bounded_visuals = []
        for visual in visuals:
            payload = visual.model_dump(mode="json", by_alias=True)
            for field in ("label", "caption", "reason", "visualDescription", "visualInterpretation"):
                value = payload.get(field)
                if isinstance(value, str):
                    payload[field] = value[:MAX_SYNTHESIS_VISUAL_TEXT_CHARS]
            bounded_visuals.append(payload)
        observations = {"pageNotes": bounded_page_notes, "importantVisuals": bounded_visuals}
        return (
            "You are PaperScope's research-paper analyst. Synthesize these observations from every page of "
            "one paper into the requested analysis. Return JSON only matching the supplied schema exactly. "
            "Produce simple, student, and researcher explanations; identify the central problem, method, "
            "contributions, results, limitations, prerequisites, and glossary. Create 5-9 high-signal graph "
            "nodes and valid edges. Graph evidence_keys may refer only to page-N or one of the supplied visual "
            "keys. Preserve visual page numbers and stable visual keys. The batch visual records are preserved separately, "
            "so return an empty important_visuals array instead of regenerating them. Do not claim evidence that is absent.\n"
            f"Schema:\n{json.dumps(schema, separators=(',', ':'))}\n"
            f"Page observations and visuals:\n{json.dumps(observations, ensure_ascii=False, separators=(',', ':'))}"
        )

    def _synthesize_sync(self, client, types, page_notes: list[GemmaPageNote], visuals: list[ImportantVisual]) -> DeepPaperAnalysis:
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=self._final_prompt(page_notes, visuals),
                config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=12000),
            )
        except Exception as exc:
            if retryable := _retryable_error(exc):
                raise retryable from exc
            raise
        text = getattr(response, "text", None)
        if not text:
            raise RetryableProviderError("Gemma returned an empty final analysis")
        try:
            analysis = DeepPaperAnalysis.model_validate(_parse_json(text))
        except (json.JSONDecodeError, ValueError) as exc:
            raise RetryableProviderError("Gemma returned invalid final-analysis JSON") from exc
        return analysis

    @staticmethod
    def _cache_path(cache_dir: Path, batch_index: int, pages: list[PageLike]) -> Path:
        first = int(pages[0].page_number)
        last = int(pages[-1].page_number)
        return cache_dir / f"batch-{batch_index:03d}-{first:03d}-{last:03d}.json"

    async def analyze_pages(
        self,
        pages: list[PageLike],
        page_images: dict[int, Path],
        cache_dir: Path,
        max_retries: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DeepPaperAnalysis:
        if not pages:
            raise ValueError("Gemma requires page text and rendered images")
        batches = _page_batches(pages)
        cache_dir.mkdir(parents=True, exist_ok=True)
        client, types = _client()
        all_notes: list[GemmaPageNote] = []
        all_visuals: list[ImportantVisual] = []
        last_error: Exception | None = None

        try:
            for batch_index, batch in enumerate(batches, start=1):
                cache_path = self._cache_path(cache_dir, batch_index, batch)
                result: GemmaPageBatch | None = None
                if cache_path.is_file():
                    try:
                        result = GemmaPageBatch.model_validate_json(cache_path.read_text(encoding="utf-8"))
                        self._validate_batch_result(result, batch)
                    except (OSError, ValueError, RetryableProviderError):
                        result = None
                        logger.warning("Ignoring invalid cached Gemma analysis batch %s", batch_index)
                if result is None:
                    for attempt in range(max_retries + 1):
                        try:
                            result = await asyncio.to_thread(
                                self._analyze_batch_sync, client, types, batch, page_images
                            )
                            break
                        except RetryableProviderError as exc:
                            last_error = exc
                            if attempt + 1 < max_retries + 1:
                                logger.warning(
                                    "Gemma indexing batch %s/%s attempt %s/%s failed",
                                    batch_index,
                                    len(batches),
                                    attempt + 1,
                                    max_retries + 1,
                                )
                            else:
                                raise ProviderAttemptsExhaustedError(
                                    f"Gemma page batch {batch_index}/{len(batches)} failed after retries"
                                ) from exc
                    if result is None:
                        raise ProviderAttemptsExhaustedError("Gemma did not complete a page-analysis batch") from last_error
                    temporary = cache_path.with_suffix(".json.tmp")
                    temporary.write_text(result.model_dump_json(by_alias=True), encoding="utf-8")
                    temporary.replace(cache_path)

                all_notes.extend(result.page_notes)
                all_visuals.extend(
                    visual.model_copy(update={"key": f"page-{visual.page}-{visual.key}"})
                    for visual in result.important_visuals
                )
                if progress_callback:
                    progress_callback(batch_index, len(batches))

            for attempt in range(max_retries + 1):
                try:
                    analysis = await asyncio.to_thread(
                        self._synthesize_sync, client, types, all_notes, all_visuals
                    )
                    return analysis.model_copy(update={"important_visuals": all_visuals})
                except RetryableProviderError as exc:
                    last_error = exc
                    if attempt + 1 >= max_retries + 1:
                        break
            raise ProviderAttemptsExhaustedError("Gemma could not produce a valid final analysis") from last_error
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()


class Gemma4RAGProvider:
    def __init__(self, model: str = GEMMA_MODEL, timeout_seconds: float | None = None) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _request_text(request: RAGModelInput) -> str:
        evidence_text = "\n\n".join(
            f"[{item.id}] ({item.source_type}, pages {item.page_start or '?'}-{item.page_end or item.page_start or '?'})\n{item.content[:MAX_EVIDENCE_CHARS]}"
            for item in request.evidence[:MAX_EVIDENCE_ITEMS]
        )
        history_text = "\n".join(
            f"{turn.role.title()}: {turn.content[:MAX_HISTORY_CHARS]}"
            for turn in request.history[-MAX_HISTORY_TURNS:]
        )
        conversation_context = f"\nConversation so far:\n{history_text}\n" if history_text else ""
        return (
            f"Answer level: {request.explanation_level}{conversation_context}\n"
            f"Current question: {request.question}\n\nEvidence:\n{evidence_text}"
        )

    @staticmethod
    def _image_bytes(image) -> tuple[bytes, str]:
        source = image.base64 or image.url
        if not source:
            raise ValueError(f"Image {image.id} has no inline data")
        mime_type = "image/webp"
        encoded = source
        if source.startswith("data:"):
            header, separator, encoded = source.partition(",")
            if not separator or ";base64" not in header:
                raise ValueError(f"Image {image.id} is not a base64 data URL")
            mime_type = header[5:].split(";", 1)[0] or mime_type
        elif source.startswith(("http://", "https://")):
            raise ValueError("Gemma RAG only accepts PaperScope's inline page images")
        return base64.b64decode(encoded, validate=True), mime_type

    @staticmethod
    def _compressed_image_part(types, image):
        data, _ = Gemma4RAGProvider._image_bytes(image)
        try:
            with Image.open(io.BytesIO(data)) as source:
                raster = source.convert("RGB")
                raster.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
                buffer = io.BytesIO()
                raster.save(buffer, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Gemma could not decode image {image.id}") from exc
        return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")

    def _answer_sync(self, request: RAGModelInput) -> str:
        client, types = _client(self.timeout_seconds)
        parts = [types.Part.from_text(text=self._request_text(request))]
        for image in request.images:
            parts.append(types.Part.from_text(text=f"{image.id}, page {image.page_number}, {image.label or image.type}:"))
            parts.append(self._compressed_image_part(types, image))
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=RAG_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=1200,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                ),
            )
        except Exception as exc:
            if retryable := _retryable_error(exc):
                raise retryable from exc
            raise
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        answer = getattr(response, "text", None)
        if not answer:
            raise RetryableProviderError("Gemma returned an empty RAG response")
        return str(answer)

    async def answer(self, request: RAGModelInput) -> RAGAnswer:
        import time

        started = time.perf_counter()
        answer = await asyncio.to_thread(self._answer_sync, request)
        return RAGAnswer(
            answer=answer,
            citations=[],
            model=self.model,
            fallback_used=False,
            grounded=False,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
