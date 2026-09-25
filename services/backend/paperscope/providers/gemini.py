from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from paperscope.config import GEMMA_MODEL, analysis_model_chain, settings
from paperscope.providers.contracts import (
    ProviderAttemptsExhaustedError,
    ProviderConfigurationError,
    RetryableProviderError,
)
from paperscope.providers.gemma import GemmaDocumentAnalysisProvider
from paperscope.schemas import DeepPaperAnalysis

ANALYSIS_PROMPT = """
You are PaperScope's document analyst. Inspect the entire native PDF, including prose,
equations, diagrams, plots, and tables. Do not infer a graph from the abstract alone.

Return JSON matching the supplied schema. Produce simple, student, and researcher
explanations of the same paper. Identify important visuals with 1-based page numbers.
Use stable visual keys such as figure-3 or table-1. Include a normalized bounding box
[x, y, width, height] only when you can locate the visual reliably. Graph nodes should
contain 5-9 high-signal concepts and evidence_keys such as page-6 or figure-3.
Avoid unsupported claims. If a detail is not present, say so explicitly.
""".strip()

logger = logging.getLogger("paperscope.gemini")


def _status_code(error: Exception) -> str:
    """Read status codes from both google-genai and HTTP-style exceptions."""
    for attribute in ("code", "status_code"):
        value = getattr(error, attribute, None)
        if value is not None:
            return str(value)
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return str(value) if value is not None else ""


def _retryable_provider_error(error: Exception) -> RetryableProviderError | None:
    message = str(error).lower()
    transient_transport_error = any(
        marker in message
        for marker in ("timeout", "server disconnected", "remoteprotocolerror", "connection reset", "connection aborted")
    )
    if _status_code(error) in {"404", "408", "429", "500", "502", "503", "504"} or transient_transport_error:
        return RetryableProviderError(str(error))
    return None


class GeminiDocumentAnalysisProvider:
    def __init__(self, model: str | None = None) -> None:
        if not settings.google_ai_api_key:
            raise ProviderConfigurationError("GOOGLE_AI_API_KEY is not configured")
        self.requested_model = model or settings.paper_analysis_model
        self.model_chain = analysis_model_chain(self.requested_model)
        self.model_used = self.model_chain[0]
        # A single ingestion run should upload the canonical PDF once, then reuse
        # the temporary Gemini file across model retries and fallback models.
        self._gemini_client = None
        self._uploaded_file = None

    def _analyze_sync(self, pdf_path: Path, model: str) -> DeepPaperAnalysis:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ProviderConfigurationError("Install google-genai in the PaperScope environment") from exc

        client = self._gemini_client
        if client is None:
            client = genai.Client(api_key=settings.google_ai_api_key)
            self._gemini_client = client

        uploaded = self._uploaded_file
        if uploaded is None:
            try:
                uploaded = client.files.upload(file=str(pdf_path), config={"mime_type": "application/pdf"})
                self._uploaded_file = uploaded
            except Exception as exc:
                if retryable := _retryable_provider_error(exc):
                    raise retryable from exc
                raise
        file_name = getattr(uploaded, "name", None)
        deadline = time.monotonic() + 90
        while getattr(uploaded, "state", None) and str(uploaded.state).upper().endswith("PROCESSING"):
            if time.monotonic() > deadline:
                raise RetryableProviderError("Gemini file processing timed out")
            time.sleep(2)
            if file_name:
                try:
                    uploaded = client.files.get(name=file_name)
                    self._uploaded_file = uploaded
                except Exception as exc:
                    if _status_code(exc) == "404" or any(marker in str(exc).lower() for marker in ("expired", "not found")):
                        self._uploaded_file = None
                    if retryable := _retryable_provider_error(exc):
                        raise retryable from exc
                    raise

        if str(getattr(uploaded, "state", "")).upper().endswith("FAILED"):
            raise RetryableProviderError("Gemini failed to process the PDF")

        try:
            response = client.models.generate_content(
                model=model,
                contents=[uploaded, ANALYSIS_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DeepPaperAnalysis,
                    temperature=0.2,
                ),
            )
        except Exception as exc:
            # Gemini file handles are temporary. If a retry observes an expired
            # or missing handle, the next attempt must upload the local source PDF
            # again instead of reusing the stale URI.
            if _status_code(exc) == "404" or any(marker in str(exc).lower() for marker in ("expired", "not found")):
                self._uploaded_file = None
            if retryable := _retryable_provider_error(exc):
                raise retryable from exc
            raise

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            try:
                return DeepPaperAnalysis.model_validate(parsed)
            except ValueError:
                # Some google-genai versions return a plain dict here while
                # others return the Pydantic instance. Fall through to the text
                # representation only when the parsed payload is incomplete.
                pass
        raw = getattr(response, "text", None)
        if not raw:
            raise RetryableProviderError("Gemini returned an empty or invalid analysis payload")
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()
        try:
            return DeepPaperAnalysis.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise RetryableProviderError("Gemini returned an invalid analysis payload") from exc

    async def analyze_pdf(
        self,
        pdf_path: Path,
        *,
        pages: list[object] | None = None,
        page_images: dict[int, Path] | None = None,
        batch_cache_dir: Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DeepPaperAnalysis:
        last_error: RetryableProviderError | None = None
        self._gemini_client = None
        self._uploaded_file = None
        try:
            for model in self.model_chain:
                if model == GEMMA_MODEL:
                    if pages is None or page_images is None or batch_cache_dir is None:
                        raise ProviderConfigurationError(
                            "Gemma fallback requires extracted page text, rendered images, and a cache directory"
                        )
                    try:
                        provider = GemmaDocumentAnalysisProvider(model=model)
                        analysis = await provider.analyze_pages(
                            pages,
                            page_images,
                            batch_cache_dir,
                            settings.gemma_analysis_max_retries,
                            progress_callback,
                        )
                        self.model_used = model
                        logger.warning(
                            "Paper analysis fell back from Gemini Flash to %s",
                            model,
                        )
                        return analysis
                    except ProviderConfigurationError:
                        raise
                    except RetryableProviderError as exc:
                        last_error = exc
                        logger.warning("Gemma paper analysis fallback failed")
                    except Exception as exc:
                        last_error = RetryableProviderError(
                            f"Gemma paper analysis failed: {type(exc).__name__}"
                        )
                        logger.warning("Gemma paper analysis fallback failed: %s", type(exc).__name__)
                    continue

                for attempt in range(settings.gemini_model_max_retries + 1):
                    try:
                        analysis = await asyncio.to_thread(self._analyze_sync, pdf_path, model)
                        self.model_used = model
                        if model != self.requested_model:
                            logger.warning("Gemini indexing fell back from %s to %s", self.requested_model, model)
                        return analysis
                    except ProviderConfigurationError:
                        raise
                    except RetryableProviderError as exc:
                        last_error = exc
                        logger.warning(
                            "Gemini indexing model %s attempt %s/%s failed",
                            model,
                            attempt + 1,
                            settings.gemini_model_max_retries + 1,
                        )
                    except Exception as exc:
                        last_error = RetryableProviderError(
                            f"Gemini Flash model {model} rejected the analysis request: {type(exc).__name__}"
                        )
                        logger.warning(
                            "Gemini indexing model %s rejected the request; advancing to the next model",
                            model,
                        )
                        break
        finally:
            self._gemini_client = None
            self._uploaded_file = None
        assert last_error is not None
        raise ProviderAttemptsExhaustedError(
            f"Gemini indexing failed after trying {' → '.join(self.model_chain)}: {last_error}"
        ) from last_error
