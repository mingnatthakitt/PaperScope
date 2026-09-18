import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

from paperscope.config import analysis_model_chain
from paperscope.providers.contracts import RetryableProviderError
from paperscope.providers.gemini import GeminiDocumentAnalysisProvider, _retryable_provider_error, _status_code
from paperscope.providers.nim import NIM_CONCISE_RESPONSE_BODY, NimProvider, _retryable_nim_error
from paperscope.schemas import ConversationTurn, DeepPaperAnalysis, RAGModelInput, RetrievedContext


def test_gemini_sdk_error_code_is_classified_as_retryable() -> None:
    error = SimpleNamespace(code=503)

    assert _status_code(error) == "503"
    assert _retryable_provider_error(error) is not None


def test_non_transient_provider_error_is_not_retried() -> None:
    error = SimpleNamespace(code=400)

    assert _retryable_provider_error(error) is None


def test_gemini_server_disconnect_is_retryable() -> None:
    error = RuntimeError("Server disconnected without sending a response.")

    assert _retryable_provider_error(error) is not None


def test_nim_prompt_includes_prior_conversation() -> None:
    provider = object.__new__(NimProvider)
    request = RAGModelInput(
        question="Why does it flatten?",
        history=[ConversationTurn(role="user", content="What is the main result?")],
        evidence=[RetrievedContext(id="E1", paper_id=uuid4(), source_type="text", content="throughput", similarity=0.9)],
        images=[],
    )

    content = provider._content_parts(request)[0]["text"]

    assert isinstance(content, str)
    assert "Conversation so far:" in content
    assert "What is the main result?" in content
    assert "Current question: Why does it flatten?" in content


def test_nim_reasoning_is_disabled_for_user_facing_completion() -> None:
    assert NIM_CONCISE_RESPONSE_BODY == {"chat_template_kwargs": {"enable_thinking": False}}


def test_nim_read_timeout_is_retryable() -> None:
    assert _retryable_nim_error(RuntimeError("Request timed out."))


def test_gemini_model_chain_is_ordered_from_newest_to_oldest() -> None:
    assert analysis_model_chain("gemini-3.8-flash") == (
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
    )
    assert analysis_model_chain("gemini-3.7-flash") == ("gemini-3.7-flash", "gemini-3.6-flash")
    assert analysis_model_chain("gemini-3.6-flash") == ("gemini-3.6-flash",)


@pytest.mark.asyncio
async def test_gemini_provider_falls_back_to_the_next_model(monkeypatch, tmp_path: Path) -> None:
    provider = object.__new__(GeminiDocumentAnalysisProvider)
    provider.requested_model = "gemini-3.8-flash"
    provider.model_chain = analysis_model_chain(provider.requested_model)
    provider.model_used = provider.model_chain[0]
    calls: list[str] = []

    def fake_analyze(_pdf_path: Path, model: str):
        calls.append(model)
        if model != "gemini-3.6-flash":
            raise RetryableProviderError(f"{model} is unavailable")
        return object()

    monkeypatch.setattr(provider, "_analyze_sync", fake_analyze)

    result = await provider.analyze_pdf(tmp_path / "paper.pdf")

    assert result is not None
    assert calls == [
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.7-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
    ]
    assert provider.model_used == "gemini-3.6-flash"


@pytest.mark.asyncio
async def test_gemini_reuses_one_uploaded_file_across_model_retries(monkeypatch, tmp_path: Path) -> None:
    upload_calls = 0
    generate_calls: list[str] = []
    analysis = DeepPaperAnalysis.model_construct()

    class FakeFile:
        name = "files/paper"
        state = "ACTIVE"

    class FakeFiles:
        def upload(self, **kwargs):
            nonlocal upload_calls
            upload_calls += 1
            return FakeFile()

        def get(self, **kwargs):
            return FakeFile()

    class FakeModels:
        def generate_content(self, *, model, **kwargs):
            generate_calls.append(model)
            if model != "gemini-3.6-flash":
                error = RuntimeError("Gemini temporarily unavailable")
                error.code = 503  # type: ignore[attr-defined]
                raise error
            return SimpleNamespace(parsed=analysis)

    class FakeClient:
        files = FakeFiles()
        models = FakeModels()

    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    types_module = ModuleType("google.genai.types")
    genai_module.Client = lambda **kwargs: FakeClient()
    types_module.GenerateContentConfig = lambda **kwargs: kwargs
    google_module.genai = genai_module
    genai_module.types = types_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)

    provider = object.__new__(GeminiDocumentAnalysisProvider)
    provider.requested_model = "gemini-3.8-flash"
    provider.model_chain = analysis_model_chain(provider.requested_model)
    provider.model_used = provider.model_chain[0]
    provider._gemini_client = None
    provider._uploaded_file = None

    result = await provider.analyze_pdf(tmp_path / "paper.pdf")

    assert result is analysis
    assert upload_calls == 1
    assert generate_calls == [
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.7-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
    ]
