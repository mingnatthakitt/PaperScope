import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from paperscope.models.tables import PaperPage, PaperVisual
from paperscope.providers.contracts import ProviderConfigurationError, RetryableProviderError
from paperscope.rag.context import build_multimodal_context, resolve_images, should_attach_images
from paperscope.rag.retrieval import retrieve_all
from paperscope.rag.service import CircuitBreaker, RagService
from paperscope.schemas import ConversationTurn, RAGAnswer, RAGModelInput, RAGRequest, RetrievedContext


@pytest.fixture
def rag_test_settings(monkeypatch):
    test_settings = SimpleNamespace(
        nvidia_api_key="test-nim-key",
        google_ai_api_key="test-google-key",
        rag_primary_max_retries=2,
        rag_fallback_max_retries=2,
        rag_gemma_max_retries=1,
        rag_timeout_seconds=60,
        rag_provider_timeout_seconds=18,
        circuit_breaker_failures=3,
        circuit_breaker_cooldown_seconds=90,
    )
    monkeypatch.setattr("paperscope.rag.service.settings", test_settings)
    return test_settings


def _rag_context(paper_id):
    evidence = [
        RetrievedContext(
            id="raw-chunk",
            paper_id=paper_id,
            source_type="text",
            content="The throughput curve plateaus after batch size 32.",
            similarity=0.9,
        )
    ]
    context = RAGModelInput(
        question="Why does throughput flatten?",
        history=[ConversationTurn(role="user", content="What does the plot show?")],
        evidence=[evidence[0].model_copy(update={"id": "E1"})],
        images=[{"id": "page:8", "type": "page", "pageNumber": 8, "base64": "image-data", "label": "Figure 5"}],
    )
    return evidence, context


def test_evidence_gets_stable_public_citation_ids() -> None:
    paper_id = uuid4()
    evidence = [
        RetrievedContext(id="chunk:1", paper_id=paper_id, source_type="text", content="method", similarity=0.9),
        RetrievedContext(id="visual:2", paper_id=paper_id, source_type="visual", content="plot", similarity=0.8),
        RetrievedContext(id="graph:3", paper_id=paper_id, source_type="graph_node", content="node", similarity=0.7),
    ]

    labeled = RagService._with_citation_ids(evidence)
    assert [item.id for item in labeled] == ["E1", "V1", "G1"]


def test_graph_citation_resolves_its_evidence_asset() -> None:
    paper_id = uuid4()
    visual_id = uuid4()

    class FakeSession:
        def get(self, model, identifier):
            if model is PaperVisual and identifier == visual_id:
                return SimpleNamespace(id=visual_id, page_number=8)
            return None

    context = RAGModelInput(
        question="What does the graph show?",
        evidence=[
            RetrievedContext(
                id="G1",
                paper_id=paper_id,
                source_type="graph_node",
                content="Throughput depends on dispatch.",
                similarity=0.9,
                metadata={"evidenceIds": [str(visual_id)]},
            )
        ],
        images=[],
    )

    records = RagService._citation_records(FakeSession(), context)

    assert records[0]["page"] == 8
    assert records[0]["asset_id"] == visual_id


def test_graph_evidence_attaches_the_referenced_page_image(monkeypatch, tmp_path) -> None:
    paper_id = uuid4()
    page_id = uuid4()
    image_path = tmp_path / "page.webp"
    image_path.write_bytes(b"webp-fixture")
    page = SimpleNamespace(id=page_id, paper_id=paper_id, page_number=1, image_path="pages/001.webp")

    class FakeResult:
        def all(self):
            return [page]

    class FakeSession:
        def get(self, model, identifier):
            if model is PaperPage and identifier == page_id:
                return page
            return None

        def scalars(self, statement):
            return FakeResult()

    monkeypatch.setattr("paperscope.rag.context._safe_asset_path", lambda relative_path: image_path)
    evidence = RetrievedContext(
        id="G1",
        paper_id=paper_id,
        source_type="graph_node",
        content="The pipeline produces evidence.",
        similarity=0.9,
        metadata={"evidenceIds": [str(page_id)]},
    )

    images = resolve_images(FakeSession(), "What does the diagram show?", [evidence])

    assert [(image.type, image.page_number) for image in images] == [("page", 1)]


def test_text_question_does_not_attach_images_even_when_visual_evidence_is_retrieved() -> None:
    evidence = RetrievedContext(
        id="V1",
        paper_id=uuid4(),
        source_type="visual",
        content="Figure 5 shows throughput by batch size.",
        similarity=0.9,
    )

    assert not should_attach_images("What is the main contribution?", [evidence])


def test_multimodal_context_preserves_follow_up_history() -> None:
    context = build_multimodal_context(
        object(),
        "Why does it flatten?",
        [RetrievedContext(id="E1", paper_id=uuid4(), source_type="text", content="throughput", similarity=0.9)],
        history=[
            ConversationTurn(role="user", content="What is the main result?"),
            ConversationTurn(role="assistant", content="The method improves throughput.")
        ],
    )

    assert [turn.content for turn in context.history] == ["What is the main result?", "The method improves throughput."]


def test_visual_follow_up_keeps_the_relevant_image(monkeypatch, tmp_path) -> None:
    paper_id = uuid4()
    visual_id = uuid4()
    image_path = tmp_path / "page.webp"
    image_path.write_bytes(b"webp-fixture")
    visual = SimpleNamespace(
        id=visual_id,
        paper_id=paper_id,
        page_number=5,
        visual_type="plot",
        figure_label="Figure 1",
        page_image_path="pages/005.webp",
        cropped_image_path=None,
    )
    page = SimpleNamespace(id=uuid4(), paper_id=paper_id, page_number=5, image_path="pages/005.webp")

    class FakeResult:
        def all(self):
            return [page]

    class FakeSession:
        def get(self, model, identifier):
            if model is PaperVisual and identifier == visual_id:
                return visual
            return None

        def scalars(self, statement):
            return FakeResult()

    evidence = RetrievedContext(
        id="V1",
        paper_id=paper_id,
        source_type="visual",
        content="Figure 1 shows throughput.",
        similarity=0.9,
        page_start=5,
        page_end=5,
        image_refs=[str(visual_id)],
    )

    from paperscope.rag import context as context_module

    monkeypatch.setattr(context_module, "_safe_asset_path", lambda relative_path: image_path)
    images = resolve_images(
        FakeSession(),
        "Why is that important?",
        [evidence],
        history=[ConversationTurn(role="user", content="What does Figure 1 show?")],
    )

    assert len(images) == 1
    assert images[0].id == f"visual:{visual_id}"


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failure_limit() -> None:
    breaker = CircuitBreaker(failure_limit=2, cooldown_seconds=60)
    assert await breaker.allow()
    await breaker.failure()
    assert await breaker.allow()
    await breaker.failure()
    assert not await breaker.allow()
    await breaker.success()
    assert await breaker.allow()


@pytest.mark.asyncio
async def test_rag_uses_nemotron_after_primary_failures(monkeypatch, rag_test_settings) -> None:
    paper_id = uuid4()
    evidence = [
        RetrievedContext(
            id="E1",
            paper_id=paper_id,
            source_type="text",
            content="The method improves throughput.",
            similarity=0.9,
        )
    ]
    context = RAGModelInput(question="What is the result?", evidence=evidence, images=[])
    calls: list[str] = []

    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    class FailingMuse:
        def __init__(self, timeout_seconds=None):
            self.model = "muse-test"
            self.timeout_seconds = timeout_seconds

        async def answer(self, request: RAGModelInput) -> RAGAnswer:
            calls.append(self.model)
            raise RetryableProviderError("Muse unavailable")

    class WorkingNemotron:
        def __init__(self, timeout_seconds=None):
            self.model = "nemotron-test"
            self.timeout_seconds = timeout_seconds

        async def answer(self, request: RAGModelInput) -> RAGAnswer:
            calls.append(self.model)
            return RAGAnswer(
                answer="The method improves throughput. [E1]",
                citations=[],
                model=self.model,
                fallback_used=True,
                grounded=False,
                latency_ms=1,
            )

    monkeypatch.setattr("paperscope.rag.service.retrieve_all", lambda session, paper_ids, vector: evidence)
    monkeypatch.setattr("paperscope.rag.service.build_multimodal_context", lambda *args, **kwargs: context)
    monkeypatch.setattr("paperscope.rag.service.NimMuseProvider", FailingMuse)
    monkeypatch.setattr("paperscope.rag.service.NimNemotronProvider", WorkingNemotron)

    service = RagService()
    service.embeddings = FakeEmbeddings()
    result = await service.answer(
        object(),
        RAGRequest(paperIds=[paper_id], question="What is the result?"),
    )

    assert result.model == "nemotron-test"
    assert result.fallback_used
    assert calls == ["muse-test", "muse-test", "muse-test", "nemotron-test"]


@pytest.mark.asyncio
async def test_rag_uses_all_configured_nemotron_attempts(monkeypatch, rag_test_settings) -> None:
    paper_id = uuid4()
    evidence = [RetrievedContext(id="E1", paper_id=paper_id, source_type="text", content="result", similarity=0.9)]
    context = RAGModelInput(question="What is the result?", evidence=evidence, images=[])
    calls: list[str] = []

    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    class FailingMuse:
        def __init__(self, timeout_seconds=None):
            self.model = "muse-test"
            self.timeout_seconds = timeout_seconds

        async def answer(self, request: RAGModelInput) -> RAGAnswer:
            calls.append(self.model)
            raise RetryableProviderError("Muse unavailable")

    class NemotronWithTwoFailures:
        def __init__(self, timeout_seconds=None):
            self.model = "nemotron-test"
            self.timeout_seconds = timeout_seconds
            self.attempts = 0

        async def answer(self, request: RAGModelInput) -> RAGAnswer:
            calls.append(self.model)
            self.attempts += 1
            if self.attempts < 3:
                raise RetryableProviderError("Nemotron unavailable")
            return RAGAnswer(answer="The result is [E1].", citations=[], model=self.model, latency_ms=1)

    monkeypatch.setattr("paperscope.rag.service.retrieve_all", lambda session, paper_ids, vector: evidence)
    monkeypatch.setattr("paperscope.rag.service.build_multimodal_context", lambda *args, **kwargs: context)
    monkeypatch.setattr("paperscope.rag.service.NimMuseProvider", FailingMuse)
    monkeypatch.setattr("paperscope.rag.service.NimNemotronProvider", NemotronWithTwoFailures)
    service = RagService()
    rag_test_settings.rag_primary_max_retries = 0
    rag_test_settings.rag_fallback_max_retries = 2
    rag_test_settings.rag_provider_timeout_seconds = 1
    service.embeddings = FakeEmbeddings()
    result = await service.answer(object(), RAGRequest(paperIds=[paper_id], question="What is the result?"))

    assert result.model == "nemotron-test"
    assert calls == ["muse-test", "nemotron-test", "nemotron-test", "nemotron-test"]


@pytest.mark.parametrize(
    ("preferred", "expected_order"),
    [
        ("muse", ["muse", "nemotron", "gemma"]),
        ("nemotron", ["nemotron", "muse", "gemma"]),
        ("gemma", ["gemma", "muse", "nemotron"]),
    ],
)
@pytest.mark.asyncio
async def test_rag_selected_provider_runs_first_and_fallbacks_keep_the_same_request(
    monkeypatch, rag_test_settings, preferred, expected_order
) -> None:
    paper_id = uuid4()
    evidence, context = _rag_context(paper_id)
    calls = []

    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    class ScriptedProvider:
        def __init__(self, key):
            self.key = key
            self.model = f"{key}-test"
            self.timeout_seconds = None

        async def answer(self, request: RAGModelInput) -> RAGAnswer:
            calls.append((self.key, request))
            if self.key != expected_order[-1]:
                raise RetryableProviderError(f"{self.key} unavailable")
            return RAGAnswer(answer="The curve plateaus after 32. [E1]", model=self.model, latency_ms=1)

    monkeypatch.setattr("paperscope.rag.service.retrieve_all", lambda *_args: evidence)
    monkeypatch.setattr("paperscope.rag.service.build_multimodal_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(
        RagService,
        "_provider",
        staticmethod(lambda key, _timeout: ScriptedProvider(key)),
    )
    rag_test_settings.rag_primary_max_retries = 0
    rag_test_settings.rag_fallback_max_retries = 0
    rag_test_settings.rag_gemma_max_retries = 0
    service = RagService()
    service.embeddings = FakeEmbeddings()

    result = await service.answer(
        object(),
        RAGRequest(paperIds=[paper_id], question="Why does throughput flatten?", answerModel=preferred),
    )

    assert [key for key, _request in calls] == expected_order
    assert all(request is context for _key, request in calls)
    assert result.model == f"{expected_order[-1]}-test"
    assert result.fallback_used is True
    assert result.grounded is True


@pytest.mark.parametrize("preferred", ["muse", "nemotron", "gemma"])
@pytest.mark.asyncio
async def test_rag_stops_after_the_selected_provider_succeeds(monkeypatch, rag_test_settings, preferred) -> None:
    paper_id = uuid4()
    evidence, context = _rag_context(paper_id)
    calls = []

    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    class WorkingProvider:
        def __init__(self, key):
            self.model = f"{key}-test"
            self.key = key
            self.timeout_seconds = None

        async def answer(self, request: RAGModelInput) -> RAGAnswer:
            calls.append((self.key, request))
            return RAGAnswer(answer="The evidence supports this. [E1]", model=self.model, latency_ms=1)

    monkeypatch.setattr("paperscope.rag.service.retrieve_all", lambda *_args: evidence)
    monkeypatch.setattr("paperscope.rag.service.build_multimodal_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(RagService, "_provider", staticmethod(lambda key, _timeout: WorkingProvider(key)))
    service = RagService()
    service.embeddings = FakeEmbeddings()

    result = await service.answer(
        object(),
        RAGRequest(paperIds=[paper_id], question="Why does throughput flatten?", answerModel=preferred),
    )

    assert calls == [(preferred, context)]
    assert result.model == f"{preferred}-test"
    assert result.fallback_used is False
    assert result.grounded is True


@pytest.mark.asyncio
async def test_rag_skips_providers_without_keys_and_uses_gemma_if_available(
    monkeypatch, rag_test_settings
) -> None:
    paper_id = uuid4()
    evidence, context = _rag_context(paper_id)
    rag_test_settings.nvidia_api_key = None
    calls = []

    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    class GemmaOnly:
        model = "gemma-test"
        timeout_seconds = None

        async def answer(self, request: RAGModelInput) -> RAGAnswer:
            calls.append(request)
            return RAGAnswer(answer="The evidence supports this. [E1]", model=self.model, latency_ms=1)

    monkeypatch.setattr("paperscope.rag.service.retrieve_all", lambda *_args: evidence)
    monkeypatch.setattr("paperscope.rag.service.build_multimodal_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(
        RagService,
        "_provider",
        staticmethod(lambda key, _timeout: GemmaOnly() if key == "gemma" else pytest.fail(f"unexpected {key}")),
    )
    service = RagService()
    service.embeddings = FakeEmbeddings()

    result = await service.answer(
        object(),
        RAGRequest(paperIds=[paper_id], question="Why does throughput flatten?", answerModel="muse"),
    )

    assert calls == [context]
    assert result.model == "gemma-test"
    assert result.fallback_used is True


@pytest.mark.asyncio
async def test_rag_requires_at_least_one_provider_key(monkeypatch, rag_test_settings) -> None:
    paper_id = uuid4()
    evidence, context = _rag_context(paper_id)
    rag_test_settings.nvidia_api_key = None
    rag_test_settings.google_ai_api_key = None

    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    monkeypatch.setattr("paperscope.rag.service.retrieve_all", lambda *_args: evidence)
    monkeypatch.setattr("paperscope.rag.service.build_multimodal_context", lambda *_args, **_kwargs: context)
    service = RagService()
    service.embeddings = FakeEmbeddings()

    with pytest.raises(ProviderConfigurationError, match="Configure NVIDIA_API_KEY or GOOGLE_AI_API_KEY"):
        await service.answer(object(), RAGRequest(paperIds=[paper_id], question="Why does throughput flatten?"))


@pytest.mark.asyncio
async def test_rag_reserves_time_for_later_providers_and_honors_exhausted_deadline(
    monkeypatch, rag_test_settings
) -> None:
    paper_id = uuid4()
    evidence, context = _rag_context(paper_id)
    rag_test_settings.rag_primary_max_retries = 0
    rag_test_settings.rag_fallback_max_retries = 0
    rag_test_settings.rag_gemma_max_retries = 0
    provider_deadlines = []

    class FakeEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    async def exhaust_provider(self, provider, _request, deadline):
        provider_deadlines.append((provider.model, deadline - time.monotonic()))
        raise RetryableProviderError("simulated provider failure")

    monkeypatch.setattr("paperscope.rag.service.retrieve_all", lambda *_args: evidence)
    monkeypatch.setattr("paperscope.rag.service.build_multimodal_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(RagService, "_provider", staticmethod(lambda key, _timeout: SimpleNamespace(model=key)))
    monkeypatch.setattr(RagService, "_answer_with_deadline", exhaust_provider)
    service = RagService()
    service.embeddings = FakeEmbeddings()

    with pytest.raises(RetryableProviderError, match="All configured RAG providers failed"):
        await service.answer(object(), RAGRequest(paperIds=[paper_id], question="Why does throughput flatten?"))

    assert [model for model, _remaining in provider_deadlines] == ["muse", "nemotron", "gemma"]
    assert provider_deadlines[1][1] > provider_deadlines[0][1] + 15
    assert provider_deadlines[2][1] > provider_deadlines[1][1] + 15


@pytest.mark.asyncio
async def test_rag_does_not_start_a_provider_after_its_deadline(rag_test_settings) -> None:
    paper_id = uuid4()
    _evidence, context = _rag_context(paper_id)
    touched = False

    class NeverCalled:
        model = "already-too-late"

        async def answer(self, _request):
            nonlocal touched
            touched = True

    with pytest.raises(RetryableProviderError, match="time budget exhausted"):
        await RagService._answer_with_deadline(NeverCalled(), context, time.monotonic() - 1)
    assert touched is False


def test_retrieval_reserves_evidence_for_each_selected_paper(monkeypatch) -> None:
    first_paper = uuid4()
    second_paper = uuid4()
    candidates = [
        RetrievedContext(id="first-best", paper_id=first_paper, source_type="text", content="first best", similarity=0.99),
        RetrievedContext(id="first-second", paper_id=first_paper, source_type="text", content="first second", similarity=0.98),
        RetrievedContext(id="second-best", paper_id=second_paper, source_type="text", content="second best", similarity=0.50),
    ]
    monkeypatch.setattr("paperscope.rag.retrieval.retrieve_chunks", lambda *args: candidates)
    monkeypatch.setattr("paperscope.rag.retrieval.retrieve_visuals", lambda *args: [])
    monkeypatch.setattr("paperscope.rag.retrieval.retrieve_graph_nodes", lambda *args: [])

    result = retrieve_all(object(), [first_paper, second_paper], [1.0])

    assert [item.id for item in result] == ["first-best", "second-best", "first-second"]
