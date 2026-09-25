from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from paperscope.models.tables import IngestionJob, Paper
from paperscope.models.tables import PaperAnalysis as PaperAnalysisRow
from paperscope.processing.ingest import IngestionPipeline
from paperscope.providers.gemma import (
    MAX_BATCH_TEXT_CHARS,
    MAX_PAGE_TEXT_CHARS,
    Gemma4RAGProvider,
    GemmaDocumentAnalysisProvider,
    GemmaPageBatch,
    GemmaPageNote,
    _page_batches,
)
from paperscope.schemas import (
    AnalysisLevels,
    ConversationTurn,
    DeepPaperAnalysis,
    GraphNode,
    PaperAnalysis,
    PaperGraph,
    RAGModelInput,
    RetrievedContext,
)


def _analysis() -> DeepPaperAnalysis:
    level = PaperAnalysis(
        summary="A concise summary.",
        problem="A research problem.",
        motivation="A motivating gap.",
        key_insight="A useful insight.",
        method="A method.",
        results="A result.",
    )
    return DeepPaperAnalysis(
        levels=AnalysisLevels(simple=level, student=level, researcher=level),
        graph=PaperGraph(
            nodes=[GraphNode(id="method", label="Method", type="process", explanation="The proposed method.")]
        ),
    )


class _FakePart:
    @staticmethod
    def from_text(*, text: str):
        return ("text", text)

    @staticmethod
    def from_bytes(*, data: bytes, mime_type: str):
        return ("image", data, mime_type)


class _FakeTypes:
    Part = _FakePart

    @staticmethod
    def GenerateContentConfig(**kwargs):
        return kwargs

    @staticmethod
    def ThinkingConfig(**kwargs):
        return kwargs


def _png_bytes(width: int = 32, height: int = 24) -> bytes:
    image = Image.new("RGB", (width, height), color=(30, 150, 130))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_ingestion_runs_local_embeddings_after_gemma_fallback(monkeypatch, tmp_path: Path) -> None:
    paper_id = uuid4()
    job_id = uuid4()
    original = tmp_path / "original.pdf"
    original.write_bytes(b"pdf placeholder")
    paper = SimpleNamespace(
        id=paper_id,
        sha256="a" * 64,
        original_path="original.pdf",
        title="Original title",
        page_count=1,
        status="processing",
        metadata_json={"analysisModel": "gemini-3.8-flash"},
    )
    job = SimpleNamespace(
        id=job_id,
        paper_id=paper_id,
        stage="analyzing",
        status="running",
        progress=50,
        message="Analyzing",
    )
    page = SimpleNamespace(page_number=1, image_path="pages/001.webp", text_content="Extracted text")
    chunk = SimpleNamespace(content="chunk to embed")

    class ScalarResult:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

    class FakeSession:
        def __init__(self):
            self.read_count = 0
            self.added = []

        def get(self, model, identifier):
            if model is IngestionJob and identifier == job_id:
                return job
            if model is Paper and identifier == paper_id:
                return paper
            return None

        def scalars(self, _statement):
            self.read_count += 1
            return ScalarResult({1: [page], 2: [chunk], 3: [], 4: [chunk], 5: []}.get(self.read_count, []))

        def scalar(self, _statement):
            return None

        def add(self, row):
            self.added.append(row)

        def flush(self):
            pass

        def commit(self):
            pass

    session = FakeSession()
    embedding_calls = []

    class GemmaFallback:
        def __init__(self, model):
            assert model == "gemini-3.8-flash"
            self.model_used = "gemma-4-31b-it"

        async def analyze_pdf(self, _pdf_path, **kwargs):
            assert kwargs["pages"][0].page_number == 1
            assert kwargs["page_images"][1].name == "001.webp"
            return _analysis()

    monkeypatch.setattr("paperscope.processing.ingest.settings", SimpleNamespace(
        papers_root=tmp_path,
        paper_analysis_model="gemini-3.8-flash",
        analysis_prompt_version="analysis-v1",
        gemma_analysis_prompt_version="gemma-pages-v1",
    ))
    monkeypatch.setattr("paperscope.processing.ingest.GeminiDocumentAnalysisProvider", GemmaFallback)
    pipeline = IngestionPipeline(session)
    monkeypatch.setattr(pipeline, "_update_job", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_clear_analysis_outputs", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_persist_visuals", lambda *_args: [])
    monkeypatch.setattr(pipeline, "_persist_graph", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_write_manifest", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_embed", lambda chunks, visuals, nodes: embedding_calls.append((chunks, visuals, nodes)))

    pipeline.process(job_id)

    stored_analysis = next(row for row in session.added if isinstance(row, PaperAnalysisRow))
    assert stored_analysis.model == "gemma-4-31b-it"
    assert stored_analysis.prompt_version == "gemma-pages-v1"
    assert paper.metadata_json["analysisFallbackUsed"] is True
    assert paper.metadata_json["analysisModelUsed"] == "gemma-4-31b-it"
    assert embedding_calls == [([chunk], [], [])]
    assert paper.status == "ready" and job.status == "succeeded"


def test_page_batches_bound_both_page_count_and_text_size() -> None:
    pages = [SimpleNamespace(page_number=index, text="x" * (MAX_PAGE_TEXT_CHARS + 100)) for index in range(1, 12)]

    batches = _page_batches(pages)

    assert [len(batch) for batch in batches] == [2, 2, 2, 2, 2, 1]
    assert all(len(batch) <= 10 for batch in batches)
    assert all(
        sum(len(GemmaDocumentAnalysisProvider._page_text(page)) for page in batch) <= MAX_BATCH_TEXT_CHARS
        for batch in batches
    )
    assert all("Page text truncated" in GemmaDocumentAnalysisProvider._page_text(page) for page in pages)


def test_gemma_indexing_sends_extracted_text_and_resized_rendered_image(tmp_path: Path) -> None:
    page_image = tmp_path / "page.png"
    page_image.write_bytes(_png_bytes(1800, 1000))
    page = SimpleNamespace(page_number=7, text="The extracted page text is preserved.")
    result = GemmaPageBatch(page_notes=[GemmaPageNote(page_number=7, notes="The page introduces the method.")])
    captured = {}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return SimpleNamespace(text=result.model_dump_json(by_alias=True))

    provider = GemmaDocumentAnalysisProvider()
    parsed = provider._analyze_batch_sync(
        SimpleNamespace(models=FakeModels()),
        _FakeTypes,
        [page],
        {7: page_image},
    )

    assert parsed.page_notes[0].page_number == 7
    assert captured["model"] == "gemma-4-31b-it"
    assert any(part[0] == "text" and "The extracted page text is preserved." in part[1] for part in captured["contents"])
    image_parts = [part for part in captured["contents"] if part[0] == "image"]
    assert len(image_parts) == 1
    assert image_parts[0][2] == "image/jpeg"
    with Image.open(io.BytesIO(image_parts[0][1])) as image:
        assert image.width <= 1280 and image.height <= 1280


@pytest.mark.asyncio
async def test_gemma_analysis_batch_cache_resumes_and_rejects_wrong_page_coverage(monkeypatch, tmp_path: Path) -> None:
    client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("paperscope.providers.gemma._client", lambda *args: (client, _FakeTypes))
    pages = [SimpleNamespace(page_number=1, text="first"), SimpleNamespace(page_number=2, text="second")]
    cache_dir = tmp_path / "gemma-cache"
    images = {page.page_number: tmp_path / f"{page.page_number}.webp" for page in pages}
    valid_batch = GemmaPageBatch(
        page_notes=[GemmaPageNote(page_number=1, notes="One."), GemmaPageNote(page_number=2, notes="Two.")],
        important_visuals=[
            {
                "key": "figure-1",
                "page": 1,
                "type": "plot",
                "reason": "Shows the main result.",
                "visualDescription": "Throughput rises then plateaus.",
                "visualInterpretation": "Gains taper at larger batch sizes.",
            }
        ],
    )
    first_provider = GemmaDocumentAnalysisProvider()
    batch_calls = 0

    def analyze_batch(_client, _types, batch, _images):
        nonlocal batch_calls
        batch_calls += 1
        return valid_batch

    monkeypatch.setattr(first_provider, "_analyze_batch_sync", analyze_batch)
    monkeypatch.setattr(first_provider, "_synthesize_sync", lambda *_args: _analysis())
    progress = []
    result = await first_provider.analyze_pages(
        pages, images, cache_dir, 0, lambda completed, total: progress.append((completed, total))
    )
    assert batch_calls == 1
    assert progress == [(1, 1)]
    assert result.important_visuals[0].key == "page-1-figure-1"

    resumed_provider = GemmaDocumentAnalysisProvider()

    def unexpected_batch_call(*_args):
        raise AssertionError("a valid cached batch should be reused")

    monkeypatch.setattr(resumed_provider, "_analyze_batch_sync", unexpected_batch_call)
    monkeypatch.setattr(resumed_provider, "_synthesize_sync", lambda *_args: _analysis())
    await resumed_provider.analyze_pages(pages, images, cache_dir, 0)
    assert batch_calls == 1

    cache_path = resumed_provider._cache_path(cache_dir, 1, pages)
    invalid_batch = GemmaPageBatch(page_notes=[GemmaPageNote(page_number=99, notes="Wrong page.")])
    cache_path.write_text(invalid_batch.model_dump_json(by_alias=True), encoding="utf-8")
    recovery_provider = GemmaDocumentAnalysisProvider()
    recovered_calls = 0

    def recover_batch(_client, _types, batch, _images):
        nonlocal recovered_calls
        recovered_calls += 1
        return valid_batch

    monkeypatch.setattr(recovery_provider, "_analyze_batch_sync", recover_batch)
    monkeypatch.setattr(recovery_provider, "_synthesize_sync", lambda *_args: _analysis())
    await recovery_provider.analyze_pages(pages, images, cache_dir, 0)
    assert recovered_calls == 1


@pytest.mark.asyncio
async def test_gemma_rag_provider_sends_history_evidence_and_requested_images(monkeypatch) -> None:
    captured = {}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured.update(model=model, contents=contents, config=config)
            return SimpleNamespace(text="The chart supports the claim. [E1] [V1]")

    client = SimpleNamespace(models=FakeModels(), close=lambda: None)

    def fake_client(timeout):
        captured["timeout"] = timeout
        return client, _FakeTypes

    monkeypatch.setattr("paperscope.providers.gemma._client", fake_client)
    request = RAGModelInput(
        question="Why does throughput flatten?",
        history=[ConversationTurn(role="user", content="What does Figure 2 show?")],
        evidence=[
            RetrievedContext(
                id="E1",
                paper_id=uuid4(),
                source_type="text",
                content="Throughput stops improving after batch size 32.",
                similarity=0.9,
            ),
            RetrievedContext(
                id="V1",
                paper_id=uuid4(),
                source_type="visual",
                content="Figure 2 shows a throughput curve that plateaus.",
                similarity=0.8,
            ),
        ],
        images=[
            {
                "id": "page:2",
                "type": "page",
                "pageNumber": 2,
                "base64": f"data:image/png;base64,{base64.b64encode(_png_bytes()).decode('ascii')}",
                "label": "Figure 2",
            }
        ],
    )

    result = await Gemma4RAGProvider(timeout_seconds=5).answer(request)

    text_parts = [part[1] for part in captured["contents"] if part[0] == "text"]
    image_parts = [part for part in captured["contents"] if part[0] == "image"]
    assert result.answer.endswith("[E1] [V1]")
    assert captured["model"] == "gemma-4-31b-it"
    assert captured["timeout"] == 5
    assert any("What does Figure 2 show?" in text for text in text_parts)
    assert any("Current question: Why does throughput flatten?" in text for text in text_parts)
    assert any("Throughput stops improving after batch size 32." in text for text in text_parts)
    assert len(image_parts) == 1 and image_parts[0][2] == "image/jpeg"
