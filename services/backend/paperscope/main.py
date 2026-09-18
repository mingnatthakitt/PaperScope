from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from paperscope.config import ANALYSIS_MODEL_FALLBACKS, analysis_model_chain, settings
from paperscope.db.session import database_ready, get_db
from paperscope.metrics import SlidingWindowRateLimiter, directory_size, metrics
from paperscope.models.tables import (
    IngestionJob,
    Paper,
    PaperAnalysis,
    PaperGraph,
    PaperGraphEdge,
    PaperGraphNode,
    PaperPage,
    PaperVisual,
)
from paperscope.processing.pdf import PDFLimitError, PDFValidationError, sha256_file, validate_pdf
from paperscope.providers.contracts import ProviderConfigurationError, RetryableProviderError
from paperscope.providers.discovery import ResearchDiscovery
from paperscope.rag.service import RagService
from paperscope.schemas import (
    ExplanationLevel,
    GraphEdge,
    GraphNodeResponse,
    GraphResponse,
    IngestResponse,
    IngestSource,
    JobStatusResponse,
    PaperResponse,
    RAGAnswer,
    RAGRequest,
    SearchRequest,
    SearchResponse,
    VisualResponse,
)
from paperscope.schemas import (
    PaperAnalysis as PaperAnalysisSchema,
)

logger = logging.getLogger("paperscope.api")
rag_rate_limiter = SlidingWindowRateLimiter(settings.rag_rate_limit_per_minute)

discovery = ResearchDiscovery()
rag_service = RagService()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    started = time.perf_counter()
    try:
        await asyncio.to_thread(rag_service.warm_embeddings)
        logger.info("Qwen embedding model warmed in %.2fs", time.perf_counter() - started)
    except Exception:
        # Keep the API available if the local model cannot load; the first RAG
        # request will surface the provider configuration error instead.
        logger.exception("Qwen embedding warm-up failed")
    yield


app = FastAPI(title="PaperScope API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    started = time.perf_counter()
    incoming_request_id = request.headers.get("x-request-id", "")
    request_id = incoming_request_id[:128] if incoming_request_id.isprintable() else ""
    request_id = request_id or str(uuid4())
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        metrics.record_request(request.url.path, 500, latency_ms)
        logger.exception(
            "request_id=%s method=%s path=%s status=500 latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            latency_ms,
        )
        raise
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    metrics.record_request(request.url.path, response.status_code, latency_ms)
    response.headers["x-request-id"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


def _asset_path(relative_path: str) -> Path:
    root = settings.papers_root.resolve()
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid asset path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return path


def _job_response(job: IngestionJob) -> JobStatusResponse:
    return JobStatusResponse(
        id=job.id,
        paper_id=job.paper_id,
        stage=job.stage,
        status=job.status,
        progress=job.progress,
        message=job.message,
        attempts=job.attempts,
        error_code=job.error_code,
        error_message=job.error_message,
        updated_at=job.updated_at.isoformat() if job.updated_at else "",
    )


async def _save_upload(upload: UploadFile) -> Path:
    staging = settings.papers_root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / f"{uuid4()}.pdf"
    size = 0
    try:
        with path.open("wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_pdf_bytes:
                    raise HTTPException(status_code=413, detail="PDF exceeds the 100 MB limit")
                destination.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


async def _download_source(url: str) -> Path:
    if not url.startswith(("https://", "http://")):
        raise HTTPException(status_code=400, detail="sourceUrl must be http or https")
    staging = settings.papers_root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / f"{uuid4()}.pdf"
    size = 0
    try:
        async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client, client.stream("GET", url) as response:
            response.raise_for_status()
            with path.open("wb") as destination:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_pdf_bytes:
                        raise HTTPException(status_code=413, detail="PDF exceeds the 100 MB limit")
                    destination.write(chunk)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except httpx.HTTPError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=f"Could not download source PDF: {exc}") from exc
    return path


@app.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/health/ready")
def health_ready() -> dict[str, object]:
    ready = database_ready()
    try:
        usage = shutil.disk_usage(settings.papers_root)
        papers_bytes = directory_size(settings.papers_root)
        free_bytes = usage.free
    except OSError:
        papers_bytes = 0
        free_bytes = 0
    return {
        "status": "ready" if ready else "not_ready",
        "database": ready,
        "embeddingWarm": rag_service.embedding_warmed,
        "papersRoot": str(settings.papers_root),
        "papersBytes": papers_bytes,
        "freeBytes": free_bytes,
    }


@app.get("/health/metrics")
def health_metrics(db: Session = Depends(get_db)) -> dict[str, object]:
    snapshot = metrics.snapshot()
    try:
        free_bytes = shutil.disk_usage(settings.papers_root).free
    except OSError:
        free_bytes = 0
    snapshot["assets"] = {
        "root": str(settings.papers_root),
        "bytes": directory_size(settings.papers_root),
        "freeBytes": free_bytes,
    }
    snapshot["ingestion"] = {
        "papers": db.scalar(select(func.count(Paper.id))) or 0,
        "readyPapers": db.scalar(select(func.count(Paper.id)).where(Paper.status == "ready")) or 0,
        "queuedJobs": db.scalar(select(func.count(IngestionJob.id)).where(IngestionJob.status == "queued")) or 0,
        "runningJobs": db.scalar(select(func.count(IngestionJob.id)).where(IngestionJob.status == "running")) or 0,
        "failedJobs": db.scalar(select(func.count(IngestionJob.id)).where(IngestionJob.status == "failed")) or 0,
    }
    return snapshot


@app.post("/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    started = time.perf_counter()
    try:
        intent, candidates, reranked = await asyncio.wait_for(
            discovery.search(request.query, request.limit),
            timeout=settings.discovery_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Research discovery timed out") from exc
    except RetryableProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        metrics.record_search((time.perf_counter() - started) * 1000)
    return SearchResponse(intent=intent, candidates=candidates, reranked=reranked)


@app.post(
    "/v1/papers/ingest",
    response_model=IngestResponse,
    status_code=202,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": IngestSource.model_json_schema(by_alias=True)},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "sourceUrl": {"type": "string", "format": "uri"},
                            "metadata": {"type": "string", "description": "JSON object"},
                            "analysisModel": {"type": "string", "enum": list(ANALYSIS_MODEL_FALLBACKS)},
                        },
                    }
                },
            },
        }
    },
)
async def ingest_paper(
    request: Request,
    db: Session = Depends(get_db),
) -> IngestResponse:
    file: UploadFile | None = None
    source_url: str | None = None
    metadata: dict[str, object] = {}
    analysis_model = analysis_model_chain(settings.paper_analysis_model)[0]
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type == "application/json":
        try:
            payload = IngestSource.model_validate(await request.json())
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail="JSON body must contain a valid sourceUrl") from exc
        source_url = payload.source_url
        analysis_model = payload.analysis_model
        metadata = {**payload.metadata, "analysisModel": analysis_model}
    elif content_type == "multipart/form-data":
        try:
            form = await request.form()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Multipart form data could not be parsed") from exc
        candidate_file = form.get("file")
        if isinstance(candidate_file, UploadFile):
            file = candidate_file
        candidate_source_url = form.get("sourceUrl") or form.get("source_url")
        if candidate_source_url is not None:
            if not isinstance(candidate_source_url, str):
                raise HTTPException(status_code=400, detail="sourceUrl must be a string")
            source_url = candidate_source_url
        raw_metadata = form.get("metadata")
        if raw_metadata:
            if not isinstance(raw_metadata, str):
                raise HTTPException(status_code=400, detail="metadata must be a JSON object")
            try:
                parsed_metadata = json.loads(raw_metadata)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc
            if not isinstance(parsed_metadata, dict):
                raise HTTPException(status_code=400, detail="metadata must be a JSON object")
            metadata = parsed_metadata
        candidate_analysis_model = form.get("analysisModel") or form.get("analysis_model")
        if candidate_analysis_model is not None:
            if not isinstance(candidate_analysis_model, str) or candidate_analysis_model not in ANALYSIS_MODEL_FALLBACKS:
                raise HTTPException(status_code=400, detail="analysisModel must be one of the supported Gemini Flash models")
            analysis_model = candidate_analysis_model
        metadata = {**metadata, "analysisModel": analysis_model}
    elif content_type:
        raise HTTPException(status_code=415, detail="Use application/json or multipart/form-data")
    if file and source_url:
        raise HTTPException(status_code=400, detail="Provide either a PDF file or sourceUrl, not both")
    if not file and not source_url:
        raise HTTPException(status_code=400, detail="Provide a PDF file or sourceUrl")
    staged = await _save_upload(file) if file else await _download_source(source_url or "")
    try:
        page_count = validate_pdf(staged, settings.max_pdf_bytes, settings.max_pdf_pages)
        digest = sha256_file(staged)
        existing = db.scalar(select(Paper).where(Paper.sha256 == digest))
        if existing:
            staged.unlink(missing_ok=True)
            job = db.scalar(select(IngestionJob).where(IngestionJob.paper_id == existing.id).order_by(desc(IngestionJob.created_at)))
            if existing.status in {"queued", "processing"}:
                raise HTTPException(status_code=409, detail="This paper is already being ingested")
            if existing.status == "failed" and job:
                existing.metadata_json = {**(existing.metadata_json or {}), **metadata, "analysisModel": analysis_model}
                job.status = "queued"
                job.attempts = 0
                job.error_code = None
                job.error_message = None
                job.message = "Queued for ingestion retry"
                job.locked_until = None
                existing.status = "queued"
                db.commit()
                return IngestResponse(paper_id=existing.id, job_id=job.id, status="queued")
            if not job:
                job = IngestionJob(
                    paper_id=existing.id,
                    stage="ready" if existing.status == "ready" else "failed",
                    status="succeeded" if existing.status == "ready" else "failed",
                    progress=100 if existing.status == "ready" else 0,
                    message="Already ingested" if existing.status == "ready" else "Ingestion previously failed",
                )
                db.add(job)
                db.commit()
            return IngestResponse(paper_id=existing.id, job_id=job.id, status=existing.status)  # type: ignore[arg-type]

        paper_dir = settings.papers_root / digest
        paper_dir.mkdir(parents=True, exist_ok=True)
        original = paper_dir / "original.pdf"
        shutil.move(str(staged), str(original))
        metadata_title = metadata.get("title")
        metadata_authors = metadata.get("authors")
        metadata_doi = metadata.get("doi")
        metadata_arxiv_id = metadata.get("arxivId")
        paper = Paper(
            sha256=digest,
            title=file.filename if file else metadata_title if isinstance(metadata_title, str) else None,
            authors=metadata_authors if isinstance(metadata_authors, list) and all(isinstance(author, str) for author in metadata_authors) else [],
            doi=metadata_doi if isinstance(metadata_doi, str) else None,
            arxiv_id=metadata_arxiv_id if isinstance(metadata_arxiv_id, str) else None,
            source_url=source_url,
            original_path=str(original.resolve().relative_to(settings.papers_root.resolve())),
            page_count=page_count,
            status="queued",
            metadata_json={**metadata, "analysisModel": analysis_model},
        )
        db.add(paper)
        db.flush()
        job = IngestionJob(paper_id=paper.id)
        db.add(job)
        db.commit()
        return IngestResponse(paper_id=paper.id, job_id=job.id, status="queued")
    except PDFLimitError as exc:
        staged.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except PDFValidationError as exc:
        staged.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        staged.unlink(missing_ok=True)
        raise
    except Exception:
        staged.unlink(missing_ok=True)
        raise


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: UUID, db: Session = Depends(get_db)) -> JobStatusResponse:
    job = db.get(IngestionJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_response(job)


@app.post("/v1/jobs/{job_id}/retry", response_model=IngestResponse, status_code=202)
def retry_job(job_id: UUID, db: Session = Depends(get_db)) -> IngestResponse:
    previous = db.get(IngestionJob, job_id)
    if not previous:
        raise HTTPException(status_code=404, detail="Job not found")
    paper = db.get(Paper, previous.paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    if paper.status != "failed" or previous.status != "failed":
        raise HTTPException(status_code=409, detail="Only a failed ingestion job can be retried")
    previous.status = "queued"
    previous.attempts = 0
    previous.error_code = None
    previous.error_message = None
    previous.message = "Queued for ingestion retry"
    previous.locked_until = None
    previous.paper_id = paper.id
    paper.status = "queued"
    job = previous
    db.commit()
    return IngestResponse(paper_id=paper.id, job_id=job.id, status="queued")


def _paper_response(db: Session, paper: Paper) -> PaperResponse:
    analysis_row = db.scalar(select(PaperAnalysis).where(PaperAnalysis.paper_id == paper.id).order_by(desc(PaperAnalysis.created_at)))
    analysis = None
    if analysis_row:
        levels = analysis_row.payload.get("levels", {})
        payload = levels.get("student") or next(iter(levels.values()), None)
        if payload:
            analysis = PaperAnalysisSchema.model_validate(payload)

    visuals = list(db.scalars(select(PaperVisual).where(PaperVisual.paper_id == paper.id).order_by(PaperVisual.page_number)).all())
    visual_response = [
        VisualResponse(
            id=item.id,
            paper_id=item.paper_id,
            type=item.visual_type,  # type: ignore[arg-type]
            label=item.figure_label,
            caption=item.caption,
            page_number=item.page_number,
            page_image_url=f"/v1/papers/{paper.id}/pages/{item.page_number}",
            cropped_image_url=f"/v1/papers/{paper.id}/visuals/{item.id}" if item.cropped_image_path else None,
            visual_description=item.visual_description,
            visual_interpretation=item.visual_interpretation or "",
            concepts=(item.metadata_json or {}).get("concepts", []),
        )
        for item in visuals
    ]

    graph_response = None
    graph = db.scalar(select(PaperGraph).where(PaperGraph.paper_id == paper.id).order_by(desc(PaperGraph.id)))
    if graph:
        nodes = list(db.scalars(select(PaperGraphNode).where(PaperGraphNode.graph_id == graph.id)).all())
        edges = list(db.scalars(select(PaperGraphEdge).where(PaperGraphEdge.graph_id == graph.id)).all())
        graph_response = GraphResponse(
            nodes=[GraphNodeResponse(id=node.node_key, label=node.label, type=node.node_type, explanation=node.explanation, evidence_ids=node.evidence_ids or []) for node in nodes],
            edges=[GraphEdge(id=str(edge.id), source=edge.source_node_key, target=edge.target_node_key, label=edge.label, relation=edge.relation) for edge in edges],
        )

    latest_job = db.scalar(select(IngestionJob).where(IngestionJob.paper_id == paper.id).order_by(desc(IngestionJob.created_at)))
    paper_metadata = paper.metadata_json or {}

    return PaperResponse(
        id=paper.id,
        title=paper.title or "Untitled paper",
        authors=paper.authors or [],
        year=(paper.metadata_json or {}).get("year"),
        venue=(paper.metadata_json or {}).get("venue"),
        status=paper.status,
        page_count=paper.page_count or 0,
        source_url=paper.source_url,
        analysis_model=analysis_row.model if analysis_row else paper_metadata.get("analysisModelUsed") or paper_metadata.get("analysisModel"),
        analysis_fallback_used=bool(paper_metadata.get("analysisFallbackUsed", False)),
        job=_job_response(latest_job) if latest_job else None,
        analysis=analysis,
        graph=graph_response,
        visuals=visual_response,
    )


@app.get("/v1/papers/{paper_id}", response_model=PaperResponse)
def get_paper(paper_id: UUID, db: Session = Depends(get_db)) -> PaperResponse:
    paper = db.get(Paper, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return _paper_response(db, paper)


@app.get("/v1/papers/{paper_id}/analysis", response_model=PaperAnalysisSchema)
def get_analysis(
    paper_id: UUID,
    level: ExplanationLevel = Query(default="student"),
    db: Session = Depends(get_db),
) -> PaperAnalysisSchema:
    paper = db.get(Paper, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    analysis_row = db.scalar(select(PaperAnalysis).where(PaperAnalysis.paper_id == paper.id).order_by(desc(PaperAnalysis.created_at)))
    if not analysis_row:
        raise HTTPException(status_code=409, detail="Paper analysis is not ready")
    payload = (analysis_row.payload.get("levels") or {}).get(level)
    if not payload:
        raise HTTPException(status_code=409, detail=f"The {level} explanation is not available")
    return PaperAnalysisSchema.model_validate(payload)


@app.get("/v1/papers/{paper_id}/graph", response_model=GraphResponse)
def get_graph(paper_id: UUID, db: Session = Depends(get_db)) -> GraphResponse:
    paper = db.get(Paper, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    response = _paper_response(db, paper).graph
    if not response:
        raise HTTPException(status_code=409, detail="Paper graph is not ready")
    return response


@app.get("/v1/assets/{asset_id}")
def get_asset(asset_id: UUID, db: Session = Depends(get_db)) -> FileResponse:
    page = db.get(PaperPage, asset_id)
    if page:
        return FileResponse(_asset_path(page.image_path), media_type="image/webp")
    visual = db.get(PaperVisual, asset_id)
    if visual:
        return FileResponse(_asset_path(visual.cropped_image_path or visual.page_image_path), media_type="image/webp")
    paper = db.get(Paper, asset_id)
    if paper:
        return FileResponse(_asset_path(paper.original_path), media_type="application/pdf")
    raise HTTPException(status_code=404, detail="Asset not found")


@app.get("/v1/papers/{paper_id}/pages/{page_number}")
def get_page(paper_id: UUID, page_number: int, db: Session = Depends(get_db)) -> FileResponse:
    page = db.scalar(select(PaperPage).where(PaperPage.paper_id == paper_id, PaperPage.page_number == page_number))
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(_asset_path(page.image_path), media_type="image/webp")


@app.get("/v1/papers/{paper_id}/visuals/{visual_id}")
def get_visual(paper_id: UUID, visual_id: UUID, db: Session = Depends(get_db)) -> FileResponse:
    visual = db.scalar(select(PaperVisual).where(PaperVisual.id == visual_id, PaperVisual.paper_id == paper_id))
    if not visual:
        raise HTTPException(status_code=404, detail="Visual not found")
    return FileResponse(_asset_path(visual.cropped_image_path or visual.page_image_path), media_type="image/webp")


@app.post("/v1/rag/query", response_model=RAGAnswer)
async def rag_query(request: RAGRequest, db: Session = Depends(get_db)) -> RAGAnswer:
    if not rag_rate_limiter.allow():
        metrics.record_rag_rate_limit()
        raise HTTPException(status_code=429, detail="RAG rate limit exceeded; try again shortly")
    papers = list(db.scalars(select(Paper).where(Paper.id.in_(request.paper_ids))).all())
    if len(papers) != len(set(request.paper_ids)):
        raise HTTPException(status_code=404, detail="One or more papers were not found")
    if any(paper.status != "ready" for paper in papers):
        raise HTTPException(status_code=409, detail="All selected papers must finish ingestion first")
    try:
        result = await asyncio.wait_for(rag_service.answer(db, request), timeout=settings.rag_timeout_seconds)
        metrics.record_rag(
            latency_ms=result.latency_ms,
            fallback_used=result.fallback_used,
            grounded=result.grounded,
        )
        return result
    except TimeoutError as exc:
        metrics.record_rag_error()
        raise HTTPException(status_code=504, detail=f"RAG request exceeded the {settings.rag_timeout_seconds}-second timeout") from exc
    except ProviderConfigurationError as exc:
        metrics.record_rag_error()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RetryableProviderError as exc:
        metrics.record_rag_error()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        metrics.record_rag_error()
        logger.exception("RAG request failed")
        raise HTTPException(status_code=500, detail="RAG request failed") from exc
