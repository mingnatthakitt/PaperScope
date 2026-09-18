from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from paperscope.config import ANALYSIS_MODEL_FALLBACKS, analysis_model_chain, settings
from paperscope.models.tables import (
    IngestionJob,
    Paper,
    PaperAnalysis,
    PaperChunk,
    PaperGraph,
    PaperGraphEdge,
    PaperGraphNode,
    PaperPage,
    PaperSection,
    PaperVisual,
)
from paperscope.processing.pdf import crop_normalized, detect_sections, extract_pages, render_pages, sanitize_pdf_text
from paperscope.providers.embeddings import QwenEmbeddingProvider
from paperscope.providers.gemini import GeminiDocumentAnalysisProvider
from paperscope.schemas import DeepPaperAnalysis
from paperscope.schemas import GraphEdge as AnalysisGraphEdge
from paperscope.schemas import GraphNode as AnalysisGraphNode
from paperscope.schemas import PaperGraph as AnalysisGraph

STAGES = ["extracting", "rendering", "analyzing", "embedding"]
logger = logging.getLogger("paperscope.ingest")


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _visual_filename(key: str, index: int) -> str:
    """Return a deterministic, filesystem-safe name owned by the worker."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", key).strip("-_")[:80] or "visual"
    return f"{index:03d}-{slug}.webp"


def _requested_analysis_model(paper: Paper) -> str:
    candidate = (paper.metadata_json or {}).get("analysisModel")
    if isinstance(candidate, str) and candidate in ANALYSIS_MODEL_FALLBACKS:
        return candidate
    return analysis_model_chain(settings.paper_analysis_model)[0]


def reconcile_graph(
    graph: AnalysisGraph,
    evidence_by_key: dict[str, str],
) -> tuple[list[AnalysisGraphNode], list[AnalysisGraphEdge]]:
    """Keep only graph references that can be resolved to persisted evidence.

    Model-generated graph coordinates are intentionally absent from this function. The
    browser owns layout, while the worker owns referential integrity for nodes, edges,
    and evidence IDs.
    """
    nodes: list[AnalysisGraphNode] = []
    node_keys: set[str] = set()
    for node in graph.nodes:
        if not node.id or node.id in node_keys:
            logger.warning("Discarding duplicate or empty graph node: %r", node.id)
            continue
        node_keys.add(node.id)
        evidence_ids: list[str] = []
        for key in node.evidence_keys:
            evidence_id = evidence_by_key.get(key)
            if evidence_id:
                evidence_ids.append(evidence_id)
            else:
                logger.warning("Discarding unresolved graph evidence key %r for node %r", key, node.id)
        nodes.append(node.model_copy(update={"evidence_keys": list(dict.fromkeys(evidence_ids))}))

    if not nodes:
        raise ValueError("Gemini returned a graph with no valid nodes")

    edges: list[AnalysisGraphEdge] = []
    for edge in graph.edges:
        if edge.source not in node_keys or edge.target not in node_keys:
            logger.warning("Discarding graph edge with missing endpoint: %r -> %r", edge.source, edge.target)
            continue
        edges.append(edge)
    return nodes, edges


class IngestionPipeline:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.embedding_provider = QwenEmbeddingProvider()

    def _update_job(self, job: IngestionJob, stage: str, progress: int, message: str) -> None:
        job.stage = stage
        job.progress = progress
        job.message = message
        job.locked_until = datetime.now(UTC) + timedelta(seconds=settings.ingestion_lease_seconds)
        self.session.commit()

    def _clear_derived(self, paper_id) -> None:
        graph_ids = list(self.session.scalars(select(PaperGraph.id).where(PaperGraph.paper_id == paper_id)).all())
        if graph_ids:
            self.session.execute(delete(PaperGraphEdge).where(PaperGraphEdge.graph_id.in_(graph_ids)))
            self.session.execute(delete(PaperGraphNode).where(PaperGraphNode.graph_id.in_(graph_ids)))
        self.session.execute(delete(PaperGraph).where(PaperGraph.paper_id == paper_id))
        self.session.execute(delete(PaperVisual).where(PaperVisual.paper_id == paper_id))
        self.session.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper_id))
        self.session.execute(delete(PaperSection).where(PaperSection.paper_id == paper_id))
        self.session.execute(delete(PaperPage).where(PaperPage.paper_id == paper_id))
        self.session.commit()
        paper = self.session.get(Paper, paper_id)
        if paper:
            paper_dir = settings.papers_root / paper.sha256
            for directory in (paper_dir / "pages", paper_dir / "visuals"):
                if directory.exists():
                    # Rendered assets are disposable derived state. On macOS volumes
                    # that preserve resource forks, AppleDouble sidecars can appear or
                    # disappear while a directory is being removed; ignore that race
                    # and let the next stage recreate the directory.
                    shutil.rmtree(directory, ignore_errors=True)
            (paper_dir / "manifest.json").unlink(missing_ok=True)

    def _clear_analysis_outputs(self, paper: Paper) -> None:
        graph_ids = list(self.session.scalars(select(PaperGraph.id).where(PaperGraph.paper_id == paper.id)).all())
        if graph_ids:
            self.session.execute(delete(PaperGraphEdge).where(PaperGraphEdge.graph_id.in_(graph_ids)))
            self.session.execute(delete(PaperGraphNode).where(PaperGraphNode.graph_id.in_(graph_ids)))
        self.session.execute(delete(PaperGraph).where(PaperGraph.paper_id == paper.id))
        self.session.execute(delete(PaperVisual).where(PaperVisual.paper_id == paper.id))
        self.session.commit()
        visuals_dir = settings.papers_root / paper.sha256 / "visuals"
        if visuals_dir.exists():
            shutil.rmtree(visuals_dir, ignore_errors=True)

    def _persist_pages(self, paper: Paper, page_texts, rendered) -> None:
        self.session.execute(delete(PaperPage).where(PaperPage.paper_id == paper.id))
        for page_text, page_image in zip(page_texts, rendered):
            self.session.add(
                PaperPage(
                    paper_id=paper.id,
                    page_number=page_text.page_number,
                    image_path=_relative(settings.papers_root, page_image.path),
                    width=page_image.width,
                    height=page_image.height,
                    text_content=sanitize_pdf_text(page_text.text),
                )
            )
        self.session.flush()

    def _write_manifest(self, paper: Paper, visuals: list[PaperVisual] | None = None) -> None:
        pages = list(self.session.scalars(select(PaperPage).where(PaperPage.paper_id == paper.id).order_by(PaperPage.page_number)).all())
        metadata = paper.metadata_json or {}
        payload = {
            "sha256": paper.sha256,
            "originalPath": paper.original_path,
            "pageCount": paper.page_count,
            "renderDpi": settings.render_dpi,
            "updatedAt": datetime.now(UTC).isoformat(),
            "pages": [
                {"id": str(page.id), "pageNumber": page.page_number, "imagePath": page.image_path, "width": page.width, "height": page.height}
                for page in pages
            ],
            "visuals": [
                {
                    "id": str(visual.id),
                    "pageNumber": visual.page_number,
                    "type": visual.visual_type,
                    "label": visual.figure_label,
                    "pageImagePath": visual.page_image_path,
                    "croppedImagePath": visual.cropped_image_path,
                }
                for visual in visuals or []
            ],
            "analysis": {
                "requestedModel": metadata.get("analysisModel", _requested_analysis_model(paper)),
                "model": metadata.get("analysisModelUsed", _requested_analysis_model(paper)),
                "fallbackUsed": bool(metadata.get("analysisFallbackUsed", False)),
                "promptVersion": settings.analysis_prompt_version,
            },
        }
        manifest = settings.papers_root / paper.sha256 / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(manifest)

    def _persist_text(self, paper: Paper, pages) -> tuple[list[PaperSection], list[PaperChunk]]:
        section_ranges = detect_sections(pages)
        sections: list[PaperSection] = []
        for index, (title, start, end) in enumerate(section_ranges):
            section = PaperSection(
                paper_id=paper.id,
                title=title,
                section_index=index,
                page_start=start,
                page_end=end,
            )
            sections.append(section)
            self.session.add(section)
        self.session.flush()

        from paperscope.processing.pdf import chunk_pages

        chunks = []
        for chunk in chunk_pages(pages, section_ranges):
            section_id = next(
                (section.id for section in sections if section.title == chunk.section_title),
                None,
            )
            row = PaperChunk(
                paper_id=paper.id,
                section_id=section_id,
                chunk_index=chunk.chunk_index,
                content=sanitize_pdf_text(chunk.content),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_title=sanitize_pdf_text(chunk.section_title) if chunk.section_title else chunk.section_title,
                chunk_type=chunk.chunk_type,
            )
            chunks.append(row)
            self.session.add(row)
        self.session.flush()
        return sections, chunks

    def _persist_visuals(self, paper: Paper, analysis: DeepPaperAnalysis, page_paths: dict[int, Path]) -> list[PaperVisual]:
        visuals: list[PaperVisual] = []
        visuals_dir = settings.papers_root / paper.sha256 / "visuals"
        for index, visual in enumerate(analysis.important_visuals, start=1):
            page_path = page_paths.get(visual.page)
            if not page_path:
                continue
            crop_path: Path | None = None
            if visual.bbox:
                candidate = visuals_dir / _visual_filename(visual.key, index)
                try:
                    crop_path = crop_normalized(page_path, visual.bbox, candidate)
                except (OSError, ValueError):
                    crop_path = None
            row = PaperVisual(
                paper_id=paper.id,
                page_number=visual.page,
                visual_type=visual.type,
                figure_label=visual.label,
                caption=visual.caption or visual.reason,
                visual_description=visual.visual_description,
                visual_interpretation=visual.visual_interpretation,
                page_image_path=_relative(settings.papers_root, page_path),
                cropped_image_path=_relative(settings.papers_root, crop_path) if crop_path else None,
                metadata_json={"key": visual.key, "importance": visual.importance, "concepts": visual.concepts},
            )
            visuals.append(row)
            self.session.add(row)
        self.session.flush()
        return visuals

    def _persist_graph(self, paper: Paper, analysis_row: PaperAnalysis, analysis: DeepPaperAnalysis, visuals: list[PaperVisual]) -> None:
        graph = PaperGraph(paper_id=paper.id, analysis_id=analysis_row.id, diagram_type=analysis.graph.diagram_type)
        self.session.add(graph)
        self.session.flush()
        visual_by_key = {str(visual.metadata_json.get("key")): str(visual.id) for visual in visuals}
        pages = self.session.scalars(select(PaperPage).where(PaperPage.paper_id == paper.id)).all()
        evidence_by_key = {
            **visual_by_key,
            **{f"page-{page.page_number}": str(page.id) for page in pages},
        }
        graph_nodes, graph_edges = reconcile_graph(analysis.graph, evidence_by_key)
        for node in graph_nodes:
            self.session.add(
                PaperGraphNode(
                    graph_id=graph.id,
                    node_key=node.id,
                    label=node.label,
                    node_type=node.type,
                    explanation=node.explanation,
                    evidence_ids=node.evidence_keys,
                )
            )
        for edge in graph_edges:
            self.session.add(
                PaperGraphEdge(
                    graph_id=graph.id,
                    source_node_key=edge.source,
                    target_node_key=edge.target,
                    label=edge.label,
                    relation=edge.relation,
                )
            )
        self.session.commit()

    def _embed(self, chunks: list[PaperChunk], visuals: list[PaperVisual], graph_nodes: list[PaperGraphNode]) -> None:
        texts = [
            f"Section: {chunk.section_title or 'Unknown'}\n{chunk.content}"
            for chunk in chunks
        ] + [
            f"{visual.figure_label or visual.visual_type}. {visual.visual_description}\n{visual.visual_interpretation or ''}"
            for visual in visuals
        ] + [
            f"{node.label}: {node.explanation}" for node in graph_nodes
        ]
        vectors = self.embedding_provider.embed_documents(texts)
        offset = 0
        for row in chunks:
            row.embedding = vectors[offset]
            offset += 1
        for row in visuals:
            row.embedding = vectors[offset]
            offset += 1
        for row in graph_nodes:
            row.embedding = vectors[offset]
            offset += 1
        self.session.commit()

    def process(self, job_id) -> None:
        job = self.session.get(IngestionJob, job_id)
        if not job:
            raise ValueError("Ingestion job not found")
        paper = self.session.get(Paper, job.paper_id)
        if not paper:
            raise ValueError("Paper not found")
        paper.status = "processing"
        job.status = "running"
        self.session.commit()
        original = settings.papers_root / paper.original_path
        if not original.exists():
            raise FileNotFoundError(f"Original PDF missing: {paper.original_path}")

        stage = job.stage if job.stage in {"rendering", "analyzing", "embedding"} else "queued"
        if stage == "queued":
            self._clear_derived(paper.id)
            self._update_job(job, "extracting", 10, "Extracting text and sections")
            pages = extract_pages(original)
            _, chunks = self._persist_text(paper, pages)
            self.session.commit()
            stage = "rendering"

        if stage == "rendering":
            self._update_job(job, "rendering", 25, "Rendering every paper page")
            pages = extract_pages(original)
            rendered = render_pages(original, settings.papers_root / paper.sha256 / "pages", settings.render_dpi)
            self._persist_pages(paper, pages, rendered)
            paper.page_count = len(rendered)
            self._write_manifest(paper)
            self.session.commit()
            stage = "analyzing"

        if stage == "analyzing":
            self._update_job(job, "analyzing", 50, "Inspecting the native PDF with Gemini")
            page_rows = list(self.session.scalars(select(PaperPage).where(PaperPage.paper_id == paper.id)).all())
            chunks = list(self.session.scalars(select(PaperChunk).where(PaperChunk.paper_id == paper.id).order_by(PaperChunk.chunk_index)).all())
            if not page_rows:
                raise RuntimeError("Cannot resume analysis without rendered pages")
            self._clear_analysis_outputs(paper)
            page_paths = {page.page_number: settings.papers_root / page.image_path for page in page_rows}
            requested_model = _requested_analysis_model(paper)
            model_chain = analysis_model_chain(requested_model)
            cached_rows = list(
                self.session.scalars(
                    select(PaperAnalysis).where(
                        PaperAnalysis.paper_id == paper.id,
                        PaperAnalysis.model.in_(model_chain),
                        PaperAnalysis.prompt_version == settings.analysis_prompt_version,
                    )
                ).all()
            )
            analysis_row = next(
                (row for model in model_chain for row in cached_rows if row.model == model),
                None,
            )
            if analysis_row:
                logger.info("Reusing cached Gemini analysis for paper %s", paper.id)
                analysis = DeepPaperAnalysis.model_validate(analysis_row.payload)
            else:
                analysis_provider = GeminiDocumentAnalysisProvider(model=requested_model)
                analysis = asyncio.run(analysis_provider.analyze_pdf(original))
                actual_model = analysis_provider.model_used
                analysis_row = PaperAnalysis(
                    paper_id=paper.id,
                    model=actual_model,
                    prompt_version=settings.analysis_prompt_version,
                    payload=analysis.model_dump(mode="json", by_alias=False),
                )
                self.session.add(analysis_row)
                self.session.flush()
            actual_model = analysis_row.model
            paper.metadata_json = {
                **(paper.metadata_json or {}),
                "analysisModel": requested_model,
                "analysisModelUsed": actual_model,
                "analysisFallbackUsed": actual_model != requested_model,
                "analysisModelFallbackChain": list(model_chain),
            }
            if analysis.title:
                paper.title = analysis.title
            visuals = self._persist_visuals(paper, analysis, page_paths)
            self._persist_graph(paper, analysis_row, analysis, visuals)
            self._write_manifest(paper, visuals)
            stage = "embedding"

        if stage == "embedding":
            self._update_job(job, "embedding", 76, "Embedding text, visuals, and graph nodes locally")
            chunks = list(self.session.scalars(select(PaperChunk).where(PaperChunk.paper_id == paper.id).order_by(PaperChunk.chunk_index)).all())
            visuals = list(self.session.scalars(select(PaperVisual).where(PaperVisual.paper_id == paper.id).order_by(PaperVisual.page_number)).all())
            graph_id = self.session.scalar(select(PaperGraph.id).where(PaperGraph.paper_id == paper.id))
            graph_nodes = list(self.session.scalars(select(PaperGraphNode).where(PaperGraphNode.graph_id == graph_id)).all()) if graph_id else []
            self._embed(chunks, visuals, graph_nodes)

        paper.status = "ready"
        job.status = "succeeded"
        job.stage = "ready"
        job.progress = 100
        job.message = "Paper analysis and retrieval index are ready"
        self.session.commit()
