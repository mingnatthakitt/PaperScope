from __future__ import annotations

import re
from pathlib import Path
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from paperscope.config import settings
from paperscope.models.tables import PaperPage, PaperVisual
from paperscope.providers.nim import image_data_url
from paperscope.schemas import ConversationTurn, ExplanationLevel, ImagePayload, RAGModelInput, RetrievedContext

VISUAL_TRIGGER = re.compile(
    r"\b(figure|fig\.?|diagram|plot|chart|table|architecture|graph|shown|trend|curve|line|bar|axis|legend|color|visual|image|ablation|heatmap|histogram|scatter|distribution|flatten|peaks?)\b",
    re.IGNORECASE,
)


def should_attach_images(
    question: str,
    evidence: list[RetrievedContext],
    history: list[ConversationTurn] | None = None,
) -> bool:
    # Retrieval always searches text, visuals, and graph nodes. That does not by
    # itself mean the generation request needs expensive image payloads: a normal
    # text question should stay text-only even when a visual description is among
    # the supporting evidence. Explicit visual language is the routing contract.
    if VISUAL_TRIGGER.search(question):
        return True
    if not history or not any(VISUAL_TRIGGER.search(turn.content) for turn in history[-4:]):
        return False
    return any(item.image_refs or item.source_type in {"visual", "table", "graph_node"} for item in evidence)


def _safe_asset_path(relative_path: str) -> Path:
    root = settings.papers_root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Asset path escapes PAPERS_ROOT")
    return candidate


def resolve_images(
    session: Session,
    question: str,
    evidence: list[RetrievedContext],
    history: list[ConversationTurn] | None = None,
) -> list[ImagePayload]:
    if not should_attach_images(question, evidence, history):
        return []

    page_keys: set[tuple[str, int]] = set()
    visual_ids: list[str] = []
    for item in evidence:
        if item.page_start and item.page_start == item.page_end:
            page_keys.add((str(item.paper_id), item.page_start))
        visual_ids.extend(item.image_refs)
        if item.source_type == "graph_node":
            visual_ids.extend(str(evidence_id) for evidence_id in item.metadata.get("evidenceIds", []))

    images: list[ImagePayload] = []
    seen_asset_paths: set[str] = set()
    visuals = []
    for visual_id in dict.fromkeys(visual_ids):
        try:
            asset_id = UUID(str(visual_id))
        except (TypeError, ValueError):
            continue
        try:
            visual = session.get(PaperVisual, asset_id)
        except SQLAlchemyError:
            visual = None
        if visual:
            visuals.append(visual)
            page_keys.add((str(visual.paper_id), visual.page_number))
            continue
        try:
            page = session.get(PaperPage, asset_id)
        except SQLAlchemyError:
            page = None
        if page:
            page_keys.add((str(page.paper_id), page.page_number))

    page_conditions = [
        (PaperPage.paper_id == paper_id) & (PaperPage.page_number == page_number)
        for paper_id, page_number in page_keys
    ]
    pages = session.scalars(select(PaperPage).where(or_(*page_conditions))).all() if page_conditions else []
    pages_by_key = {(str(page.paper_id), page.page_number): page for page in pages}

    # Crops are the most precise visual evidence, so add them before their full-page
    # fallbacks. Text-only visual questions still use the relevant full pages below.
    for visual in visuals:
        crop = visual.cropped_image_path or visual.page_image_path
        path = _safe_asset_path(crop)
        if not path.exists():
            continue
        asset_key = str(path.resolve())
        if asset_key in seen_asset_paths:
            continue
        seen_asset_paths.add(asset_key)
        image_type = "table" if visual.visual_type == "table" else "figure"
        images.append(
            ImagePayload(
                id=f"visual:{visual.id}",
                type=image_type,
                page_number=visual.page_number,
                base64=image_data_url(path.read_bytes()),
                label=visual.figure_label or visual.visual_type,
            )
        )

    for page in pages_by_key.values():
        path = _safe_asset_path(page.image_path)
        if not path.exists():
            continue
        asset_key = str(path.resolve())
        if asset_key in seen_asset_paths:
            continue
        seen_asset_paths.add(asset_key)
        images.append(
            ImagePayload(
                id=f"page:{page.paper_id}:{page.page_number}",
                type="page",
                page_number=page.page_number,
                base64=image_data_url(path.read_bytes()),
                label=f"Page {page.page_number}",
            )
        )

    # Keep the request bounded while preserving one full page per page and the most
    # specific visual assets first.
    deduped: list[ImagePayload] = []
    seen: set[str] = set()
    for image in images:
        if image.id in seen:
            continue
        seen.add(image.id)
        deduped.append(image)
        if len(deduped) == 6:
            break
    return deduped


def build_multimodal_context(
    session: Session,
    question: str,
    evidence: list[RetrievedContext],
    explanation_level: ExplanationLevel = "student",
    history: list[ConversationTurn] | None = None,
) -> RAGModelInput:
    return RAGModelInput(
        question=question,
        explanation_level=explanation_level,
        history=history or [],
        evidence=evidence,
        images=resolve_images(session, question, evidence, history),
    )
