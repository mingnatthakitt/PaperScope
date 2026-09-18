from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)


ExplanationLevel = Literal["simple", "student", "researcher"]
AnalysisModel = Literal["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]
VisualType = Literal["architecture", "diagram", "plot", "chart", "table", "algorithm", "equation", "other"]
GraphNodeType = Literal["input", "component", "process", "decision", "concept", "result", "baseline", "proposed"]


class GlossaryEntry(CamelModel):
    term: str
    definition: str


class PaperAnalysis(CamelModel):
    summary: str
    problem: str
    motivation: str
    key_insight: str
    method: str
    contributions: list[str] = Field(default_factory=list)
    results: str
    limitations: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)


class AnalysisLevels(CamelModel):
    """The fixed explanation levels requested from Gemini.

    Explicit fields keep the structured-output wire schema compatible while preserving
    the same JSON shape used by the API and database payloads.
    """

    simple: PaperAnalysis
    student: PaperAnalysis
    researcher: PaperAnalysis


class ImportantVisual(CamelModel):
    key: str
    label: str | None = None
    caption: str | None = None
    page: int = Field(ge=1)
    type: VisualType
    importance: str = "supporting"
    reason: str
    visual_description: str
    visual_interpretation: str
    concepts: list[str] = Field(default_factory=list)
    bbox: list[float] | None = None

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 4:
            raise ValueError("bbox must contain x, y, width, height")
        if value is not None and any(number < 0 or number > 1 for number in value):
            raise ValueError("bbox values must be normalized between 0 and 1")
        return value


class GraphNode(CamelModel):
    id: str
    label: str
    type: GraphNodeType
    explanation: str
    evidence_keys: list[str] = Field(default_factory=list)


class GraphEdge(CamelModel):
    id: str
    source: str
    target: str
    label: str | None = None
    relation: Literal[
        "flows_to",
        "contains",
        "produces",
        "depends_on",
        "compares_with",
        "improves",
        "motivates",
        "uses",
    ] | None = None


class PaperGraph(CamelModel):
    diagram_type: Literal["pipeline", "architecture", "concept_map", "comparison"] = "concept_map"
    nodes: list[GraphNode] = Field(min_length=1, max_length=12)
    edges: list[GraphEdge] = Field(default_factory=list)


class PaperConcept(CamelModel):
    name: str
    explanation: str


class DeepPaperAnalysis(CamelModel):
    title: str | None = None
    levels: AnalysisLevels
    important_visuals: list[ImportantVisual] = Field(default_factory=list)
    graph: PaperGraph
    concepts: list[PaperConcept] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class SearchRequest(CamelModel):
    query: str = Field(min_length=3, max_length=500)
    limit: int = Field(default=10, ge=1, le=20)


class ResearchIntent(CamelModel):
    original_query: str
    normalized_topic: str
    core_concepts: list[str] = Field(default_factory=list)
    related_terms: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    target_problems: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(min_length=1, max_length=5)


class PaperCandidate(CamelModel):
    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str = ""
    relevance_score: float = Field(ge=0, le=1)
    relevance_reason: str
    source: Literal["Semantic Scholar", "arXiv", "Local"]
    doi: str | None = None
    arxiv_id: str | None = None
    source_url: str | None = None


class SearchResponse(CamelModel):
    intent: ResearchIntent
    candidates: list[PaperCandidate]
    reranked: bool


class IngestResponse(CamelModel):
    paper_id: UUID
    job_id: UUID
    status: Literal["queued", "processing", "ready", "failed"]


class IngestSource(CamelModel):
    source_url: str = Field(min_length=8, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    analysis_model: AnalysisModel = "gemini-3.8-flash"


class JobStatusResponse(CamelModel):
    id: UUID
    paper_id: UUID
    stage: str
    status: str
    progress: int = Field(ge=0, le=100)
    message: str
    attempts: int
    error_code: str | None = None
    error_message: str | None = None
    updated_at: str


class VisualResponse(CamelModel):
    id: UUID
    paper_id: UUID
    type: VisualType
    label: str | None = None
    caption: str | None = None
    page_number: int
    page_image_url: str
    cropped_image_url: str | None = None
    visual_description: str
    visual_interpretation: str
    concepts: list[str] = Field(default_factory=list)


class GraphNodeResponse(CamelModel):
    id: str
    label: str
    type: GraphNodeType
    explanation: str
    evidence_ids: list[str] = Field(default_factory=list)


class GraphResponse(CamelModel):
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdge]


class PaperResponse(CamelModel):
    id: UUID
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    status: str
    page_count: int = 0
    source_url: str | None = None
    analysis_model: str | None = None
    analysis_fallback_used: bool = False
    job: JobStatusResponse | None = None
    analysis: PaperAnalysis | None = None
    graph: GraphResponse | None = None
    visuals: list[VisualResponse] = Field(default_factory=list)


class RetrievedContext(CamelModel):
    id: str
    paper_id: UUID
    source_type: Literal["text", "visual", "table", "graph_node"]
    content: str
    similarity: float
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    visual_id: UUID | None = None
    image_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImagePayload(CamelModel):
    id: str
    type: Literal["page", "figure", "table"]
    page_number: int
    url: str | None = None
    base64: str | None = None
    label: str | None = None


class ConversationTurn(CamelModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class RAGModelInput(CamelModel):
    question: str
    explanation_level: ExplanationLevel = "student"
    history: list[ConversationTurn] = Field(default_factory=list, max_length=12)
    evidence: list[RetrievedContext]
    images: list[ImagePayload]


class EvidenceCitation(CamelModel):
    id: str
    paper_id: UUID
    source_type: Literal["text", "visual", "table", "graph_node"]
    page: int | None = None
    section: str | None = None
    label: str | None = None
    excerpt: str
    asset_id: UUID | None = None


class RAGAnswer(CamelModel):
    answer: str
    citations: list[EvidenceCitation] = Field(default_factory=list)
    model: str
    fallback_used: bool = False
    grounded: bool = False
    latency_ms: int


class RAGRequest(CamelModel):
    paper_ids: list[UUID] = Field(min_length=1, max_length=3)
    question: str = Field(min_length=3, max_length=2000)
    explanation_level: ExplanationLevel = "student"
    history: list[ConversationTurn] = Field(default_factory=list, max_length=12)
