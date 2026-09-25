export type ExplanationLevel = "simple" | "student" | "researcher";

export const ANALYSIS_MODELS = [
  {
    id: "gemini-3.8-flash",
    label: "Gemini Flash 3.8",
    shortLabel: "Flash 3.8",
    fallback: "3.7 → 3.6 → 3.5 → Gemma 4",
  },
  {
    id: "gemini-3.7-flash",
    label: "Gemini Flash 3.7",
    shortLabel: "Flash 3.7",
    fallback: "3.6 → 3.5 → Gemma 4",
  },
  {
    id: "gemini-3.6-flash",
    label: "Gemini Flash 3.6",
    shortLabel: "Flash 3.6",
    fallback: "3.5 → Gemma 4",
  },
  {
    id: "gemini-3.5-flash",
    label: "Gemini Flash 3.5",
    shortLabel: "Flash 3.5",
    fallback: "Gemma 4",
  },
] as const;

export type AnalysisModel = (typeof ANALYSIS_MODELS)[number]["id"];
export const DEFAULT_ANALYSIS_MODEL: AnalysisModel = "gemini-3.8-flash";

export const ANSWER_MODELS = [
  { id: "muse", label: "Muse Glimmer" },
  { id: "nemotron", label: "Nemotron" },
  { id: "gemma", label: "Gemma 4 31B" },
] as const;
export type AnswerModel = (typeof ANSWER_MODELS)[number]["id"];
export const DEFAULT_ANSWER_MODEL: AnswerModel = "muse";

export type PaperStatus = "queued" | "processing" | "ready" | "failed";

export type SourceType = "text" | "visual" | "table" | "graph_node";

export interface PaperCandidate {
  id: string;
  title: string;
  authors: string[];
  year: number | null;
  venue?: string;
  abstract: string;
  relevanceScore: number;
  relevanceReason: string;
  source: "Semantic Scholar" | "arXiv" | "Local";
  doi?: string;
  arxivId?: string;
  sourceUrl?: string;
  ready?: boolean;
}

export interface ResearchIntent {
  originalQuery: string;
  normalizedTopic: string;
  coreConcepts: string[];
  relatedTerms: string[];
  methods: string[];
  targetProblems: string[];
  searchQueries: string[];
}

export interface SearchResponse {
  intent: ResearchIntent;
  candidates: PaperCandidate[];
  reranked: boolean;
  demo?: boolean;
}

export interface GraphNode {
  id: string;
  label: string;
  type: "input" | "component" | "process" | "decision" | "concept" | "result" | "baseline" | "proposed";
  explanation: string;
  evidenceIds: string[];
  x?: number;
  y?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  relation?: string;
}

export interface VisualArtifact {
  id: string;
  paperId: string;
  type: "architecture" | "diagram" | "plot" | "chart" | "table" | "algorithm" | "equation" | "other";
  label?: string;
  caption?: string;
  pageNumber: number;
  pageImageUrl?: string;
  croppedImageUrl?: string;
  visualDescription: string;
  visualInterpretation: string;
  concepts: string[];
}

export interface PaperAnalysis {
  summary: string;
  problem: string;
  motivation: string;
  keyInsight: string;
  method: string;
  contributions: string[];
  results: string;
  limitations: string[];
  prerequisites: string[];
  glossary: { term: string; definition: string }[];
}

export interface PaperDetail {
  id: string;
  title: string;
  authors: string[];
  year: number | null;
  venue: string | null;
  status: PaperStatus;
  pageCount: number;
  sourceUrl?: string;
  analysisModel?: string;
  analysisFallbackUsed?: boolean;
  job?: JobStatus | null;
  analysis: PaperAnalysis | null;
  graph: { nodes: GraphNode[]; edges: GraphEdge[] } | null;
  visuals: VisualArtifact[];
  demo?: boolean;
}

export interface JobStatus {
  id: string;
  paperId: string;
  stage: string;
  status?: string;
  progress: number;
  message: string;
  attempts?: number;
  errorCode?: string;
  errorMessage?: string;
  updatedAt: string;
}

export interface EvidenceCitation {
  id: string;
  paperId: string;
  sourceType: SourceType;
  page?: number;
  section?: string;
  label?: string;
  excerpt: string;
  assetId?: string;
}

export type ConversationRole = "user" | "assistant";

export interface ConversationTurn {
  role: ConversationRole;
  content: string;
}

export interface RagAnswer {
  answer: string;
  citations: EvidenceCitation[];
  model: string;
  fallbackUsed: boolean;
  grounded: boolean;
  latencyMs: number;
  demo?: boolean;
}
