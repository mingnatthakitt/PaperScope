import { demoPaper } from "./mock-data";
import { DEFAULT_ANALYSIS_MODEL } from "./types";
import type { AnalysisModel, AnswerModel, ConversationTurn, ExplanationLevel, JobStatus, PaperAnalysis, PaperDetail, RagAnswer, SearchResponse } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export function assetUrl(path: string) {
  return path.startsWith("http") ? path : `${API_URL}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    const body = await response.text();
    let detail: string | undefined;
    try {
      const parsed = JSON.parse(body) as { detail?: unknown };
      if (typeof parsed.detail === "string") detail = parsed.detail;
    } catch {
      // Some upstream failures return an empty or non-JSON body.
    }
    throw new Error(detail || `PaperScope API returned ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function searchPapers(query: string): Promise<SearchResponse> {
  return request<SearchResponse>("/v1/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit: 10 }),
  });
}

export async function getPaper(paperId: string): Promise<PaperDetail> {
  try {
    return await request<PaperDetail>(`/v1/papers/${paperId}`);
  } catch (error) {
    if (paperId === "eagle-3") return demoPaper;
    throw error;
  }
}

export async function getJob(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`/v1/jobs/${jobId}`);
}

export async function getPaperAnalysis(paperId: string, level: ExplanationLevel): Promise<PaperAnalysis> {
  return request<PaperAnalysis>(`/v1/papers/${paperId}/analysis?level=${level}`);
}

export async function ingestPdf(file: File, analysisModel: AnalysisModel = DEFAULT_ANALYSIS_MODEL): Promise<{ paperId: string; jobId: string; status: string }> {
  const form = new FormData();
  form.append("file", file);
  form.append("analysisModel", analysisModel);
  return request<{ paperId: string; jobId: string; status: string }>("/v1/papers/ingest", {
    method: "POST",
    body: form,
  });
}

export async function ingestSource(sourceUrl: string, metadata: Record<string, unknown> = {}, analysisModel: AnalysisModel = DEFAULT_ANALYSIS_MODEL): Promise<{ paperId: string; jobId: string; status: string }> {
  return request<{ paperId: string; jobId: string; status: string }>("/v1/papers/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sourceUrl, metadata, analysisModel }),
  });
}

export async function retryJob(jobId: string): Promise<{ paperId: string; jobId: string; status: string }> {
  return request<{ paperId: string; jobId: string; status: string }>(`/v1/jobs/${jobId}/retry`, {
    method: "POST",
  });
}

export async function askPaper(paperIds: string[], question: string, history: ConversationTurn[] = [], answerModel: AnswerModel = "muse"): Promise<RagAnswer> {
  try {
    return await request<RagAnswer>("/v1/rag/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paperIds, question, explanationLevel: "student", history, answerModel }),
    });
  } catch (error) {
    if (paperIds.length !== 1 || !paperIds.includes("eagle-3")) throw error;
    const isVisualQuestion = /(figure|plot|curve|chart|diagram|table|architecture|shown|visual)/i.test(question);
    return {
      answer: isVisualQuestion
        ? "Figure 5 shows that throughput improves while the draft block is still removing serial verification work. It then flattens because expert dispatch and memory traffic become the limiting costs, so adding more proposed tokens no longer increases useful parallelism."
        : "The paper uses a small draft model to propose several tokens, then checks those tokens in parallel with an expert-aware verifier. The key constraint is the acceptance boundary: only the longest consistent prefix is emitted before the loop continues.",
      citations: isVisualQuestion
        ? [{ id: "V1", paperId: "eagle-3", sourceType: "visual", page: 8, label: "Figure 5", excerpt: "Throughput rises quickly and then reaches a broad plateau as draft length increases." }]
        : [{ id: "E1", paperId: "eagle-3", sourceType: "text", page: 4, section: "Method", excerpt: "The verifier accepts the longest consistent prefix of the proposed token block." }],
      model: "demo / local fixture",
      fallbackUsed: false,
      grounded: true,
      latencyMs: 680,
      demo: true,
    };
  }
}
