"use client";

import Link from "next/link";
import { ArrowLeft, ArrowUpRight, Check, GitMerge, PaperPlaneTilt, Sparkle } from "@phosphor-icons/react";
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { askPaper, getPaper } from "@/lib/api";
import { demoCandidates } from "@/lib/mock-data";
import { cn } from "@/lib/utils";
import type { PaperCandidate, RagAnswer } from "@/lib/types";
import { AnswerModelSelector, useAnswerModel } from "./model-selection";

const defaultPaperIds = demoCandidates.slice(0, 2).map((paper) => paper.id);
const ragPaperIdPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const candidateStorageKey = "paperscope:selected-candidates";

function canQueryPaper(paper: PaperCandidate) {
  return ragPaperIdPattern.test(paper.id) && paper.ready === true;
}

function subscribeToCandidateStorage(onChange: () => void) {
  window.addEventListener("storage", onChange);
  return () => window.removeEventListener("storage", onChange);
}

function getCandidateStorageValue() {
  return window.sessionStorage.getItem(candidateStorageKey) ?? "";
}

function parseStoredCandidates(value: string): PaperCandidate[] {
  try {
    const parsed = JSON.parse(value || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function candidateFromPaper(paper: Awaited<ReturnType<typeof getPaper>>): PaperCandidate {
  return {
    id: paper.id,
    title: paper.title,
    authors: paper.authors,
    year: paper.year,
    venue: paper.venue ?? undefined,
    abstract: paper.analysis?.summary ?? "This paper has been imported into the local workspace.",
    relevanceScore: 1,
    relevanceReason: "Imported into this local reading set.",
    source: "Local",
    ready: paper.status === "ready",
  };
}

export function CompareWorkspace({ paperIds }: { paperIds: string[] }) {
  const { answerModel } = useAnswerModel();
  const selectedIds = useMemo(() => (paperIds.length >= 2 ? paperIds : defaultPaperIds).slice(0, 3), [paperIds]);
  const storedCandidateValue = useSyncExternalStore(subscribeToCandidateStorage, getCandidateStorageValue, () => "");
  const storedCandidates = useMemo(() => parseStoredCandidates(storedCandidateValue), [storedCandidateValue]);
  const knownPapers = useMemo(() => selectedIds.map((id) => demoCandidates.find((paper) => paper.id === id) ?? storedCandidates.find((paper) => paper.id === id)).filter((paper): paper is PaperCandidate => Boolean(paper)), [selectedIds, storedCandidates]);
  const [loadedPapers, setLoadedPapers] = useState<PaperCandidate[]>([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<RagAnswer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const papers = useMemo(() => [...knownPapers, ...loadedPapers.filter((paper) => !knownPapers.some((known) => known.id === paper.id))], [knownPapers, loadedPapers]);
  const missingIds = useMemo(() => selectedIds.filter((id) => {
    const paper = papers.find((candidate) => candidate.id === id);
    return !paper || (ragPaperIdPattern.test(id) && paper.ready !== true);
  }), [papers, selectedIds]);
  const loadingPapers = !error && missingIds.length > 0;
  const canAsk = papers.length === selectedIds.length && selectedIds.every((id) => {
    const paper = papers.find((candidate) => candidate.id === id);
    return paper ? canQueryPaper(paper) : false;
  });
  const missingKey = missingIds.join(",");

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ids = missingKey ? missingKey.split(",") : [];
    if (ids.length === 0) {
      return () => { active = false; };
    }

    async function loadStatuses() {
      try {
        const loaded = await Promise.all(ids.map((id) => getPaper(id).then(candidateFromPaper)));
        if (!active) return;
        setLoadedPapers((current) => {
          const byId = new Map(current.map((paper) => [paper.id, paper]));
          loaded.forEach((paper) => byId.set(paper.id, paper));
          return [...byId.values()];
        });
        if (loaded.some((paper) => !paper.ready)) {
          timer = setTimeout(() => { void loadStatuses(); }, 2000);
        }
      } catch (requestError) {
        if (active) setError(requestError instanceof Error ? requestError.message : "One or more papers could not be loaded.");
      }
    }

    void loadStatuses();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [missingKey]);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading || loadingPapers) return;
    if (!canAsk) {
      setError("Import the selected discovery results before asking a grounded comparison question.");
      return;
    }
    setQuestion(trimmed);
    setLoading(true);
    setError(null);
    try {
      setAnswer(await askPaper(selectedIds, trimmed, [], answerModel));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The comparison question could not be answered.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto w-full max-w-[1400px] px-5 py-8 lg:px-8 lg:py-10">
      <Link href="/search" className="inline-flex items-center gap-2 text-[12px] font-medium text-[var(--ink-muted)] transition hover:text-[var(--accent-strong)]"><ArrowLeft size={15} /> Back to discovery</Link>
      <header className="mt-10 max-w-[850px]"><div className="flex items-center gap-2 text-[var(--accent-strong)]"><GitMerge size={18} /><span className="text-[11px] font-semibold uppercase tracking-[0.18em]">Selected papers</span></div><h1 className="display-font mt-4 text-5xl leading-[0.98] tracking-[-0.04em] sm:text-6xl">Same question. Different answers.</h1><p className="mt-5 max-w-[650px] text-[14px] leading-6 text-[var(--ink-muted)]">Compare the assumptions, mechanisms, and evidence that separate nearby approaches.</p></header>

      {papers.some((paper) => paper.id === "eagle-3") && <p className="mt-7 max-w-[720px] rounded-[10px] border border-[var(--line)] bg-[var(--surface)] px-4 py-3 text-[11px] leading-5 text-[var(--ink-muted)]"><span className="font-semibold text-[var(--ink)]">Demo context only.</span> The sample reader is available for exploring the interface, but it is not included in grounded comparison retrieval. Import real PDFs to ask across papers.</p>}

      <section className="mt-12 grid gap-5 lg:grid-cols-3">
        {loadingPapers && <div className="h-64 animate-pulse rounded-[14px] bg-[var(--surface-muted)]" />}
        {papers.map((paper, index) => <article key={paper.id} className="border-t-2 border-[var(--ink)] pt-4"><div className="flex items-center justify-between gap-4"><span className="mono-font text-[10px] uppercase tracking-[0.16em] text-[var(--accent-strong)]">Paper {index + 1}</span><span className="text-[11px] text-[var(--ink-faint)]">{paper.year ?? "Year unavailable"} / {paper.source}</span></div><h2 className="mt-4 text-[18px] font-semibold leading-6 tracking-[-0.02em]">{paper.title}</h2><p className="mt-2 text-[12px] text-[var(--ink-muted)]">{paper.authors.join(", ")}</p><p className="mt-5 text-[13px] leading-6 text-[var(--ink-muted)]">{paper.abstract}</p><div className={cn("mt-6 flex items-center gap-2 border-t border-[var(--line)] pt-3 text-[11px]", canQueryPaper(paper) ? "text-[var(--positive)]" : "text-[var(--ink-faint)]")}><Check size={14} /> {paper.id === "eagle-3" ? "Demo context only" : canQueryPaper(paper) ? "Ready for grounded retrieval" : "Import this PDF before asking"}</div></article>)}
      </section>

      <section className="mt-14 grid gap-10 border-t border-[var(--line-strong)] pt-6 lg:grid-cols-[0.7fr_1.3fr]">
        <div><p className="text-[12px] font-semibold">Ask across {selectedIds.length} papers</p><p className="mt-2 text-[12px] leading-5 text-[var(--ink-muted)]">{canAsk ? "The answer identifies which selected paper supports each claim." : "Discovery results are shown for context. Ingest the PDFs before asking a grounded comparison question."}</p></div>
        <div><div className="mb-3 flex justify-end"><AnswerModelSelector /></div><form onSubmit={submit} className="flex items-center gap-3 rounded-[13px] border border-[var(--line-strong)] bg-[var(--surface)] p-2 focus-within:border-[var(--accent)] focus-within:ring-4 focus-within:ring-[var(--accent-soft)]"><label htmlFor="compare-question" className="sr-only">Ask a comparison question</label><input id="compare-question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Which method handles expert imbalance better?" className="min-w-0 flex-1 bg-transparent px-3 py-2 text-[13px] outline-none placeholder:text-[var(--ink-faint)]" /><button type="submit" disabled={loading || loadingPapers || !canAsk} className="flex h-10 w-10 items-center justify-center rounded-[9px] bg-[var(--ink)] text-[var(--canvas)] transition hover:bg-[var(--accent-strong)] disabled:opacity-50" aria-label="Ask comparison question">{loading ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-[var(--canvas)] border-t-transparent" /> : <PaperPlaneTilt size={16} />}</button></form>
          {error && <p role="alert" className="mt-4 rounded-[10px] border border-[var(--line)] bg-[var(--surface)] p-3 text-[11px] leading-5 text-[var(--accent-strong)]">{error}</p>}
          {answer && <div className="mt-6 rounded-[14px] border border-[var(--line)] bg-[var(--surface)] p-5"><div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] uppercase tracking-[0.15em] text-[var(--positive)]"><Sparkle size={13} /> {answer.grounded ? "Grounded comparison" : "Evidence needs review"}<span className="normal-case tracking-normal text-[var(--ink-faint)]">· {answer.model}{answer.fallbackUsed ? " · fallback" : ""}</span></div><p className="mt-4 text-[14px] leading-7">{answer.answer}</p><div className="mt-5 flex flex-wrap gap-2">{answer.citations.map((citation) => <span key={citation.id} className="rounded-full bg-[var(--accent-soft)] px-2.5 py-1 text-[10px] font-semibold text-[var(--accent-strong)]">{citation.id} / {citation.label ?? citation.sourceType}</span>)}</div></div>}
          <Link href="/search" className="mt-5 inline-flex items-center gap-2 text-[12px] font-semibold text-[var(--accent-strong)] hover:underline hover:underline-offset-4">Add another paper <ArrowUpRight size={14} /></Link>
        </div>
      </section>
    </main>
  );
}
