"use client";

import { ArrowUpRight, Check, FileArrowUp, Funnel, MagnifyingGlass, Sparkle, SpinnerGap, WarningCircle } from "@phosphor-icons/react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { ingestPdf, ingestSource, searchPapers } from "@/lib/api";
import { cn, formatAuthors, formatScore } from "@/lib/utils";
import { ANALYSIS_MODELS, type PaperCandidate, type SearchResponse } from "@/lib/types";
import { useAnalysisModel } from "./model-selection";

function CandidateRow({ paper, selected, opening, onToggle, onOpen }: { paper: PaperCandidate; selected: boolean; opening: boolean; onToggle: () => void; onOpen: () => void }) {
  const isDemo = paper.id === "eagle-3";
  const canOpen = isDemo || Boolean(paper.sourceUrl);
  return (
    <article className={cn("group relative border-t border-[var(--line)] py-5 transition", selected && "bg-[var(--accent-soft)]/35") }>
      <div className="grid gap-4 sm:grid-cols-[38px_1fr_auto] sm:gap-5">
        <button
          type="button"
          onClick={onToggle}
          disabled={opening}
          aria-pressed={selected}
          aria-label={`${selected ? "Remove" : "Select"} ${paper.title}`}
          className={cn("mt-0.5 flex h-7 w-7 items-center justify-center rounded-[8px] border border-[var(--line-strong)] bg-[var(--surface)] text-transparent transition hover:border-[var(--accent)]", selected && "border-[var(--accent)] bg-[var(--accent)] text-white")}
        >
          <Check size={15} weight="bold" />
        </button>
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-[var(--ink-faint)]">
            <span className="font-semibold uppercase tracking-[0.12em] text-[var(--accent-strong)]">{paper.source}</span>
            <span>{paper.year}</span>
            {paper.venue && <span>{paper.venue}</span>}
          </div>
          <h2 className="max-w-[740px] text-[17px] font-semibold leading-6 tracking-[-0.02em] text-[var(--ink)]">{paper.title}</h2>
          <p className="mt-1 text-[12px] text-[var(--ink-muted)]">{formatAuthors(paper.authors)}</p>
          <p className="mt-3 max-w-[760px] text-[13px] leading-6 text-[var(--ink-muted)]">{paper.abstract}</p>
          <p className="mt-3 flex items-start gap-2 text-[11px] leading-5 text-[var(--positive)]"><Sparkle size={14} className="mt-0.5 shrink-0" />{paper.relevanceReason}</p>
        </div>
        <div className="flex items-start justify-between gap-4 sm:flex-col sm:items-end">
          <div className="text-right">
            <span className="mono-font text-[18px] font-semibold tracking-[-0.04em] text-[var(--ink)]">{formatScore(paper.relevanceScore)}</span>
            <p className="mt-0.5 text-[10px] text-[var(--ink-faint)]">relevance</p>
          </div>
          <button type="button" onClick={onOpen} disabled={!canOpen || opening} title={!canOpen ? "No open PDF is available for this result" : undefined} className="inline-flex items-center gap-1 text-[12px] font-semibold text-[var(--ink-muted)] opacity-100 transition hover:text-[var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-40 sm:opacity-0 sm:group-hover:opacity-100">
            {opening ? <><SpinnerGap size={14} className="animate-spin" /> Importing...</> : isDemo ? "Open demo reader" : "Import & open"}
            {!opening && <ArrowUpRight size={14} />}
          </button>
        </div>
      </div>
    </article>
  );
}

function SearchSkeleton() {
  return (
    <div className="space-y-1" aria-label="Loading papers" aria-busy="true">
      {[1, 2, 3].map((item) => (
        <div key={item} className="animate-pulse border-t border-[var(--line)] py-6">
          <div className="grid gap-4 sm:grid-cols-[38px_1fr_90px]">
            <div className="h-7 w-7 rounded-[8px] bg-[var(--surface-muted)]" />
            <div className="space-y-3">
              <div className="h-2.5 w-32 rounded bg-[var(--surface-muted)]" />
              <div className="h-5 max-w-[520px] rounded bg-[var(--surface-muted)]" />
              <div className="h-3 w-full max-w-[680px] rounded bg-[var(--surface-muted)]" />
              <div className="h-3 w-3/4 rounded bg-[var(--surface-muted)]" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export function SearchExperience({ initialQuery }: { initialQuery: string }) {
  const router = useRouter();
  const { model: analysisModel } = useAnalysisModel();
  const [query, setQuery] = useState(initialQuery || "efficient speculative decoding for MoE inference");
  const [submittedQuery, setSubmittedQuery] = useState(initialQuery || "efficient speculative decoding for MoE inference");
  const [searchNonce, setSearchNonce] = useState(0);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [onlyOpenPdf, setOnlyOpenPdf] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [openingId, setOpeningId] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    searchPapers(submittedQuery).then((result) => {
      if (active) {
        setResponse(result);
      setLoading(false);
      }
    }).catch((error) => {
      if (active) {
        setSearchError(error instanceof Error ? error.message : "Research discovery could not be completed.");
        setLoading(false);
      }
    });
    return () => {
      active = false;
    };
  }, [searchNonce, submittedQuery]);

  const selectedPapers = useMemo(() => response?.candidates.filter((paper) => selected.includes(paper.id)) ?? [], [response, selected]);
  const visibleCandidates = useMemo(() => response?.candidates.filter((paper) => !onlyOpenPdf || Boolean(paper.sourceUrl)) ?? [], [onlyOpenPdf, response]);

  function rememberCandidates(candidates: PaperCandidate[]) {
    if (typeof window !== "undefined") window.sessionStorage.setItem("paperscope:selected-candidates", JSON.stringify(candidates));
  }

  function togglePaper(id: string) {
    setSelected((current) => current.includes(id) ? current.filter((item) => item !== id) : current.length < 3 ? [...current, id] : current);
  }

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextQuery = query.trim();
    if (nextQuery) {
      setLoading(true);
      setSearchError(null);
      setResponse(null);
      setSelected([]);
      setSubmittedQuery(nextQuery);
      setSearchNonce((current) => current + 1);
    }
  }

  async function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (file.type && file.type !== "application/pdf") {
      setUploadError("Choose a PDF file to start ingestion.");
      return;
    }
    setUploading(true);
    setUploadError(null);
    try {
      const result = await ingestPdf(file, analysisModel);
      router.push(`/papers/${result.paperId}`);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "The PDF could not be uploaded.");
    } finally {
      setUploading(false);
    }
  }

  async function handleOpenCandidate(paper: PaperCandidate) {
    if (openingId) return;
    rememberCandidates([paper]);
    if (paper.id === "eagle-3") {
      router.push(`/papers/${paper.id}`);
      return;
    }
    if (!paper.sourceUrl) {
      setUploadError("This result has no open PDF link. Choose a result with an available PDF or upload the paper.");
      return;
    }
    setOpeningId(paper.id);
    setUploadError(null);
    try {
      const result = await ingestSource(paper.sourceUrl, {
        title: paper.title,
        authors: paper.authors,
        year: paper.year,
        venue: paper.venue,
        doi: paper.doi,
        arxivId: paper.arxivId,
      }, analysisModel);
      router.push(`/papers/${result.paperId}`);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "The paper could not be imported.");
    } finally {
      setOpeningId(null);
    }
  }

  async function openSelected() {
    if (selectedPapers.length > 1) {
      if (selectedPapers.some((paper) => !paper.sourceUrl || paper.id === "eagle-3")) {
        setUploadError("Every selected paper needs an open PDF link before it can be compared.");
        return;
      }
      setOpeningId("__batch__");
      setUploadError(null);
      try {
        const imported = await Promise.all(selectedPapers.map((paper) => ingestSource(paper.sourceUrl as string, {
          title: paper.title,
          authors: paper.authors,
          year: paper.year,
          venue: paper.venue,
          doi: paper.doi,
          arxivId: paper.arxivId,
        }, analysisModel)));
        rememberCandidates(selectedPapers);
        router.push(`/compare?papers=${imported.map((result) => encodeURIComponent(result.paperId)).join(",")}`);
      } catch (error) {
        setUploadError(error instanceof Error ? error.message : "One or more selected papers could not be imported.");
      } finally {
        setOpeningId(null);
      }
      return;
    }
    if (selectedPapers.length === 1) void handleOpenCandidate(selectedPapers[0]);
  }

  return (
    <main className="mx-auto w-full max-w-[1400px] px-5 py-9 lg:px-8 lg:py-12">
      <div className="grid gap-12 lg:grid-cols-[minmax(0,1fr)_290px] lg:gap-20">
        <section>
          <form onSubmit={submit} className="flex max-w-[860px] items-center gap-3 border-b border-[var(--ink)] pb-3">
            <MagnifyingGlass size={20} className="shrink-0 text-[var(--accent)]" />
            <label htmlFor="search-query" className="sr-only">Research topic</label>
            <input id="search-query" value={query} onChange={(event) => setQuery(event.target.value)} className="min-w-0 flex-1 bg-transparent text-[20px] font-medium tracking-[-0.025em] outline-none placeholder:text-[var(--ink-faint)]" />
            <button type="submit" className="inline-flex shrink-0 items-center gap-2 rounded-[9px] bg-[var(--ink)] px-4 py-2.5 text-[12px] font-semibold text-[var(--canvas)] transition hover:bg-[var(--accent-strong)] active:translate-y-px">Search <ArrowUpRight size={14} /></button>
          </form>

          <div className="mt-10 flex items-end justify-between gap-5">
            <div>
              <p className="mono-font text-[10px] uppercase tracking-[0.2em] text-[var(--ink-faint)]">Discovery</p>
              <h1 className="display-font mt-3 text-4xl leading-none sm:text-5xl">Relevant work, in context.</h1>
            </div>
            <button type="button" aria-pressed={onlyOpenPdf} onClick={() => setOnlyOpenPdf((current) => !current)} className={cn("hidden items-center gap-2 rounded-[9px] border px-3 py-2 text-[12px] font-medium transition sm:inline-flex", onlyOpenPdf ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent-strong)]" : "border-[var(--line)] text-[var(--ink-muted)] hover:border-[var(--line-strong)] hover:text-[var(--ink)]")}><Funnel size={15} /> {onlyOpenPdf ? "Open PDFs" : "Filters"}</button>
          </div>

          {searchError && <div className="mt-7 flex items-start justify-between gap-4 border-l-2 border-[var(--accent-strong)] bg-[var(--surface)] px-3 py-3 text-[11px] text-[var(--ink-muted)]"><span><span className="font-semibold text-[var(--ink)]">Search unavailable.</span> {searchError}</span><button type="button" onClick={() => { setLoading(true); setSearchError(null); setResponse(null); setSelected([]); setSearchNonce((current) => current + 1); }} className="shrink-0 font-semibold text-[var(--accent-strong)] hover:underline">Retry</button></div>}

          {loading ? <div className="mt-8"><SearchSkeleton /></div> : response ? (
            <>
              <div className="mt-8 grid gap-5 border-y border-[var(--line)] py-4 sm:grid-cols-3">
                <div><p className="text-[10px] uppercase tracking-[0.16em] text-[var(--ink-faint)]">Interpreted topic</p><p className="mt-1 text-[13px] font-semibold">{response.intent.normalizedTopic}</p></div>
                <div><p className="text-[10px] uppercase tracking-[0.16em] text-[var(--ink-faint)]">Core concepts</p><p className="mt-1 text-[13px] text-[var(--ink-muted)]">{response.intent.coreConcepts.join(", ")}</p></div>
                <div><p className="text-[10px] uppercase tracking-[0.16em] text-[var(--ink-faint)]">Search paths</p><p className="mt-1 text-[13px] text-[var(--ink-muted)]">{response.intent.searchQueries.length} expanded queries</p></div>
              </div>
              <div className="mt-7">
                <div className="mb-1 flex items-center justify-between text-[11px] text-[var(--ink-faint)]"><span>{visibleCandidates.length} papers surfaced</span><span>{response.reranked ? "AI reranked" : "Original ranking"}</span></div>
                {visibleCandidates.map((paper) => <CandidateRow key={paper.id} paper={paper} selected={selected.includes(paper.id)} opening={openingId === paper.id || (openingId === "__batch__" && selected.includes(paper.id))} onToggle={() => togglePaper(paper.id)} onOpen={() => void handleOpenCandidate(paper)} />)}
              </div>
            </>
          ) : null}
        </section>

        <aside className="lg:pt-[184px]">
          <div className="sticky top-[100px] border-t border-[var(--line-strong)] pt-4">
            <div className="mb-8 border-b border-[var(--line)] pb-7">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-[12px] font-semibold">Import a paper</p>
                  <p className="mt-2 text-[11px] leading-5 text-[var(--ink-muted)]">Upload a PDF to extract pages, visuals, and a searchable concept graph.</p>
                </div>
                <FileArrowUp size={17} className="mt-0.5 text-[var(--accent)]" />
              </div>
              {(() => {
                const selectedModel = ANALYSIS_MODELS.find((option) => option.id === analysisModel) ?? ANALYSIS_MODELS[0];
                return (
                  <div className="mt-4 rounded-[10px] border border-[var(--line)] bg-[var(--surface)] px-3 py-2.5">
                    <div className="flex items-center justify-between gap-3 text-[10px]">
                      <span className="uppercase tracking-[0.14em] text-[var(--ink-faint)]">Index model</span>
                      <span className="font-semibold text-[var(--ink)]">{selectedModel.shortLabel}</span>
                    </div>
                    <p className="mt-1 text-[10px] leading-4 text-[var(--ink-muted)]">Fallback path: {selectedModel.fallback}</p>
                  </div>
                );
              })()}
              <label className={cn("mt-4 flex cursor-pointer items-center justify-center gap-2 rounded-[10px] border border-dashed border-[var(--line-strong)] px-3 py-3 text-[11px] font-semibold text-[var(--ink-muted)] transition hover:border-[var(--accent)] hover:text-[var(--accent-strong)]", uploading && "pointer-events-none opacity-60")}>
                {uploading ? <><SpinnerGap size={15} className="animate-spin" /> Uploading...</> : <><FileArrowUp size={15} /> Choose PDF</>}
                <input type="file" accept="application/pdf,.pdf" className="sr-only" onChange={handleUpload} disabled={uploading} />
              </label>
              {uploadError && <p className="mt-3 flex items-start gap-1.5 text-[10px] leading-4 text-[var(--accent-strong)]"><WarningCircle size={13} className="mt-0.5 shrink-0" />{uploadError}</p>}
            </div>
            <p className="text-[12px] font-semibold">Build a reading set</p>
            <p className="mt-2 text-[12px] leading-5 text-[var(--ink-muted)]">Select up to three papers to compare methods, results, and assumptions side by side.</p>
            <div className="mt-5 space-y-2">
              {selectedPapers.length === 0 ? <p className="rounded-[10px] border border-dashed border-[var(--line-strong)] p-3 text-[11px] leading-5 text-[var(--ink-faint)]">Your selected papers will appear here.</p> : selectedPapers.map((paper) => <div key={paper.id} className="flex items-start gap-2 rounded-[10px] bg-[var(--surface)] p-3 text-[11px] leading-4"><span className="mono-font text-[var(--accent)]">{String(selected.indexOf(paper.id) + 1).padStart(2, "0")}</span><span>{paper.title}</span></div>)}
            </div>
            <button type="button" onClick={() => { void openSelected(); }} disabled={selectedPapers.length === 0 || Boolean(openingId)} className={cn("mt-5 inline-flex w-full items-center justify-center gap-2 rounded-[10px] bg-[var(--ink)] px-4 py-3 text-[12px] font-semibold text-[var(--canvas)] transition hover:bg-[var(--accent-strong)] active:translate-y-px disabled:cursor-not-allowed disabled:opacity-40")}>
              {selectedPapers.length > 1 ? "Import & compare" : "Open selected paper"}
              <ArrowUpRight size={15} />
            </button>
            <p className="mt-3 text-center text-[10px] text-[var(--ink-faint)]">{selectedPapers.length}/3 selected</p>
          </div>
        </aside>
      </div>
    </main>
  );
}
