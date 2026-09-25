"use client";

import Image from "next/image";
import Link from "next/link";
import { ArrowClockwise, ArrowLeft, ArrowUpRight, BookOpen, Check, Graph, Info, PaperPlaneTilt, Plus, Quotes, Sparkle, Table, X } from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { askPaper, assetUrl, getPaper, getPaperAnalysis, retryJob } from "@/lib/api";
import { demoPaper } from "@/lib/mock-data";
import { cn, formatAuthors, formatPageCount } from "@/lib/utils";
import { ANALYSIS_MODELS, type ConversationTurn, type EvidenceCitation, type ExplanationLevel, type PaperAnalysis, type PaperDetail, type VisualArtifact } from "@/lib/types";
import { ConceptGraph } from "./concept-graph";
import { AnswerModelSelector, useAnswerModel } from "./model-selection";

type WorkspaceTab = "analysis" | "graph" | "visual" | "ask";
type ReadyPaper = PaperDetail & {
  analysis: NonNullable<PaperDetail["analysis"]>;
  graph: NonNullable<PaperDetail["graph"]>;
};

function isReadyPaper(paper: PaperDetail): paper is ReadyPaper {
  return paper.status === "ready" && paper.analysis !== null && paper.graph !== null;
}

function analysisModelLabel(model: string | undefined) {
  if (model === "gemma-4-31b-it") return "Gemma 4 31B";
  return ANALYSIS_MODELS.find((option) => option.id === model)?.label ?? model;
}

const tabs: { id: WorkspaceTab; label: string; icon: typeof BookOpen }[] = [
  { id: "analysis", label: "Analysis", icon: BookOpen },
  { id: "graph", label: "Concept graph", icon: Graph },
  { id: "visual", label: "Visual evidence", icon: Table },
  { id: "ask", label: "Ask paper", icon: Quotes },
];

function PaperPreview({ visual }: { visual: VisualArtifact }) {
  const imageUrl = visual.croppedImageUrl ?? visual.pageImageUrl;
  if (imageUrl) {
    return (
      <div className="relative aspect-[16/9] overflow-hidden rounded-[10px] border border-[var(--line)] bg-[var(--surface-muted)]">
        <Image src={assetUrl(imageUrl)} alt={`${visual.label ?? visual.type} from page ${visual.pageNumber}`} fill sizes="(max-width: 768px) 100vw, 560px" className="object-contain" unoptimized />
      </div>
    );
  }

  return (
    <div className="paper-lines relative aspect-[16/9] overflow-hidden rounded-[10px] border border-[var(--line)] bg-[var(--surface-strong)] p-5">
      <div className="absolute right-5 top-5 flex h-20 w-32 items-end gap-1 border-b border-l border-[var(--line-strong)] p-2">
        {[34, 48, 42, 65, 59, 76, 70].map((height, index) => <span key={index} className={cn("w-2 rounded-t-sm", index > 4 ? "bg-[var(--accent)]" : "bg-[var(--accent-soft)]")} style={{ height: `${height}%` }} />)}
      </div>
      <div className="relative max-w-[48%] pt-1">
        <p className="mono-font text-[9px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">Page {visual.pageNumber}</p>
        <div className="mt-4 space-y-2">
          <div className="h-1.5 w-full rounded bg-[var(--line-strong)]" />
          <div className="h-1.5 w-4/5 rounded bg-[var(--line)]" />
          <div className="h-1.5 w-full rounded bg-[var(--line)]" />
          <div className="h-1.5 w-3/5 rounded bg-[var(--line)]" />
        </div>
        <div className="mt-7 border-l-2 border-[var(--accent)] pl-3 text-[10px] leading-4 text-[var(--ink-muted)]">{visual.caption ?? "Visual artifact linked to the paper analysis."}</div>
      </div>
      <div className="absolute bottom-4 left-5 right-5 flex items-center justify-between text-[9px] uppercase tracking-[0.14em] text-[var(--ink-faint)]"><span>{visual.type}</span><span>original page raster</span></div>
    </div>
  );
}

function PaperProcessing({ paper, onRetry, retrying }: { paper: PaperDetail; onRetry?: () => void; retrying?: boolean }) {
  const failed = paper.status === "failed";
  const progress = paper.job?.progress ?? (paper.status === "processing" ? 55 : 0);
  return (
    <main className="mx-auto w-full max-w-[760px] px-5 py-16 lg:px-8">
      <div className="border-t border-[var(--line-strong)] pt-5">
        <p className="mono-font text-[10px] uppercase tracking-[0.18em] text-[var(--accent-strong)]">{failed ? "Ingestion failed" : "Preparing paper"}</p>
        <h1 className="display-font mt-4 text-4xl leading-none sm:text-5xl">{failed ? "This paper needs another pass." : "Making the paper searchable."}</h1>
        <p className="mt-4 text-[13px] leading-6 text-[var(--ink-muted)]">{paper.job?.message ?? "The worker is extracting text, rendering pages, and preparing the paper analysis."}</p>
        {paper.analysisModel && <p className="mt-3 text-[11px] text-[var(--ink-faint)]">Index model: {analysisModelLabel(paper.analysisModel)}{paper.analysisFallbackUsed ? " · fallback path active" : ""}</p>}
        <div className="mt-8 h-2 overflow-hidden rounded-full bg-[var(--surface-muted)]" aria-label={`${progress}% complete`}>
          <div className={cn("h-full rounded-full transition-all", failed ? "bg-[var(--accent-strong)]" : "bg-[var(--accent)]")} style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} />
        </div>
        <div className="mt-3 flex items-center justify-between text-[10px] text-[var(--ink-faint)]"><span>{paper.job?.stage ?? paper.status}</span><span>{progress}%</span></div>
        {failed && paper.job?.errorMessage && <p className="mt-7 rounded-[10px] border border-[var(--line)] bg-[var(--surface)] p-4 text-[11px] leading-5 text-[var(--ink-muted)]">{paper.job.errorMessage}</p>}
        <div className="mt-8 flex flex-wrap items-center gap-5">
          <Link href="/search" className="inline-flex items-center gap-2 text-[12px] font-semibold text-[var(--accent-strong)] hover:underline hover:underline-offset-4"><ArrowLeft size={15} /> Back to discovery</Link>
          {failed && onRetry && <button type="button" onClick={onRetry} disabled={retrying} className="inline-flex items-center gap-2 rounded-[9px] bg-[var(--ink)] px-3 py-2 text-[11px] font-semibold text-[var(--canvas)] transition hover:bg-[var(--accent-strong)] disabled:cursor-wait disabled:opacity-50"><ArrowClockwise size={14} className={retrying ? "animate-spin" : undefined} /> {retrying ? "Retrying..." : "Retry ingestion"}</button>}
        </div>
      </div>
    </main>
  );
}

function PaperError({ message }: { message: string }) {
  return (
    <main className="mx-auto w-full max-w-[760px] px-5 py-16 lg:px-8">
      <div className="border-t border-[var(--line-strong)] pt-5">
        <p className="mono-font text-[10px] uppercase tracking-[0.18em] text-[var(--accent-strong)]">Paper unavailable</p>
        <h1 className="display-font mt-4 text-4xl leading-none sm:text-5xl">This paper is not in the workspace.</h1>
        <p className="mt-4 text-[13px] leading-6 text-[var(--ink-muted)]">{message}</p>
        <Link href="/search" className="mt-8 inline-flex items-center gap-2 text-[12px] font-semibold text-[var(--accent-strong)] hover:underline hover:underline-offset-4"><ArrowLeft size={15} /> Back to discovery</Link>
      </div>
    </main>
  );
}

function EvidenceRow({ label, page, sourceType, onClick }: { label: string; page?: number; sourceType: string; onClick?: () => void }) {
  return (
    <button type="button" onClick={onClick} className="group flex w-full items-center justify-between border-t border-[var(--line)] py-3 text-left transition hover:bg-[var(--surface-muted)]">
      <span className="flex min-w-0 items-center gap-3"><span className="mono-font text-[10px] text-[var(--accent)]">{sourceType}</span><span className="truncate text-[12px] font-medium text-[var(--ink)]">{label}</span></span>
      <span className="flex shrink-0 items-center gap-2 text-[10px] text-[var(--ink-faint)]">{page ? `p. ${page}` : "linked"}<ArrowUpRight size={13} className="opacity-0 transition group-hover:opacity-100" /></span>
    </button>
  );
}

function AnalysisView({ analysis, level, setLevel, loading }: { analysis: PaperAnalysis; level: ExplanationLevel; setLevel: (level: ExplanationLevel) => void; loading: boolean }) {
  const levels: { id: ExplanationLevel; label: string; description: string }[] = [
    { id: "simple", label: "Simple", description: "Intuition first" },
    { id: "student", label: "Student", description: "CS background" },
    { id: "researcher", label: "Researcher", description: "Full terminology" },
  ];

  return (
    <div className="space-y-8">
      <div className="grid gap-2 sm:grid-cols-3">
        {levels.map((item) => <button key={item.id} type="button" onClick={() => setLevel(item.id)} disabled={loading} className={cn("rounded-[11px] border px-3 py-3 text-left transition", level === item.id ? "border-[var(--accent)] bg-[var(--accent-soft)]" : "border-[var(--line)] bg-[var(--surface)] hover:border-[var(--line-strong)]", loading && "cursor-wait opacity-70")}><span className="block text-[12px] font-semibold">{item.label}</span><span className="mt-1 block text-[10px] text-[var(--ink-muted)]">{loading && level === item.id ? "Loading explanation..." : item.description}</span></button>)}
      </div>

      <section className="border-t border-[var(--line-strong)] pt-5">
        <p className="mono-font text-[10px] uppercase tracking-[0.18em] text-[var(--accent-strong)]">TL;DR</p>
        <p className="mt-3 max-w-[780px] text-[18px] leading-8 tracking-[-0.02em] text-[var(--ink)]">{analysis.summary}</p>
      </section>

      <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="space-y-8">
          <section><h2 className="text-[13px] font-semibold">The problem</h2><p className="mt-2 text-[13px] leading-6 text-[var(--ink-muted)]">{analysis.problem}</p></section>
          <section><h2 className="text-[13px] font-semibold">Why it matters</h2><p className="mt-2 text-[13px] leading-6 text-[var(--ink-muted)]">{analysis.motivation}</p></section>
          <section className="border-l-2 border-[var(--accent)] pl-4"><h2 className="text-[13px] font-semibold">Key insight</h2><p className="mt-2 text-[13px] leading-6 text-[var(--ink-muted)]">{analysis.keyInsight}</p></section>
          <section><h2 className="text-[13px] font-semibold">How the method works</h2><p className="mt-2 text-[13px] leading-6 text-[var(--ink-muted)]">{analysis.method}</p></section>
        </div>
        <aside className="space-y-7">
          <section className="rounded-[14px] border border-[var(--line)] bg-[var(--surface)] p-4"><h2 className="text-[13px] font-semibold">Contributions</h2><ul className="mt-3 space-y-3">{analysis.contributions.map((item) => <li key={item} className="flex gap-2 text-[12px] leading-5 text-[var(--ink-muted)]"><Check size={15} className="mt-0.5 shrink-0 text-[var(--positive)]" />{item}</li>)}</ul></section>
          <section><h2 className="text-[13px] font-semibold">Limitations</h2><ul className="mt-3 space-y-2">{analysis.limitations.map((item) => <li key={item} className="text-[12px] leading-5 text-[var(--ink-muted)]">{item}</li>)}</ul></section>
        </aside>
      </div>

      <section className="border-t border-[var(--line)] pt-5"><h2 className="text-[13px] font-semibold">Prerequisites</h2><div className="mt-3 flex flex-wrap gap-2">{analysis.prerequisites.map((item) => <span key={item} className="rounded-full border border-[var(--line)] px-3 py-1.5 text-[10px] text-[var(--ink-muted)]">{item}</span>)}</div></section>
    </div>
  );
}

const CHAT_STORAGE_PREFIX = "paperscope-chat-v1:";
const MAX_CHAT_MESSAGES = 24;
const MAX_HISTORY_TURNS = 12;

type ChatMessage = ConversationTurn & {
  id: string;
  citations?: EvidenceCitation[];
  model?: string;
  fallbackUsed?: boolean;
  grounded?: boolean;
  latencyMs?: number;
  demo?: boolean;
};

function chatStorageKey(paperId: string) {
  return `${CHAT_STORAGE_PREFIX}${paperId}`;
}

function isStoredChatMessage(value: unknown): value is ChatMessage {
  if (typeof value !== "object" || value === null) return false;
  const message = value as Partial<ChatMessage>;
  return typeof message.id === "string" && (message.role === "user" || message.role === "assistant") && typeof message.content === "string";
}

function readChat(paperId: string): ChatMessage[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.sessionStorage.getItem(chatStorageKey(paperId));
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isStoredChatMessage).slice(-MAX_CHAT_MESSAGES) : [];
  } catch {
    return [];
  }
}

function writeChat(paperId: string, messages: ChatMessage[]) {
  if (typeof window === "undefined") return;
  try {
    const key = chatStorageKey(paperId);
    if (messages.length === 0) {
      window.sessionStorage.removeItem(key);
      return;
    }
    window.sessionStorage.setItem(key, JSON.stringify(messages.slice(-MAX_CHAT_MESSAGES)));
  } catch {
    // Session storage can be unavailable in private browsing or when quota is exhausted.
  }
}

function newChatMessageId() {
  return globalThis.crypto?.randomUUID?.() ?? `chat-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function requestHistory(messages: ChatMessage[]): ConversationTurn[] {
  return messages.slice(-MAX_HISTORY_TURNS).map(({ role, content }) => ({ role, content: content.slice(0, 4000) }));
}

function ChatBubble({ message, onCitation }: { message: ChatMessage; onCitation: (citation: EvidenceCitation) => void }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[82%] rounded-[14px] rounded-br-[4px] bg-[var(--ink)] px-4 py-3 text-[13px] leading-6 text-[var(--canvas)] shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <article className="max-w-[94%] rounded-[14px] rounded-bl-[4px] border border-[var(--line)] bg-[var(--canvas)] px-4 py-4 shadow-sm">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px]">
          <span className="font-semibold text-[var(--accent-strong)]">PaperScope</span>
          <span className={message.grounded ? "text-[var(--positive)]" : "text-[var(--ink-faint)]"}>{message.grounded ? "grounded" : "evidence needs review"}</span>
          {message.fallbackUsed && <span className="text-[var(--ink-faint)]">fallback</span>}
          {message.model && <span className="text-[var(--ink-faint)]">· {message.demo ? "fixture response" : message.model}</span>}
        </div>
        <AnswerText content={message.content} />
        {message.citations && message.citations.length > 0 && (
          <div className="mt-4 border-t border-[var(--line)] pt-3">
            <p className="mb-1 text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">Evidence</p>
            {message.citations.map((citation) => <EvidenceRow key={citation.id} label={citation.label ?? citation.excerpt} page={citation.page} sourceType={citation.id} onClick={() => onCitation(citation)} />)}
          </div>
        )}
      </article>
    </div>
  );
}

function AnswerText({ content }: { content: string }) {
  return (
    <div className="mt-3 space-y-2 text-[14px] leading-7 text-[var(--ink)]">
      {content.split("\n").map((line, index) => {
        const bullet = line.trim().match(/^(?:[-*])\s+(.+)$/);
        if (bullet) {
          return <p key={`${index}-${line}`} className="flex gap-2 pl-1"><span className="text-[var(--accent)]">•</span><span>{bullet[1]}</span></p>;
        }
        return line.trim() ? <p key={`${index}-${line}`}>{line}</p> : <div key={`${index}-blank`} className="h-1" aria-hidden="true" />;
      })}
    </div>
  );
}

function AskView({ paperId, onCitation, suggestedQuestion }: { paperId: string; onCitation: (citation: EvidenceCitation) => void; suggestedQuestion?: string | null }) {
  const { answerModel } = useAnswerModel();
  const [question, setQuestion] = useState(() => suggestedQuestion ?? "");
  const [messages, setMessages] = useState<ChatMessage[]>(() => readChat(paperId));
  const [loading, setLoading] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const requestVersion = useRef(0);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const suggestions = ["Explain the main contribution.", "What does Figure 3 show?", "Why does throughput flatten?", "What assumptions does this method make?"];

  useEffect(() => {
    writeChat(paperId, messages);
  }, [paperId, messages]);

  useEffect(() => {
    const container = chatScrollRef.current;
    if (container) container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
  }, [messages, pendingQuestion, loading]);

  async function ask(value = question) {
    const trimmed = value.trim();
    if (!trimmed || loading) return;
    const currentRequest = ++requestVersion.current;
    const history = requestHistory(messages);
    setQuestion("");
    setPendingQuestion(trimmed);
    setLoading(true);
    setError(null);
    try {
      const result = await askPaper([paperId], trimmed, history, answerModel);
      const userMessage: ChatMessage = { id: newChatMessageId(), role: "user", content: trimmed };
      const assistantMessage: ChatMessage = {
        id: newChatMessageId(),
        role: "assistant",
        content: result.answer,
        citations: result.citations,
        model: result.model,
        fallbackUsed: result.fallbackUsed,
        grounded: result.grounded,
        latencyMs: result.latencyMs,
        demo: result.demo,
      };
      if (currentRequest === requestVersion.current) {
        setMessages((current) => [...current, userMessage, assistantMessage].slice(-MAX_CHAT_MESSAGES));
      }
    } catch (requestError) {
      if (currentRequest === requestVersion.current) {
        setQuestion(trimmed);
        setError(requestError instanceof Error ? requestError.message : "The question could not be answered.");
      }
    } finally {
      if (currentRequest === requestVersion.current) {
        setPendingQuestion(null);
        setLoading(false);
      }
    }
  }

  function clearChat() {
    requestVersion.current += 1;
    setMessages([]);
    setQuestion("");
    setPendingQuestion(null);
    setLoading(false);
    setError(null);
  }

  return (
    <div className="max-w-[820px]">
      <div className="mb-6 flex items-start gap-3 border-b border-[var(--line)] pb-5"><span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[9px] bg-[var(--accent-soft)] text-[var(--accent-strong)]"><Sparkle size={16} /></span><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-[14px] font-semibold">Ask about this paper</h2><div className="flex flex-wrap items-center gap-3"><AnswerModelSelector /><button type="button" onClick={clearChat} className="inline-flex items-center gap-1.5 rounded-full border border-[var(--line)] px-2.5 py-1.5 text-[10px] font-semibold text-[var(--ink-muted)] transition hover:border-[var(--accent)] hover:text-[var(--accent-strong)]"><Plus size={12} /> New chat</button></div></div><p className="mt-1 text-[12px] leading-5 text-[var(--ink-muted)]">Follow up naturally. Your conversation is stored in this browser tab only and disappears when you start a new chat or close the tab.</p></div></div>
      <div className="overflow-hidden rounded-[16px] border border-[var(--line)] bg-[var(--surface)]">
        <div ref={chatScrollRef} className="max-h-[520px] space-y-5 overflow-y-auto p-4 sm:p-5" aria-live="polite">
          {messages.length === 0 && !pendingQuestion && <div className="rounded-[12px] border border-dashed border-[var(--line-strong)] bg-[var(--canvas)] p-5"><p className="text-[13px] font-semibold">Start with a question</p><p className="mt-2 max-w-[560px] text-[12px] leading-5 text-[var(--ink-muted)]">Ask for the paper&apos;s intuition, a method detail, or what a figure actually shows. Follow-ups keep the conversation in context.</p><div className="mt-4 flex flex-wrap gap-2">{suggestions.map((suggestion) => <button key={suggestion} type="button" onClick={() => { void ask(suggestion); }} disabled={loading} className="rounded-full border border-[var(--line)] px-3 py-2 text-left text-[11px] text-[var(--ink-muted)] transition hover:border-[var(--accent)] hover:text-[var(--accent-strong)] disabled:cursor-wait disabled:opacity-50">{suggestion}</button>)}</div></div>}
          {messages.map((message) => <ChatBubble key={message.id} message={message} onCitation={onCitation} />)}
          {pendingQuestion && <div className="flex justify-end"><div className="max-w-[82%] rounded-[14px] rounded-br-[4px] bg-[var(--ink)] px-4 py-3 text-[13px] leading-6 text-[var(--canvas)] opacity-80">{pendingQuestion}</div></div>}
          {loading && <div className="flex justify-start"><div className="rounded-[14px] rounded-bl-[4px] border border-[var(--line)] bg-[var(--canvas)] px-4 py-3 text-[11px] text-[var(--ink-muted)]"><span className="mr-2 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--accent)]" /> Reading the paper&apos;s evidence…</div></div>}
        </div>
        <form onSubmit={(event) => { event.preventDefault(); void ask(); }} className="flex items-end gap-3 border-t border-[var(--line)] bg-[var(--canvas)] p-3 focus-within:border-[var(--accent)] focus-within:ring-4 focus-within:ring-[var(--accent-soft)]"><label htmlFor="paper-question" className="sr-only">Ask a question</label><textarea id="paper-question" rows={2} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(); } }} placeholder="Ask a follow-up…" className="min-h-[44px] min-w-0 flex-1 resize-none bg-transparent px-2 py-2 text-[13px] leading-5 outline-none placeholder:text-[var(--ink-faint)]" /><button type="submit" disabled={loading || !question.trim()} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[9px] bg-[var(--ink)] text-[var(--canvas)] transition hover:bg-[var(--accent-strong)] disabled:cursor-wait disabled:opacity-50" aria-label="Send question">{loading ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-[var(--canvas)] border-t-transparent" /> : <PaperPlaneTilt size={16} />}</button></form>
      </div>
      {error && <p role="alert" className="mt-4 rounded-[10px] border border-[var(--line)] bg-[var(--surface)] p-3 text-[11px] leading-5 text-[var(--accent-strong)]">{error}</p>}
    </div>
  );
}

function EvidenceDrawer({
  paper,
  citation,
  visuals,
  evidenceIds,
  onClose,
}: {
  paper: ReadyPaper;
  citation: EvidenceCitation | null;
  visuals: VisualArtifact[];
  evidenceIds: string[];
  onClose: () => void;
}) {
  const visual = visuals[0];
  const imageUrl = paper.demo
    ? undefined
    : visual?.croppedImageUrl ?? visual?.pageImageUrl ?? (evidenceIds[0] ? `/v1/assets/${evidenceIds[0]}` : undefined) ?? (citation?.page ? `/v1/papers/${paper.id}/pages/${citation.page}` : undefined);
  const label = citation?.label ?? visual?.label ?? citation?.sourceType ?? "Paper evidence";

  return (
    <div className="fixed inset-0 z-50 bg-black/20 p-4 sm:p-6" role="presentation" onClick={onClose}>
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="Evidence details"
        className="ml-auto flex h-full w-full max-w-[460px] flex-col overflow-y-auto rounded-[16px] border border-[var(--line)] bg-[var(--canvas)] p-5 shadow-[0_24px_80px_rgba(31,46,40,0.2)] sm:p-6"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-[var(--line-strong)] pb-4">
          <div>
            <p className="mono-font text-[10px] uppercase tracking-[0.18em] text-[var(--accent-strong)]">Evidence {citation?.id ?? "linked"}</p>
            <h2 className="mt-2 text-[16px] font-semibold leading-5">{label}</h2>
          </div>
          <button type="button" onClick={onClose} className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[8px] border border-[var(--line)] text-[var(--ink-muted)] transition hover:border-[var(--accent)] hover:text-[var(--accent-strong)]" aria-label="Close evidence drawer"><X size={15} /></button>
        </div>

        {imageUrl ? <div className="relative mt-5 aspect-[4/3] overflow-hidden rounded-[11px] border border-[var(--line)] bg-[var(--surface-muted)]"><Image src={assetUrl(imageUrl)} alt={`${label}${citation?.page ? ` on page ${citation.page}` : ""}`} fill sizes="460px" className="object-contain" unoptimized /></div> : <div className="paper-lines mt-5 rounded-[11px] border border-[var(--line)] bg-[var(--surface)] p-5 text-[11px] leading-5 text-[var(--ink-muted)]">The original page image will appear here after this paper finishes local ingestion.</div>}

        <div className="mt-5 space-y-5">
          {citation && <section><p className="mono-font text-[10px] uppercase tracking-[0.16em] text-[var(--ink-faint)]">Excerpt</p><p className="mt-2 text-[13px] leading-6 text-[var(--ink-muted)]">{citation.excerpt}</p></section>}
          {visual && <section><p className="mono-font text-[10px] uppercase tracking-[0.16em] text-[var(--ink-faint)]">Visual interpretation</p><p className="mt-2 text-[13px] leading-6 text-[var(--ink-muted)]">{visual.visualInterpretation}</p></section>}
          <div className="flex flex-wrap gap-2 text-[10px] text-[var(--ink-faint)]"><span className="rounded-full bg-[var(--surface-muted)] px-2.5 py-1">{citation?.sourceType ?? visual?.type ?? "evidence"}</span>{citation?.page && <span className="rounded-full bg-[var(--surface-muted)] px-2.5 py-1">Page {citation.page}</span>}{citation?.section && <span className="rounded-full bg-[var(--surface-muted)] px-2.5 py-1">{citation.section}</span>}</div>
        </div>
      </aside>
    </div>
  );
}

export function PaperWorkspace({ paperId }: { paperId: string }) {
  const [paper, setPaper] = useState<PaperDetail | null>(paperId === "eagle-3" ? demoPaper : null);
  const [tab, setTab] = useState<WorkspaceTab>("analysis");
  const [askSeed, setAskSeed] = useState<string | null>(null);
  const [level, setLevel] = useState<ExplanationLevel>("student");
  const [activeAnalysis, setActiveAnalysis] = useState<PaperAnalysis | null>(paper?.analysis ?? null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState<string[]>([]);
  const [selectedCitation, setSelectedCitation] = useState<EvidenceCitation | null>(null);
  const [loading, setLoading] = useState(paper === null);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function loadPaper() {
      try {
        const result = await getPaper(paperId);
        if (active) {
          setPaper(result);
          setActiveAnalysis(result.analysis);
          setLoading(false);
          if (result.status === "queued" || result.status === "processing") {
            timer = setTimeout(() => { void loadPaper(); }, 2000);
          }
        }
      } catch (requestError) {
        if (active) {
          setError(requestError instanceof Error ? requestError.message : "The paper could not be loaded.");
          setLoading(false);
        }
      }
    }
    void loadPaper();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [paperId, reloadToken]);

  async function retryIngestion() {
    if (!paper?.job || retrying) return;
    setRetrying(true);
    setError(null);
    setLoading(true);
    try {
      await retryJob(paper.job.id);
      setReloadToken((value) => value + 1);
    } catch (requestError) {
      setLoading(false);
      setError(requestError instanceof Error ? requestError.message : "The ingestion retry could not be started.");
    } finally {
      setRetrying(false);
    }
  }

  const selectedVisuals = useMemo(() => paper?.visuals.filter((visual) => selectedEvidence.some((id) => visual.id === id || visual.label === id)) ?? [], [paper, selectedEvidence]);

  function openCitation(citation: EvidenceCitation) {
    setSelectedCitation(citation);
    setSelectedEvidence(citation.assetId ? [citation.assetId] : []);
  }

  function closeEvidence() {
    setSelectedCitation(null);
    setSelectedEvidence([]);
  }

  async function changeExplanationLevel(nextLevel: ExplanationLevel) {
    if (nextLevel === level || analysisLoading) return;
    if (paperId === "eagle-3" && paper?.analysis) {
      setLevel(nextLevel);
      setActiveAnalysis(paper.analysis);
      return;
    }
    const previousLevel = level;
    setAnalysisLoading(true);
    try {
      setActiveAnalysis(await getPaperAnalysis(paperId, nextLevel));
      setLevel(nextLevel);
    } catch {
      setLevel(previousLevel);
      setActiveAnalysis(paper?.analysis ?? null);
    } finally {
      setAnalysisLoading(false);
    }
  }

  if (error) return <PaperError message={error} />;

  if (loading || !paper) {
    return <main className="mx-auto w-full max-w-[1400px] px-5 py-12 lg:px-8"><div className="animate-pulse space-y-5"><div className="h-3 w-24 rounded bg-[var(--surface-muted)]" /><div className="h-12 max-w-[700px] rounded bg-[var(--surface-muted)]" /><div className="h-[420px] rounded-[16px] bg-[var(--surface-muted)]" /></div></main>;
  }

  if (!isReadyPaper(paper) || !activeAnalysis) {
    return <PaperProcessing paper={paper} onRetry={paper.status === "failed" ? () => { void retryIngestion(); } : undefined} retrying={retrying} />;
  }

  return (
    <main className="mx-auto w-full max-w-[1400px] px-5 py-8 lg:px-8 lg:py-10">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <Link href="/search" className="inline-flex items-center gap-2 text-[12px] font-medium text-[var(--ink-muted)] transition hover:text-[var(--accent-strong)]"><ArrowLeft size={15} /> Back to discovery</Link>
        <div className="flex items-center gap-2 text-[10px] text-[var(--positive)]"><span className="h-1.5 w-1.5 rounded-full bg-[var(--positive)]" /> Analysis ready</div>
      </div>

      <header className="mt-9 max-w-[960px]">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)]"><span className="text-[var(--accent-strong)]">{paper.venue}</span><span>{paper.year}</span><span>{formatPageCount(paper.pageCount)}</span>{paper.analysisModel && <span className="rounded-full bg-[var(--accent-soft)] px-2.5 py-1 normal-case tracking-normal text-[var(--accent-strong)]">Indexed with {analysisModelLabel(paper.analysisModel)}{paper.analysisFallbackUsed ? " · fallback" : ""}</span>}</div>
        <h1 className="display-font mt-4 text-4xl leading-[1.02] tracking-[-0.04em] sm:text-6xl">{paper.title}</h1>
        <p className="mt-4 text-[13px] text-[var(--ink-muted)]">{formatAuthors(paper.authors)}</p>
      </header>

      <div className="mt-10 grid gap-8 lg:grid-cols-[minmax(0,1fr)_300px] lg:gap-16">
        <section className="min-w-0">
          <div className="flex gap-1 overflow-x-auto border-b border-[var(--line)]" role="tablist" aria-label="Paper views">
            {tabs.map(({ id, label, icon: Icon }) => <button key={id} type="button" role="tab" aria-selected={tab === id} onClick={() => { setTab(id); if (id !== "ask") setAskSeed(null); }} className={cn("relative flex shrink-0 items-center gap-2 px-3 py-3 text-[12px] font-medium text-[var(--ink-muted)] transition hover:text-[var(--ink)]", tab === id && "text-[var(--ink)] after:absolute after:inset-x-2 after:bottom-[-1px] after:h-0.5 after:bg-[var(--accent)]")}><Icon size={15} />{label}</button>)}
          </div>
          <div className="pt-8">
            {tab === "analysis" && <AnalysisView analysis={activeAnalysis} level={level} setLevel={(nextLevel) => { void changeExplanationLevel(nextLevel); }} loading={analysisLoading} />}
            {tab === "graph" && <div className="space-y-5"><div><h2 className="text-[15px] font-semibold">How the paper fits together</h2><p className="mt-1 text-[12px] leading-5 text-[var(--ink-muted)]">A simplified representation generated from the paper&apos;s prose, figures, and tables.</p></div><ConceptGraph graph={paper.graph} onEvidence={(evidenceIds) => { setSelectedCitation(null); setSelectedEvidence(evidenceIds); }} /></div>}
            {tab === "visual" && <div className="space-y-6"><div><h2 className="text-[15px] font-semibold">The figures, in context</h2><p className="mt-1 text-[12px] leading-5 text-[var(--ink-muted)]">Visual descriptions are searchable. The original page remains the source of truth.</p></div>{paper.visuals.map((visual) => <article key={visual.id} className="border-t border-[var(--line)] pt-5"><PaperPreview visual={visual} /><div className="mt-4 grid gap-5 sm:grid-cols-[180px_1fr]"><div><p className="mono-font text-[10px] uppercase tracking-[0.15em] text-[var(--accent-strong)]">{visual.label ?? visual.type}</p><p className="mt-2 text-[11px] text-[var(--ink-faint)]">Page {visual.pageNumber}</p><p className="mt-4 text-[12px] font-medium leading-5">{visual.caption}</p></div><div><p className="text-[12px] leading-6 text-[var(--ink-muted)]">{visual.visualDescription}</p><p className="mt-3 border-l-2 border-[var(--accent)] pl-3 text-[12px] leading-6 text-[var(--ink-muted)]">{visual.visualInterpretation}</p><div className="mt-4 flex flex-wrap gap-2">{visual.concepts.map((concept) => <span key={concept} className="rounded-full bg-[var(--surface-muted)] px-2.5 py-1 text-[10px] text-[var(--ink-muted)]">{concept}</span>)}</div></div></div></article>)}</div>}
            {tab === "ask" && <AskView key={`${paper.id}-${askSeed ?? "none"}`} paperId={paper.id} onCitation={openCitation} suggestedQuestion={askSeed} />}
          </div>
        </section>

        <aside className="lg:pt-[52px]">
          <div className="space-y-7 lg:sticky lg:top-[92px]">
            <section className="border-t border-[var(--line-strong)] pt-4"><div className="flex items-center justify-between"><h2 className="text-[12px] font-semibold">Paper signals</h2><Info size={14} className="text-[var(--ink-faint)]" /></div><div className="mt-4 grid grid-cols-3 gap-2"><div className="rounded-[10px] bg-[var(--surface)] p-3"><p className="mono-font text-[19px] tracking-[-0.05em]">{paper.pageCount}</p><p className="mt-1 text-[10px] text-[var(--ink-faint)]">pages</p></div><div className="rounded-[10px] bg-[var(--surface)] p-3"><p className="mono-font text-[19px] tracking-[-0.05em]">{paper.visuals.length}</p><p className="mt-1 text-[10px] text-[var(--ink-faint)]">visuals</p></div><div className="rounded-[10px] bg-[var(--surface)] p-3"><p className="mono-font text-[19px] tracking-[-0.05em]">{paper.graph.nodes.length}</p><p className="mt-1 text-[10px] text-[var(--ink-faint)]">nodes</p></div></div></section>
            <section><h2 className="text-[12px] font-semibold">Suggested questions</h2><div className="mt-3 space-y-1">{["Explain the main contribution.", "What does Figure 3 show?", "Why does throughput flatten?"].map((question) => <button key={question} type="button" onClick={() => { setAskSeed(question); setTab("ask"); }} className="flex w-full items-start justify-between gap-3 border-b border-[var(--line)] py-3 text-left text-[11px] leading-4 text-[var(--ink-muted)] transition hover:text-[var(--accent-strong)]"><span>{question}</span><ArrowUpRight size={13} className="mt-0.5 shrink-0" /></button>)}</div></section>
            <section><h2 className="text-[12px] font-semibold">Linked evidence</h2><div className="mt-2">{paper.visuals.map((visual, index) => <EvidenceRow key={visual.id} label={visual.label ?? visual.type} page={visual.pageNumber} sourceType={`V${index + 1}`} onClick={() => { setSelectedCitation(null); setSelectedEvidence([visual.id]); }} />)}{paper.demo ? <EvidenceRow label="Method and acceptance boundary" page={4} sourceType="E1" onClick={() => openCitation({ id: "E1", paperId: paper.id, sourceType: "text", page: 4, section: "Method", excerpt: "The verifier accepts the longest consistent prefix of the proposed token block." })} /> : paper.visuals.length === 0 ? <p className="border-t border-[var(--line)] py-3 text-[11px] leading-5 text-[var(--ink-faint)]">Text evidence appears with citations after you ask a question.</p> : null}</div>{selectedVisuals.length > 0 && <button type="button" onClick={closeEvidence} className="mt-3 inline-flex items-center gap-1 text-[10px] text-[var(--ink-faint)] hover:text-[var(--ink)]"><X size={12} /> Clear selection</button>}</section>
            <Link href="/compare" className="flex items-center justify-between rounded-[12px] border border-[var(--line)] bg-[var(--surface)] p-3 text-[11px] font-semibold transition hover:border-[var(--accent)]"><span>Compare with another paper</span><ArrowUpRight size={14} className="text-[var(--accent)]" /></Link>
          </div>
        </aside>
      </div>
      {(selectedCitation || selectedEvidence.length > 0) && <EvidenceDrawer paper={paper} citation={selectedCitation} visuals={selectedVisuals} evidenceIds={selectedEvidence} onClose={closeEvidence} />}
    </main>
  );
}
