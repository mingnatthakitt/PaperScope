"use client";

import Link from "next/link";
import { ArrowUpRight, CaretRight, FileText, Graph, MagnifyingGlass, Sparkle, Table, Waveform } from "@phosphor-icons/react";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

const topics = [
  "speculative decoding for MoE inference",
  "vision transformers for medical imaging",
  "retrieval augmented generation evaluation",
];

function EvidencePreview() {
  return (
    <div className="relative overflow-hidden rounded-[20px] border border-[var(--line)] bg-[var(--surface)] p-4 shadow-[var(--shadow)]">
      <div className="page-grid absolute inset-0 opacity-60" />
      <div className="relative">
        <div className="mb-4 flex items-center justify-between border-b border-[var(--line)] pb-3">
          <div>
            <p className="mono-font text-[10px] uppercase tracking-[0.2em] text-[var(--ink-faint)]">Paper graph</p>
            <p className="mt-1 text-[13px] font-semibold">EAGLE-3 / method overview</p>
          </div>
          <span className="rounded-full bg-[var(--accent-soft)] px-2.5 py-1 text-[10px] font-semibold text-[var(--accent-strong)]">6 nodes</span>
        </div>

        <div className="grid grid-cols-[1fr_34px_1fr] items-center gap-2">
          <div className="space-y-2">
            <div className="rounded-[12px] border border-[var(--line)] bg-[var(--surface-strong)] p-3">
              <div className="flex items-center gap-2 text-[11px] font-semibold"><FileText size={14} className="text-[var(--accent)]" /> Prompt tokens</div>
              <p className="mt-1 text-[10px] leading-4 text-[var(--ink-muted)]">The decoding loop begins here.</p>
            </div>
            <div className="rounded-[12px] border border-[var(--line)] bg-[var(--surface-strong)] p-3">
              <div className="flex items-center gap-2 text-[11px] font-semibold"><Sparkle size={14} className="text-[var(--accent)]" /> Draft block</div>
              <p className="mt-1 text-[10px] leading-4 text-[var(--ink-muted)]">Several candidate tokens proposed together.</p>
            </div>
          </div>
          <div className="flex flex-col items-center gap-2 text-[var(--ink-faint)]">
            <div className="h-px w-full bg-[var(--line-strong)]" />
            <CaretRight size={16} />
            <div className="h-px w-full bg-[var(--line-strong)]" />
          </div>
          <div className="space-y-2">
            <div className="rounded-[12px] border border-[var(--accent)] bg-[var(--accent-soft)] p-3">
              <div className="flex items-center gap-2 text-[11px] font-semibold"><Graph size={14} className="text-[var(--accent-strong)]" /> Parallel verifier</div>
              <p className="mt-1 text-[10px] leading-4 text-[var(--ink-muted)]">The target model checks the block once.</p>
            </div>
            <div className="rounded-[12px] border border-[var(--line)] bg-[var(--surface-strong)] p-3">
              <div className="flex items-center gap-2 text-[11px] font-semibold"><Waveform size={14} className="text-[var(--positive)]" /> Acceptance boundary</div>
              <p className="mt-1 text-[10px] leading-4 text-[var(--ink-muted)]">Only the consistent prefix is emitted.</p>
            </div>
          </div>
        </div>

        <div className="mt-4 flex items-center justify-between rounded-[12px] border border-dashed border-[var(--line-strong)] px-3 py-2.5">
          <span className="flex items-center gap-2 text-[11px] font-medium"><Table size={14} className="text-[var(--ink-muted)]" /> Figure 3 linked to 2 passages</span>
          <ArrowUpRight size={14} className="text-[var(--ink-muted)]" />
        </div>
      </div>
    </div>
  );
}

export function HomeExperience() {
  const router = useRouter();
  const [query, setQuery] = useState("");

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = query.trim();
    router.push(value ? `/search?q=${encodeURIComponent(value)}` : "/search");
  }

  return (
    <main>
      <section className="relative overflow-hidden border-b border-[var(--line)]">
        <div className="page-grid pointer-events-none absolute inset-0 opacity-70" />
        <div className="relative mx-auto grid min-h-[calc(100dvh-68px)] w-full max-w-[1400px] items-center gap-12 px-5 py-16 lg:grid-cols-[1.03fr_0.97fr] lg:gap-20 lg:px-8 lg:py-20">
          <div className="enter-rise max-w-[690px]">
            <div className="mb-6 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--accent-strong)]">
              <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
              Multimodal research workspace
            </div>
            <h1 className="display-font max-w-[650px] text-5xl leading-[0.98] text-[var(--ink)] sm:text-6xl lg:text-[76px]">
              Understand research beyond the abstract.
            </h1>
            <p className="mt-7 max-w-[540px] text-[16px] leading-7 text-[var(--ink-muted)]">
              Find the papers that matter, see what their figures actually show, and ask grounded questions with the evidence in view.
            </p>

            <form onSubmit={submit} className="mt-9 max-w-[610px]">
              <label htmlFor="research-topic" className="mb-2 block text-[12px] font-semibold text-[var(--ink)]">Research topic</label>
              <div className="flex flex-col gap-2 rounded-[14px] border border-[var(--line-strong)] bg-[var(--surface)] p-2 shadow-[0_12px_30px_rgba(31,46,40,0.06)] transition focus-within:border-[var(--accent)] focus-within:ring-4 focus-within:ring-[var(--accent-soft)] sm:flex-row sm:items-center">
                <MagnifyingGlass size={20} className="ml-2 hidden shrink-0 text-[var(--ink-faint)] sm:block" />
                <input
                  id="research-topic"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Try: efficient speculative decoding for MoE inference"
                  className="min-w-0 flex-1 bg-transparent px-3 py-3 text-[14px] text-[var(--ink)] outline-none placeholder:text-[var(--ink-faint)]"
                />
                <button type="submit" className="inline-flex h-11 items-center justify-center gap-2 rounded-[10px] bg-[var(--ink)] px-5 text-[13px] font-semibold text-[var(--canvas)] transition hover:bg-[var(--accent-strong)] active:translate-y-px">
                  Explore
                  <ArrowUpRight size={16} />
                </button>
              </div>
            </form>

            <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2">
              {topics.map((topic) => (
                <button key={topic} type="button" onClick={() => setQuery(topic)} className="text-left text-[11px] text-[var(--ink-muted)] underline decoration-[var(--line-strong)] underline-offset-4 transition hover:text-[var(--accent-strong)]">
                  {topic}
                </button>
              ))}
            </div>
          </div>

          <div className="enter-rise-delay relative mx-auto w-full max-w-[590px] lg:mr-0">
            <EvidencePreview />
            <div className="mt-3 grid grid-cols-3 gap-3 text-center">
              {[
                ["Text", "sections and method"],
                ["Visuals", "figures and tables"],
                ["Graph", "ideas and relations"],
              ].map(([label, copy]) => (
                <div key={label} className="border-t border-[var(--line)] pt-3">
                  <p className="text-[12px] font-semibold">{label}</p>
                  <p className="mt-1 text-[10px] leading-4 text-[var(--ink-faint)]">{copy}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto grid w-full max-w-[1400px] gap-10 px-5 py-16 lg:grid-cols-[0.75fr_1.25fr] lg:px-8 lg:py-24">
        <div>
          <p className="mono-font text-[10px] uppercase tracking-[0.2em] text-[var(--ink-faint)]">The working model</p>
          <h2 className="display-font mt-4 max-w-[420px] text-4xl leading-[1.02] sm:text-5xl">A paper is more than its text.</h2>
          <p className="mt-5 max-w-[390px] text-[14px] leading-6 text-[var(--ink-muted)]">PaperScope keeps the prose, the pixels, and the simplified idea structure connected so each answer can point back to something real.</p>
          <Link href="/search" className="mt-7 inline-flex items-center gap-2 text-[13px] font-semibold text-[var(--accent-strong)] hover:underline hover:underline-offset-4">
            Start with a paper
            <CaretRight size={15} />
          </Link>
        </div>
        <div className="grid gap-5 sm:grid-cols-3">
          <div className="border-t border-[var(--line-strong)] pt-4 sm:col-span-2">
            <p className="text-[13px] font-semibold">Evidence stays attached</p>
            <p className="mt-2 max-w-[320px] text-[13px] leading-6 text-[var(--ink-muted)]">Passages map to pages. Visual descriptions map to the original page pixels. Graph nodes map to both.</p>
          </div>
          <div className="border-t border-[var(--line-strong)] pt-4">
            <p className="text-[13px] font-semibold">One analysis, reused</p>
            <p className="mt-2 text-[13px] leading-6 text-[var(--ink-muted)]">Deep paper understanding is persisted, then made searchable locally.</p>
          </div>
          <div className="border-t border-[var(--line-strong)] pt-4 sm:col-span-3">
            <p className="text-[13px] font-semibold">Visual questions get visual context</p>
            <p className="mt-2 max-w-[510px] text-[13px] leading-6 text-[var(--ink-muted)]">When a question depends on a plot, diagram, or table, the generation model receives the relevant page image instead of relying on a caption alone.</p>
          </div>
        </div>
      </section>
    </main>
  );
}
