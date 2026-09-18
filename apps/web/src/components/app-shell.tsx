import Link from "next/link";
import { MagnifyingGlass, GitMerge } from "@phosphor-icons/react/dist/ssr";
import { AnalysisModelSelector, ModelSelectionProvider } from "./model-selection";
import { ThemeToggle } from "./theme-toggle";

export function AppShell({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <ModelSelectionProvider>
      <div className="min-h-[100dvh] bg-[var(--canvas)] text-[var(--ink)]">
        <header className="sticky top-0 z-20 border-b border-[var(--line)] bg-[color-mix(in_srgb,var(--canvas)_90%,transparent)] backdrop-blur-xl">
          <div className="mx-auto flex h-[68px] w-full max-w-[1400px] items-center justify-between gap-4 px-5 lg:px-8">
            <Link href="/" className="min-w-0" aria-label="PaperScope home">
              <span className="wordmark display-font">PaperScope</span>
            </Link>

            <nav className="hidden items-center gap-1 text-[13px] text-[var(--ink-muted)] md:flex" aria-label="Primary navigation">
              <Link className="nav-link" href="/search">
                <MagnifyingGlass size={15} />
                Explore
              </Link>
              <Link className="nav-link" href="/compare">
                <GitMerge size={15} />
                Compare
              </Link>
            </nav>

            <div className="flex shrink-0 items-center gap-2">
              <span className="hidden rounded-full border border-[var(--line)] px-3 py-1.5 text-[11px] font-medium text-[var(--ink-muted)] lg:inline-flex">
                Local workspace
              </span>
              <AnalysisModelSelector />
              <ThemeToggle />
            </div>
          </div>
        </header>
        {children}
      </div>
    </ModelSelectionProvider>
  );
}
