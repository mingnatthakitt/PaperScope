"use client";

import { Moon, Sun } from "@phosphor-icons/react";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const saved = window.localStorage.getItem("paperscope-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = saved ? saved === "dark" : prefersDark;
    document.documentElement.dataset.theme = isDark ? "dark" : "light";
    const syncButtonState = window.setTimeout(() => setDark(isDark), 0);
    return () => window.clearTimeout(syncButtonState);
  }, []);

  function toggle() {
    const next = !dark;
    document.documentElement.dataset.theme = next ? "dark" : "light";
    window.localStorage.setItem("paperscope-theme", next ? "dark" : "light");
    setDark(next);
  }

  return (
    <button
      type="button"
      onClick={toggle}
      className="flex h-9 w-9 items-center justify-center rounded-[10px] border border-[var(--line)] text-[var(--ink-muted)] transition hover:border-[var(--line-strong)] hover:text-[var(--ink)] active:scale-[0.96]"
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {dark ? <Sun size={17} /> : <Moon size={17} />}
    </button>
  );
}
