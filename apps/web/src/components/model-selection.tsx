"use client";

import { CaretDown, Cpu } from "@phosphor-icons/react";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { ANALYSIS_MODELS, DEFAULT_ANALYSIS_MODEL, type AnalysisModel } from "@/lib/types";

const STORAGE_KEY = "paperscope-analysis-model";

type ModelSelectionContextValue = {
  model: AnalysisModel;
  setModel: (model: AnalysisModel) => void;
};

const ModelSelectionContext = createContext<ModelSelectionContextValue | null>(null);

function isAnalysisModel(value: string | null): value is AnalysisModel {
  return ANALYSIS_MODELS.some((option) => option.id === value);
}

export function ModelSelectionProvider({ children }: Readonly<{ children: React.ReactNode }>) {
  const [model, setModelState] = useState<AnalysisModel>(DEFAULT_ANALYSIS_MODEL);

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    const syncSavedModel = window.setTimeout(() => {
      if (isAnalysisModel(saved)) setModelState(saved);
    }, 0);
    return () => window.clearTimeout(syncSavedModel);
  }, []);

  const setModel = useCallback((nextModel: AnalysisModel) => {
    setModelState(nextModel);
    window.localStorage.setItem(STORAGE_KEY, nextModel);
  }, []);

  const value = useMemo(() => ({ model, setModel }), [model, setModel]);
  return <ModelSelectionContext.Provider value={value}>{children}</ModelSelectionContext.Provider>;
}

export function useAnalysisModel() {
  const context = useContext(ModelSelectionContext);
  if (!context) throw new Error("useAnalysisModel must be used within ModelSelectionProvider");
  return context;
}

export function AnalysisModelSelector() {
  const { model, setModel } = useAnalysisModel();
  const selected = ANALYSIS_MODELS.find((option) => option.id === model) ?? ANALYSIS_MODELS[0];

  return (
    <div className="flex items-center gap-2" data-testid="analysis-model-selector">
      <Cpu size={15} className="hidden shrink-0 text-[var(--accent)] sm:block" aria-hidden="true" />
      <div className="flex min-w-0 items-center gap-1.5">
        <label htmlFor="analysis-model" className="sr-only">Main paper indexing model</label>
        <select
          id="analysis-model"
          value={model}
          onChange={(event) => setModel(event.target.value as AnalysisModel)}
          className="max-w-[102px] cursor-pointer appearance-none truncate bg-transparent py-2 pr-4 text-[11px] font-semibold tracking-[-0.01em] text-[var(--ink)] outline-none sm:max-w-none sm:text-[12px]"
          aria-describedby="analysis-model-fallback"
          title={`${selected.label}; fallback ${selected.fallback}`}
        >
          {ANALYSIS_MODELS.map((option, index) => (
            <option key={option.id} value={option.id}>
              {option.label}{index === 0 ? " · default" : ""}
            </option>
          ))}
        </select>
        <CaretDown size={12} className="pointer-events-none -ml-3 shrink-0 text-[var(--ink-faint)]" aria-hidden="true" />
      </div>
      <span id="analysis-model-fallback" className="sr-only">
        If this model is unavailable, PaperScope falls back to Gemini Flash {selected.fallback}.
      </span>
    </div>
  );
}
