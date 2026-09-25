"use client";

import { CaretDown, Cpu } from "@phosphor-icons/react";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { ANALYSIS_MODELS, ANSWER_MODELS, DEFAULT_ANALYSIS_MODEL, DEFAULT_ANSWER_MODEL, type AnalysisModel, type AnswerModel } from "@/lib/types";

const ANALYSIS_STORAGE_KEY = "paperscope-analysis-model";
const ANSWER_STORAGE_KEY = "paperscope-answer-model";

type ModelSelectionContextValue = {
  model: AnalysisModel;
  setModel: (model: AnalysisModel) => void;
  answerModel: AnswerModel;
  setAnswerModel: (model: AnswerModel) => void;
};

const ModelSelectionContext = createContext<ModelSelectionContextValue | null>(null);

function isAnalysisModel(value: string | null): value is AnalysisModel {
  return ANALYSIS_MODELS.some((option) => option.id === value);
}

function isAnswerModel(value: string | null): value is AnswerModel {
  return ANSWER_MODELS.some((option) => option.id === value);
}

export function ModelSelectionProvider({ children }: Readonly<{ children: React.ReactNode }>) {
  const [model, setModelState] = useState<AnalysisModel>(DEFAULT_ANALYSIS_MODEL);
  const [answerModel, setAnswerModelState] = useState<AnswerModel>(DEFAULT_ANSWER_MODEL);

  useEffect(() => {
    let saved: string | null = null;
    let savedAnswerModel: string | null = null;
    try {
      saved = window.localStorage.getItem(ANALYSIS_STORAGE_KEY);
      savedAnswerModel = window.localStorage.getItem(ANSWER_STORAGE_KEY);
    } catch {
      // Storage may be disabled by browser privacy settings.
    }
    const syncSavedModel = window.setTimeout(() => {
      if (isAnalysisModel(saved)) setModelState(saved);
      if (isAnswerModel(savedAnswerModel)) setAnswerModelState(savedAnswerModel);
    }, 0);
    return () => window.clearTimeout(syncSavedModel);
  }, []);

  const setModel = useCallback((nextModel: AnalysisModel) => {
    setModelState(nextModel);
    try {
      window.localStorage.setItem(ANALYSIS_STORAGE_KEY, nextModel);
    } catch {
      // Keep the choice for this page even when browser storage is unavailable.
    }
  }, []);

  const setAnswerModel = useCallback((nextModel: AnswerModel) => {
    setAnswerModelState(nextModel);
    try {
      window.localStorage.setItem(ANSWER_STORAGE_KEY, nextModel);
    } catch {
      // Keep the choice for this page even when browser storage is unavailable.
    }
  }, []);

  const value = useMemo(
    () => ({ model, setModel, answerModel, setAnswerModel }),
    [model, setModel, answerModel, setAnswerModel],
  );
  return <ModelSelectionContext.Provider value={value}>{children}</ModelSelectionContext.Provider>;
}

export function useAnalysisModel() {
  const context = useContext(ModelSelectionContext);
  if (!context) throw new Error("useAnalysisModel must be used within ModelSelectionProvider");
  return context;
}

export function useAnswerModel() {
  const context = useContext(ModelSelectionContext);
  if (!context) throw new Error("useAnswerModel must be used within ModelSelectionProvider");
  return { answerModel: context.answerModel, setAnswerModel: context.setAnswerModel };
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
        If this model is unavailable, PaperScope tries this fallback chain: {selected.fallback}.
      </span>
    </div>
  );
}

export function AnswerModelSelector() {
  const { answerModel, setAnswerModel } = useAnswerModel();
  return (
    <div className="flex min-w-0 items-center gap-2" data-testid="answer-model-selector">
      <label htmlFor="answer-model" className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--ink-faint)]">Answer model</label>
      <select
        id="answer-model"
        value={answerModel}
        onChange={(event) => setAnswerModel(event.target.value as AnswerModel)}
        className="max-w-[150px] cursor-pointer rounded-full border border-[var(--line)] bg-[var(--canvas)] px-3 py-2 text-[11px] font-semibold text-[var(--ink)] outline-none transition focus:border-[var(--accent)]"
      >
        {ANSWER_MODELS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
      </select>
    </div>
  );
}
