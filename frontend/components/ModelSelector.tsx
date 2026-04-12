"use client";

import type { ModelType } from "@/lib/types";
import { MODEL_CONFIGS } from "@/lib/types";

interface ModelSelectorProps {
  selected: ModelType;
  onChange: (model: ModelType) => void;
  onRecommend: () => void;
  onCompare: () => void;
  loading: boolean;
  comparing: boolean;
  hasUser: boolean;
}

export function ModelSelector({
  selected,
  onChange,
  onRecommend,
  onCompare,
  loading,
  comparing,
  hasUser,
}: ModelSelectorProps) {
  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 flex-wrap">
      {/* Model toggle */}
      <div className="flex items-center gap-1 p-1 rounded-xl bg-bg-surface border border-bg-border">
        {MODEL_CONFIGS.map((m) => {
          const isActive = selected === m.id;
          return (
            <button
              key={m.id}
              onClick={() => onChange(m.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                isActive
                  ? `${m.bgColor} ${m.color} border ${m.borderColor}`
                  : "text-text-secondary hover:text-text-primary"
              }`}
              title={m.description}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isActive ? "bg-current" : "bg-text-muted"
                }`}
              />
              {m.fullName}
            </button>
          );
        })}
      </div>

      {/* Action buttons */}
      <div className="flex gap-2">
        <button
          onClick={onRecommend}
          disabled={loading || !hasUser}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-brand-blue text-white text-sm font-semibold hover:bg-brand-blue/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-brand-blue/20"
        >
          {loading ? (
            <>
              <svg
                className="w-4 h-4 animate-spin"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Loading…
            </>
          ) : (
            <>
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
              Get Recommendations
            </>
          )}
        </button>

        <button
          onClick={onCompare}
          disabled={comparing || !hasUser}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-bg-border text-text-secondary text-sm font-semibold hover:border-brand-blue/50 hover:text-text-primary disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {comparing ? (
            <>
              <svg
                className="w-4 h-4 animate-spin"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Comparing…
            </>
          ) : (
            <>
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
              Compare All
            </>
          )}
        </button>
      </div>

      {!hasUser && (
        <p className="text-xs text-text-muted italic">
          ↑ Select a persona or pick items first
        </p>
      )}
    </div>
  );
}
