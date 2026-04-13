"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Persona } from "@/lib/types";
import { LoadingSkeleton } from "./LoadingSkeleton";

const PERSONA_ICONS = ["📚", "🔍", "📖", "🧠", "☕"];
const PERSONA_COLORS = [
  "border-brand-blue/40 hover:border-brand-blue/80",
  "border-brand-green/40 hover:border-brand-green/80",
  "border-brand-orange/40 hover:border-brand-orange/80",
  "border-brand-purple/40 hover:border-brand-purple/80",
  "border-sky-400/40 hover:border-sky-400/80",
];
const ACTIVE_COLORS = [
  "border-brand-blue bg-brand-blue/10 glow-blue",
  "border-brand-green bg-brand-green/10 glow-green",
  "border-brand-orange bg-brand-orange/10 glow-orange",
  "border-brand-purple bg-brand-purple/10 glow-purple",
  "border-sky-400 bg-sky-400/10",
];

interface PersonaSelectorProps {
  onSelect: (persona: Persona | null) => void;
  selectedPersonaId: number | null;
  onBuildOwn: () => void;
}

export function PersonaSelector({
  onSelect,
  selectedPersonaId,
  onBuildOwn,
}: PersonaSelectorProps) {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .personas()
      .then(setPersonas)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <section id="demo" className="py-20 px-6">
      <div className="max-w-6xl mx-auto">
        {/* Section header */}
        <div className="mb-10">
          <div className="tag-pill bg-bg-elevated border border-bg-border text-text-secondary mb-4">
            Step 1
          </div>
          <h2 className="font-display text-3xl md:text-4xl font-700 text-text-primary mb-3">
            Choose a Persona
          </h2>
          <p className="text-text-secondary max-w-lg">
            Select a demo user profile to see personalized recommendations, or
            build your own by picking items manually.
          </p>
        </div>

        {loading && (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <LoadingSkeleton key={i} className="h-36 rounded-xl" />
            ))}
          </div>
        )}

        {error && (
          <div className="p-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 text-sm">
            Could not load personas from API: {error}
          </div>
        )}

        {!loading && !error && (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {personas.map((p, i) => {
              const isActive = selectedPersonaId === p.persona_id;
              return (
                <button
                  key={p.persona_id}
                  onClick={() => onSelect(isActive ? null : p)}
                  className={`relative flex flex-col items-center gap-3 p-5 rounded-xl border transition-all duration-200 text-left group ${
                    isActive
                      ? ACTIVE_COLORS[i % ACTIVE_COLORS.length]
                      : `border-bg-border bg-bg-surface hover:bg-bg-elevated ${
                          PERSONA_COLORS[i % PERSONA_COLORS.length]
                        }`
                  }`}
                >
                  {isActive && (
                    <div className="absolute top-2 right-2 w-4 h-4 rounded-full bg-brand-blue flex items-center justify-center">
                      <svg
                        className="w-2.5 h-2.5 text-white"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </div>
                  )}
                  <span className="text-3xl">{PERSONA_ICONS[i % PERSONA_ICONS.length]}</span>
                  <div className="text-center">
                    <p className="text-sm font-semibold text-text-primary leading-tight">
                      {p.name}
                    </p>
                    <p className="text-xs text-text-secondary mt-1 leading-relaxed">
                      {p.description}
                    </p>
                  </div>
                  <div className="text-xs font-mono text-text-muted">
                    {p.liked_item_idxs.length} items
                  </div>
                </button>
              );
            })}

            {/* Build Your Own card */}
            <button
              onClick={onBuildOwn}
              className="flex flex-col items-center gap-3 p-5 rounded-xl border border-dashed border-bg-border hover:border-brand-blue/50 bg-bg-surface hover:bg-brand-blue/5 transition-all duration-200 group"
            >
              <span className="text-3xl group-hover:scale-110 transition-transform duration-200">
                ✨
              </span>
              <div className="text-center">
                <p className="text-sm font-semibold text-text-primary">
                  Build Your Own
                </p>
                <p className="text-xs text-text-secondary mt-1">
                  Manually pick up to 10 items
                </p>
              </div>
              <div className="text-xs font-mono text-brand-blue">
                Custom →
              </div>
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
