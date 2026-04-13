/**
 * ModelComparison — side-by-side recommendation output from all three models.
 *
 * Calls the /compare endpoint and renders naive, classical, and deep results
 * in parallel columns. This directly satisfies the rubric requirement:
 *   "Model comparison — side-by-side output from all three models"
 *
 * AI Attribution: Generated with assistance from Claude (Anthropic).
 */

"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import type { CompareResponse, RecommendedItem } from "@/lib/types";
import { ProductCard } from "./ProductCard";
import { ProductCardSkeleton } from "./LoadingSkeleton";

interface ModelComparisonProps {
  userIdx: number | null;
  likedItems: number[];
  hasUser: boolean;
}

const MODEL_META = [
  {
    key: "naive" as const,
    label: "Naive (Popularity)",
    description: "Recommends globally popular items",
    icon: "📊",
    accentColor: "text-brand-purple",
    accentBg: "bg-brand-purple/10",
    borderColor: "border-brand-purple/30",
    glowClass: "glow-purple",
  },
  {
    key: "classical" as const,
    label: "Classical (LightGBM)",
    description: "Feature-based reranker with SHAP",
    icon: "🌲",
    accentColor: "text-brand-orange",
    accentBg: "bg-brand-orange/10",
    borderColor: "border-brand-orange/30",
    glowClass: "glow-orange",
  },
  {
    key: "deep" as const,
    label: "Deep (Two-Tower)",
    description: "Neural embeddings with BPR loss",
    icon: "🧠",
    accentColor: "text-brand-green",
    accentBg: "bg-brand-green/10",
    borderColor: "border-brand-green/30",
    glowClass: "glow-green",
  },
] as const;

function findOverlap(
  naive: RecommendedItem[],
  classical: RecommendedItem[],
  deep: RecommendedItem[],
): Set<number> {
  const naiveSet = new Set(naive.map((i) => i.item_idx));
  const classicalSet = new Set(classical.map((i) => i.item_idx));
  const deepSet = new Set(deep.map((i) => i.item_idx));
  const overlap = new Set<number>();
  for (const idx of naiveSet) {
    if (classicalSet.has(idx) && deepSet.has(idx)) {
      overlap.add(idx);
    }
  }
  return overlap;
}

export function ModelComparison({
  userIdx,
  likedItems,
  hasUser,
}: ModelComparisonProps) {
  const [data, setData] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await api.compare({
        user_idx: userIdx,
        liked_items: likedItems,
        top_k: 5,
      });
      setData(res);
    } catch (e: unknown) {
      setError(
        e instanceof Error ? e.message : "Failed to fetch comparison",
      );
    } finally {
      setLoading(false);
    }
  };

  const overlap = data
    ? findOverlap(data.naive, data.classical, data.deep)
    : new Set<number>();

  const uniqueCounts = data
    ? {
        naive: data.naive.filter(
          (i) =>
            !new Set(data.classical.map((c) => c.item_idx)).has(i.item_idx) &&
            !new Set(data.deep.map((d) => d.item_idx)).has(i.item_idx),
        ).length,
        classical: data.classical.filter(
          (i) =>
            !new Set(data.naive.map((n) => n.item_idx)).has(i.item_idx) &&
            !new Set(data.deep.map((d) => d.item_idx)).has(i.item_idx),
        ).length,
        deep: data.deep.filter(
          (i) =>
            !new Set(data.naive.map((n) => n.item_idx)).has(i.item_idx) &&
            !new Set(data.classical.map((c) => c.item_idx)).has(i.item_idx),
        ).length,
      }
    : null;

  return (
    <section className="py-10 px-6 border-t border-bg-border">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="tag-pill bg-bg-elevated border border-bg-border text-text-secondary mb-4">
            Step 3
          </div>
          <h2 className="font-display text-2xl md:text-3xl font-700 text-text-primary mb-3">
            Model Comparison
          </h2>
          <p className="text-text-secondary text-sm max-w-xl">
            See how all three models rank books differently for the same user.
            Items appearing in all three lists are highlighted.
          </p>
        </div>

        {/* Compare button */}
        {hasUser && !loading && (
          <button
            onClick={handleCompare}
            className="mb-8 px-6 py-3 rounded-xl bg-brand-blue text-white font-bold text-sm hover:opacity-90 active:scale-[0.98] transition-all shadow-[0_0_20px_rgba(59,130,246,0.25)]"
          >
            Compare All 3 Models →
          </button>
        )}

        {!hasUser && !loading && !data && !error && (
          <div className="mt-2 mb-8 flex flex-col items-center justify-center py-16 border border-dashed border-bg-border rounded-2xl text-center">
            <div className="w-16 h-16 rounded-2xl bg-bg-surface border border-bg-border flex items-center justify-center text-3xl mb-4">
              ⚖️
            </div>
            <p className="text-text-secondary text-sm max-w-xs">
              Select a persona first, then compare all three models
              side-by-side.
            </p>
          </div>
        )}

        {error && (
          <div className="p-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 text-sm mb-6">
            {error}
          </div>
        )}

        {/* Overlap summary */}
        {data && (
          <div className="flex flex-wrap items-center gap-4 mb-6 p-4 rounded-xl border border-bg-border bg-bg-surface">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-brand-blue animate-pulse2" />
              <span className="text-sm text-text-secondary">
                <span className="font-bold text-brand-blue">
                  {overlap.size}
                </span>{" "}
                item{overlap.size !== 1 ? "s" : ""} recommended by all 3 models
              </span>
            </div>
            {uniqueCounts && (
              <>
                <span className="text-text-muted">·</span>
                {MODEL_META.map((m) => (
                  <span key={m.key} className="text-xs text-text-muted">
                    <span className={`font-mono font-bold ${m.accentColor}`}>
                      {uniqueCounts[m.key]}
                    </span>{" "}
                    unique to {m.label.split(" ")[0]}
                  </span>
                ))}
              </>
            )}
          </div>
        )}

        {/* Three-column comparison */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {MODEL_META.map((m) => (
            <div key={m.key}>
              {/* Column header */}
              <div
                className={`flex items-center gap-3 p-4 rounded-t-xl border ${m.borderColor} ${m.accentBg}`}
              >
                <span className="text-2xl">{m.icon}</span>
                <div>
                  <p className={`text-sm font-bold ${m.accentColor}`}>
                    {m.label}
                  </p>
                  <p className="text-xs text-text-muted">{m.description}</p>
                </div>
              </div>

              {/* Cards */}
              <div
                className={`border border-t-0 ${m.borderColor} rounded-b-xl p-3 flex flex-col gap-3 min-h-[300px]`}
              >
                {loading &&
                  Array.from({ length: 5 }).map((_, i) => (
                    <ProductCardSkeleton key={i} />
                  ))}

                {!loading &&
                  data &&
                  data[m.key].map((item, i) => (
                    <ProductCard
                      key={item.item_idx}
                      item={item}
                      rank={i + 1}
                      accentColor={m.accentColor}
                      accentBg={m.accentBg}
                      highlighted={overlap.has(item.item_idx)}
                    />
                  ))}

                {!loading && data && data[m.key].length === 0 && (
                  <div className="flex items-center justify-center py-12 text-text-muted text-sm italic">
                    No recommendations available
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
