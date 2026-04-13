"use client";

import type { RecommendedItem } from "@/lib/types";
import { ProductCard } from "./ProductCard";
import { ProductCardSkeleton } from "./LoadingSkeleton";

interface RecommendationPanelProps {
  recommendations: RecommendedItem[] | null;
  loading: boolean;
  error: string | null;
}

export function RecommendationPanel({
  recommendations,
  loading,
  error,
}: RecommendationPanelProps) {
  if (!recommendations && !loading && !error) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="w-16 h-16 rounded-2xl bg-bg-surface border border-bg-border flex items-center justify-center text-3xl mb-4">
          🎯
        </div>
        <p className="text-text-secondary text-sm max-w-xs">
          Select a persona and click{" "}
          <span className="text-brand-green font-medium">Get Recommendations</span>{" "}
          to see personalized results.
        </p>
      </div>
    );
  }

  return (
    <div>
      {/* Header badge */}
      {recommendations && (
        <div className="flex items-center gap-3 mb-6 p-4 rounded-xl border border-brand-green/30 bg-brand-green/5">
          <div className="w-2.5 h-2.5 rounded-full bg-brand-purple animate-pulse2 shadow-[0_0_8px_rgba(139,92,246,0.6)]" />
          <div>
            <span className="text-sm font-bold text-brand-purple font-display">
              Top Picks for You
            </span>
            <span className="text-text-muted text-sm"> · Personalized book recommendations</span>
          </div>
          <span className="ml-auto tag-pill bg-brand-purple/10 border-brand-purple/30 text-brand-purple text-xs font-mono">
            {recommendations.length} results
          </span>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 text-sm mb-4">
          {error}
        </div>
      )}

      <div className="flex flex-col gap-3">
        {loading &&
          Array.from({ length: 5 }).map((_, i) => (
            <ProductCardSkeleton key={i} />
          ))}

        {!loading &&
          recommendations?.map((item, i) => (
            <ProductCard
              key={item.item_idx}
              item={item}
              rank={i + 1}
              accentColor="text-brand-green"
              accentBg="bg-brand-green/10"
            />
          ))}
      </div>
    </div>
  );
}
