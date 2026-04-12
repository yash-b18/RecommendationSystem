"use client";

import type { ModelType, RecommendedItem } from "@/lib/types";
import { MODEL_CONFIGS } from "@/lib/types";
import { ProductCard } from "./ProductCard";
import { ProductCardSkeleton } from "./LoadingSkeleton";

interface RecommendationPanelProps {
  recommendations: RecommendedItem[] | null;
  loading: boolean;
  error: string | null;
  modelType: ModelType;
}

export function RecommendationPanel({
  recommendations,
  loading,
  error,
  modelType,
}: RecommendationPanelProps) {
  const modelCfg = MODEL_CONFIGS.find((m) => m.id === modelType)!;

  if (!recommendations && !loading && !error) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="w-16 h-16 rounded-2xl bg-bg-surface border border-bg-border flex items-center justify-center text-3xl mb-4">
          🎯
        </div>
        <p className="text-text-secondary text-sm max-w-xs">
          Select a persona and click{" "}
          <span className="text-brand-blue font-medium">Get Recommendations</span>{" "}
          to see personalized results.
        </p>
      </div>
    );
  }

  return (
    <div>
      {/* Model header */}
      {recommendations && (
        <div
          className={`flex items-center gap-3 mb-6 p-4 rounded-xl border ${modelCfg.borderColor} ${modelCfg.bgColor}`}
        >
          <div
            className={`w-2.5 h-2.5 rounded-full ${
              modelCfg.color.replace("text-", "bg-")
            } animate-pulse2`}
          />
          <div>
            <span
              className={`text-sm font-bold ${modelCfg.color} font-display`}
            >
              {modelCfg.fullName}
            </span>
            <span className="text-text-muted text-sm"> · {modelCfg.description}</span>
          </div>
          <span className={`ml-auto tag-pill ${modelCfg.bgColor} ${modelCfg.borderColor} ${modelCfg.color} text-xs font-mono`}>
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
              accentColor={modelCfg.color}
              accentBg={modelCfg.bgColor}
            />
          ))}
      </div>
    </div>
  );
}
