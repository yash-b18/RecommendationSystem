"use client";

import type { CompareResponse } from "@/lib/types";
import { MODEL_CONFIGS } from "@/lib/types";
import { ProductCard } from "./ProductCard";
import { ProductCardSkeleton } from "./LoadingSkeleton";

interface CompareViewProps {
  data: CompareResponse | null;
  loading: boolean;
  error: string | null;
}

export function CompareView({ data, loading, error }: CompareViewProps) {
  // Items that appear in all three model results
  const sharedItems = data
    ? new Set(
        data.naive
          .map((i) => i.item_idx)
          .filter(
            (id) =>
              data.classical.some((i) => i.item_idx === id) &&
              data.deep.some((i) => i.item_idx === id)
          )
      )
    : new Set<number>();

  const columns = data
    ? [
        { key: "naive" as const, items: data.naive },
        { key: "classical" as const, items: data.classical },
        { key: "deep" as const, items: data.deep },
      ]
    : null;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <h3 className="font-display text-lg font-700 text-text-primary">
          Model Comparison
        </h3>
        {sharedItems.size > 0 && (
          <span className="tag-pill bg-brand-blue/10 border border-brand-blue/30 text-brand-blue text-xs">
            {sharedItems.size} item{sharedItems.size > 1 ? "s" : ""} in all 3
          </span>
        )}
      </div>

      {error && (
        <div className="p-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 text-sm mb-6">
          {error}
        </div>
      )}

      {/* Legend */}
      {data && (
        <div className="flex items-center gap-4 mb-6">
          <div className="flex items-center gap-1.5 text-xs text-text-secondary">
            <div className="w-3 h-3 rounded border border-brand-blue/50 bg-brand-blue/10" />
            Appears in all 3 models
          </div>
        </div>
      )}

      {/* Three-column grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {MODEL_CONFIGS.map((m, colIdx) => {
          const colItems = columns?.find((c) => c.key === m.id)?.items;

          return (
            <div key={m.id}>
              {/* Column header */}
              <div
                className={`flex items-center gap-2 mb-4 pb-3 border-b ${m.borderColor}`}
              >
                <div
                  className={`w-2 h-2 rounded-full animate-pulse2 ${m.color.replace(
                    "text-",
                    "bg-"
                  )}`}
                />
                <span className={`font-display font-700 text-sm ${m.color}`}>
                  {m.fullName}
                </span>
                <span className="text-text-muted text-xs ml-auto font-mono">
                  {colItems?.length ?? 0} results
                </span>
              </div>

              {/* Items */}
              <div className="flex flex-col gap-3">
                {loading &&
                  Array.from({ length: 5 }).map((_, i) => (
                    <ProductCardSkeleton key={i} />
                  ))}

                {!loading &&
                  colItems?.map((item, i) => (
                    <ProductCard
                      key={item.item_idx}
                      item={item}
                      rank={i + 1}
                      accentColor={m.color}
                      accentBg={m.bgColor}
                      highlighted={sharedItems.has(item.item_idx)}
                    />
                  ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
