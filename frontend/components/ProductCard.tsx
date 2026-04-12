"use client";

import type { RecommendedItem } from "@/lib/types";
import { ExplanationBadge } from "./ExplanationBadge";

interface ProductCardProps {
  item: RecommendedItem;
  rank: number;
  accentColor: string;
  accentBg: string;
  highlighted?: boolean;
}

function ScoreBar({
  score,
  accentColor,
}: {
  score: number;
  accentColor: string;
}) {
  // Normalise score to [0,1] assuming scores are in [-1,1] for deep model
  // or [0,1] for LightGBM probability
  const pct = Math.min(Math.max((score + 1) / 2, 0), 1) * 100;

  const colorMap: Record<string, string> = {
    "text-brand-green": "bg-brand-green",
    "text-brand-orange": "bg-brand-orange",
    "text-brand-purple": "bg-brand-purple",
    "text-brand-blue": "bg-brand-blue",
  };
  const fillColor = colorMap[accentColor] ?? "bg-brand-blue";

  return (
    <div className="score-bar">
      <div
        className={`score-bar-fill ${fillColor} opacity-80`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function ProductCard({
  item,
  rank,
  accentColor,
  accentBg,
  highlighted = false,
}: ProductCardProps) {
  return (
    <div
      className={`relative flex flex-col gap-3 p-4 rounded-xl border transition-all duration-200 group ${
        highlighted
          ? "border-brand-blue/50 bg-brand-blue/5"
          : "border-bg-border bg-bg-surface hover:border-bg-elevated hover:bg-bg-elevated"
      }`}
    >
      {highlighted && (
        <div className="absolute top-2 right-2">
          <span className="tag-pill bg-brand-blue/20 border border-brand-blue/40 text-brand-blue text-[10px]">
            All 3 models
          </span>
        </div>
      )}

      {/* Header row */}
      <div className="flex items-start gap-3">
        {/* Rank */}
        <div
          className={`shrink-0 w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold font-mono ${accentBg} ${accentColor} border border-current/20`}
        >
          {rank}
        </div>

        {/* Title & meta */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-text-primary leading-tight line-clamp-2 group-hover:text-white transition-colors">
            {item.title}
          </p>
          <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
            <span className="tag-pill bg-bg-elevated border border-bg-border text-text-muted text-[10px]">
              {item.category}
            </span>
            {item.brand && item.brand !== "Unknown" && (
              <span className="tag-pill bg-bg-elevated border border-bg-border text-text-muted text-[10px]">
                {item.brand}
              </span>
            )}
          </div>
        </div>

        {/* Score */}
        <div
          className={`shrink-0 text-right font-mono text-xs ${accentColor}`}
        >
          <div className="font-600">{item.score.toFixed(3)}</div>
          <div className="text-text-muted text-[10px]">score</div>
        </div>
      </div>

      {/* Score bar */}
      <ScoreBar score={item.score} accentColor={accentColor} />

      {/* Stats row */}
      <div className="flex items-center gap-3 text-xs text-text-muted font-mono">
        {item.avg_rating && (
          <span className="flex items-center gap-1">
            <svg
              className="w-3 h-3 text-amber-400"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
            {item.avg_rating.toFixed(1)}
          </span>
        )}
        {item.num_ratings && (
          <span>{item.num_ratings.toLocaleString()} reviews</span>
        )}
        {item.price && (
          <span className="text-brand-green ml-auto">
            ${item.price.toFixed(2)}
          </span>
        )}
      </div>

      {/* Explanation */}
      <ExplanationBadge explanation={item.explanation} />
    </div>
  );
}
