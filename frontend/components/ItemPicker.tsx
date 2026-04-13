"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { PopularItem } from "@/lib/types";
import { LoadingSkeleton } from "./LoadingSkeleton";

function StarRating({ rating }: { rating: number }) {
  return (
    <div className="flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map((star) => (
        <svg
          key={star}
          className={`w-3 h-3 ${
            star <= Math.round(rating) ? "text-amber-400" : "text-bg-border"
          }`}
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
        </svg>
      ))}
      <span className="text-xs text-text-muted ml-1">{rating.toFixed(1)}</span>
    </div>
  );
}

interface ItemPickerProps {
  selectedItems: number[];
  onToggleItem: (itemIdx: number) => void;
  onClose: () => void;
}

export function ItemPicker({
  selectedItems,
  onToggleItem,
  onClose,
}: ItemPickerProps) {
  const [items, setItems] = useState<PopularItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("All");

  useEffect(() => {
    api
      .popularItems(80)
      .then(setItems)
      .finally(() => setLoading(false));
  }, []);

  const categories = useMemo(() => {
    const cats = new Set(items.map((i) => i.category));
    return ["All", ...Array.from(cats).sort()];
  }, [items]);

  const filtered = useMemo(
    () =>
      items.filter((item) => {
        const matchCat =
          categoryFilter === "All" || item.category === categoryFilter;
        const matchSearch =
          search === "" ||
          item.title.toLowerCase().includes(search.toLowerCase()) ||
          item.brand.toLowerCase().includes(search.toLowerCase());
        return matchCat && matchSearch;
      }),
    [items, categoryFilter, search]
  );

  const MAX_PICKS = 10;
  const remaining = MAX_PICKS - selectedItems.length;

  return (
    <div className="py-12 px-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="tag-pill bg-bg-elevated border border-bg-border text-text-secondary mb-3">
              Item Picker
            </div>
            <h2 className="font-display text-2xl font-700 text-text-primary">
              Select Items You Like
            </h2>
            <p className="text-text-secondary text-sm mt-1">
              Pick up to {MAX_PICKS} items.{" "}
              <span
                className={remaining === 0 ? "text-red-400" : "text-brand-blue"}
              >
                {remaining} remaining
              </span>
            </p>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-text-secondary border border-bg-border rounded-lg hover:border-brand-blue/40 hover:text-text-primary transition-all"
          >
            ← Back
          </button>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-3 mb-6">
          <input
            type="text"
            placeholder="Search items…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="px-4 py-2 rounded-lg bg-bg-surface border border-bg-border text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-brand-blue/60 transition-colors w-60"
          />
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="px-4 py-2 rounded-lg bg-bg-surface border border-bg-border text-sm text-text-primary focus:outline-none focus:border-brand-blue/60 transition-colors"
          >
            {categories.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div className="flex gap-6">
          {/* Item grid */}
          <div className="flex-1">
            {loading ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {Array.from({ length: 12 }).map((_, i) => (
                  <LoadingSkeleton key={i} className="h-40 rounded-xl" />
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 max-h-[600px] overflow-y-auto pr-2">
                {filtered.map((item) => {
                  const isSelected = selectedItems.includes(item.item_idx);
                  const canSelect = isSelected || remaining > 0;
                  return (
                    <button
                      key={item.item_idx}
                      onClick={() => canSelect && onToggleItem(item.item_idx)}
                      disabled={!canSelect}
                      className={`relative flex flex-col gap-2 p-3 rounded-xl border text-left transition-all duration-200 group ${
                        isSelected
                          ? "border-brand-blue bg-brand-blue/10 glow-blue"
                          : canSelect
                          ? "border-bg-border bg-bg-surface hover:border-brand-blue/40 hover:bg-bg-elevated"
                          : "border-bg-border bg-bg-surface opacity-40 cursor-not-allowed"
                      }`}
                    >
                      {isSelected && (
                        <div className="absolute top-2 right-2 w-5 h-5 rounded-full bg-brand-blue flex items-center justify-center">
                          <svg
                            className="w-3 h-3 text-white"
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
                      <div className="w-10 h-10 rounded-lg bg-bg-elevated border border-bg-border flex items-center justify-center text-xl">
                        🎮
                      </div>
                      <p className="text-xs font-medium text-text-primary leading-tight line-clamp-2">
                        {item.title}
                      </p>
                      <span className="tag-pill bg-bg-elevated border border-bg-border text-text-muted text-[10px]">
                        {item.category}
                      </span>
                      {item.avg_rating && (
                        <StarRating rating={item.avg_rating} />
                      )}
                      {item.price && (
                        <span className="text-xs font-mono text-brand-green">
                          ${item.price.toFixed(2)}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Selected sidebar */}
          <div className="hidden md:flex flex-col w-56 shrink-0">
            <div className="sticky top-4 rounded-xl border border-bg-border bg-bg-surface p-4">
              <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center justify-between">
                Your Picks
                <span className="tag-pill bg-brand-blue/10 border border-brand-blue/30 text-brand-blue text-xs">
                  {selectedItems.length}/{MAX_PICKS}
                </span>
              </h3>
              {selectedItems.length === 0 ? (
                <p className="text-xs text-text-muted italic">
                  No items selected yet
                </p>
              ) : (
                <div className="flex flex-col gap-2">
                  {selectedItems.map((iid) => {
                    const item = items.find((i) => i.item_idx === iid);
                    return (
                      <div
                        key={iid}
                        className="flex items-center gap-2 group"
                      >
                        <button
                          onClick={() => onToggleItem(iid)}
                          className="w-4 h-4 rounded-full bg-brand-blue/20 border border-brand-blue/40 flex items-center justify-center text-brand-blue hover:bg-red-500/20 hover:border-red-500/40 hover:text-red-400 transition-all shrink-0"
                        >
                          <svg
                            className="w-2.5 h-2.5"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                          >
                            <path
                              fillRule="evenodd"
                              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                              clipRule="evenodd"
                            />
                          </svg>
                        </button>
                        <p className="text-xs text-text-secondary line-clamp-1">
                          {item?.title ?? `Item #${iid}`}
                        </p>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
