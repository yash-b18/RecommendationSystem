/**
 * Home page for the DeepReads recommendation demo.
 *
 * AI Attribution: Generated with assistance from Claude (Anthropic).
 */

"use client";

import { useCallback, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { Persona, RecommendedItem } from "@/lib/types";
import { Footer } from "@/components/Footer";
import { Hero } from "@/components/Hero";
import { ItemPicker } from "@/components/ItemPicker";
import { ModelComparison } from "@/components/ModelComparison";
import { PersonaSelector } from "@/components/PersonaSelector";
import { RecommendationPanel } from "@/components/RecommendationPanel";

type ViewMode = "persona" | "picker";

export default function HomePage() {
  const demoRef = useRef<HTMLDivElement>(null);

  // User state
  const [selectedPersona, setSelectedPersona] = useState<Persona | null>(null);
  const [customItems, setCustomItems] = useState<number[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>("persona");

  // Recommendation state
  const [recommendations, setRecommendations] = useState<RecommendedItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);

  // Derived state
  const userIdx = selectedPersona?.user_idx ?? null;
  const likedItems = selectedPersona ? selectedPersona.liked_item_idxs : customItems;
  const hasUser = likedItems.length > 0 || userIdx !== null;

  const scrollToDemo = useCallback(() => {
    demoRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const handlePersonaSelect = (persona: Persona | null) => {
    setSelectedPersona(persona);
    setCustomItems([]);
    setRecommendations(null);
    setViewMode("persona");
  };

  const handleToggleItem = (itemIdx: number) => {
    setCustomItems((prev) =>
      prev.includes(itemIdx)
        ? prev.filter((i) => i !== itemIdx)
        : prev.length < 10
        ? [...prev, itemIdx]
        : prev
    );
    setSelectedPersona(null);
  };

  const handleRecommend = async () => {
    setLoading(true);
    setRecError(null);
    setRecommendations(null);
    try {
      const res = await api.recommend({
        user_idx: userIdx,
        liked_items: likedItems,
        model: "deep",
        top_k: 10,
      });
      setRecommendations(res.recommendations);
    } catch (e: unknown) {
      setRecError(e instanceof Error ? e.message : "Failed to fetch recommendations");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen">
      {/* Hero */}
      <Hero onGetStarted={scrollToDemo} />

      {/* Demo section */}
      <div ref={demoRef} className="relative">
        <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-brand-blue/20 to-transparent" />

        {/* Step 1 — Persona selector */}
        <PersonaSelector
          onSelect={handlePersonaSelect}
          selectedPersonaId={selectedPersona?.persona_id ?? null}
          onBuildOwn={() => {
            setViewMode("picker");
            setSelectedPersona(null);
          }}
        />

        {/* Item picker */}
        {viewMode === "picker" && (
          <div className="border-t border-bg-border">
            <ItemPicker
              selectedItems={customItems}
              onToggleItem={handleToggleItem}
              onClose={() => setViewMode("persona")}
            />
          </div>
        )}

        {/* Selected context summary */}
        {hasUser && (
          <div className="px-6 pb-2">
            <div className="max-w-6xl mx-auto">
              <div className="flex items-center gap-3 p-3 rounded-xl bg-bg-surface border border-bg-border text-sm">
                {selectedPersona ? (
                  <>
                    <span className="text-text-secondary">Persona:</span>
                    <span className="font-medium text-text-primary">{selectedPersona.name}</span>
                    <span className="text-text-muted">·</span>
                    <span className="text-text-secondary">
                      {selectedPersona.liked_item_idxs.length} liked items
                    </span>
                  </>
                ) : (
                  <>
                    <span className="text-text-secondary">Custom picks:</span>
                    <span className="font-medium text-brand-blue">
                      {customItems.length} item{customItems.length !== 1 ? "s" : ""} selected
                    </span>
                  </>
                )}
                <button
                  onClick={() => {
                    setSelectedPersona(null);
                    setCustomItems([]);
                    setRecommendations(null);
                  }}
                  className="ml-auto text-xs text-text-muted hover:text-text-secondary transition-colors"
                >
                  Clear ×
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 2 — Recommendations */}
        <section className="py-10 px-6 border-t border-bg-border">
          <div className="max-w-6xl mx-auto">
            <div className="mb-6">
              <div className="tag-pill bg-bg-elevated border border-bg-border text-text-secondary mb-4">
                Step 2
              </div>
              <h2 className="font-display text-2xl md:text-3xl font-700 text-text-primary mb-3">
                Your Recommendations
              </h2>
              <p className="text-text-secondary text-sm">
                Powered by our best-performing Two-Tower neural network.
              </p>
            </div>

            {/* Get recommendations button */}
            {hasUser && !loading && (
              <button
                onClick={handleRecommend}
                className="mb-8 px-6 py-3 rounded-xl bg-brand-green text-bg-base font-bold text-sm hover:opacity-90 active:scale-[0.98] transition-all shadow-[0_0_20px_rgba(16,185,129,0.25)]"
              >
                Get Recommendations →
              </button>
            )}

            {!hasUser && !loading && !recommendations && !recError && (
              <div className="mt-2 mb-8 flex flex-col items-center justify-center py-20 border border-dashed border-bg-border rounded-2xl text-center">
                <div className="w-16 h-16 rounded-2xl bg-bg-surface border border-bg-border flex items-center justify-center text-3xl mb-4">
                  🎯
                </div>
                <p className="text-text-secondary text-sm max-w-xs">
                  Select a persona above to get personalized book recommendations.
                </p>
              </div>
            )}

            <RecommendationPanel
              recommendations={recommendations}
              loading={loading}
              error={recError}
            />
          </div>
        </section>

        {/* Step 3 — Model Comparison (side-by-side output from all three models) */}
        <ModelComparison
          userIdx={userIdx}
          likedItems={likedItems}
          hasUser={hasUser}
        />

      </div>

      <Footer />
    </main>
  );
}
