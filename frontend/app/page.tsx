"use client";

import { useCallback, useRef, useState } from "react";
import { api } from "@/lib/api";
import type {
  CompareResponse,
  ModelType,
  Persona,
  RecommendedItem,
} from "@/lib/types";
import { CompareView } from "@/components/CompareView";
import { Footer } from "@/components/Footer";
import { Hero } from "@/components/Hero";
import { ItemPicker } from "@/components/ItemPicker";
import { ModelSelector } from "@/components/ModelSelector";
import { PersonaSelector } from "@/components/PersonaSelector";
import { RecommendationPanel } from "@/components/RecommendationPanel";

type ViewMode = "persona" | "picker" | "results";

export default function HomePage() {
  const demoRef = useRef<HTMLDivElement>(null);

  // User state
  const [selectedPersona, setSelectedPersona] = useState<Persona | null>(null);
  const [customItems, setCustomItems] = useState<number[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>("persona");

  // Model state
  const [selectedModel, setSelectedModel] = useState<ModelType>("deep");

  // Recommendation state
  const [recommendations, setRecommendations] = useState<RecommendedItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);

  // Comparison state
  const [compareData, setCompareData] = useState<CompareResponse | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

  // Active view: "single" or "compare"
  const [activeView, setActiveView] = useState<"single" | "compare">("single");

  // Derived: current user_idx and liked_items
  const userIdx = selectedPersona?.user_idx ?? null;
  const likedItems = selectedPersona
    ? selectedPersona.liked_item_idxs
    : customItems;
  const hasUser = likedItems.length > 0 || userIdx !== null;

  const scrollToDemo = useCallback(() => {
    demoRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  // Persona selection
  const handlePersonaSelect = (persona: Persona | null) => {
    setSelectedPersona(persona);
    setCustomItems([]);
    setRecommendations(null);
    setCompareData(null);
    setViewMode("persona");
  };

  // Item toggle for custom builder
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

  // Get recommendations from single model
  const handleRecommend = async () => {
    setLoading(true);
    setRecError(null);
    setRecommendations(null);
    setActiveView("single");
    try {
      const res = await api.recommend({
        user_idx: userIdx,
        liked_items: likedItems,
        model: selectedModel,
        top_k: 10,
      });
      setRecommendations(res.recommendations);
    } catch (e: unknown) {
      setRecError(
        e instanceof Error ? e.message : "Failed to fetch recommendations"
      );
    } finally {
      setLoading(false);
    }
  };

  // Compare all models
  const handleCompare = async () => {
    setComparing(true);
    setCompareError(null);
    setCompareData(null);
    setActiveView("compare");
    try {
      const res = await api.compare({
        user_idx: userIdx,
        liked_items: likedItems,
        top_k: 10,
      });
      setCompareData(res);
    } catch (e: unknown) {
      setCompareError(
        e instanceof Error ? e.message : "Failed to fetch comparison"
      );
    } finally {
      setComparing(false);
    }
  };

  return (
    <main className="min-h-screen">
      {/* Hero */}
      <Hero onGetStarted={scrollToDemo} />

      {/* Demo section */}
      <div ref={demoRef} className="relative">
        {/* Subtle section divider */}
        <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-brand-blue/20 to-transparent" />

        {/* Persona selector */}
        <PersonaSelector
          onSelect={handlePersonaSelect}
          selectedPersonaId={selectedPersona?.persona_id ?? null}
          onBuildOwn={() => {
            setViewMode("picker");
            setSelectedPersona(null);
          }}
        />

        {/* Item picker (shown when "Build Your Own" is clicked) */}
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
                    <span className="font-medium text-text-primary">
                      {selectedPersona.name}
                    </span>
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
                    setCompareData(null);
                  }}
                  className="ml-auto text-xs text-text-muted hover:text-text-secondary transition-colors"
                >
                  Clear ×
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Model selector + results */}
        <section className="py-10 px-6 border-t border-bg-border">
          <div className="max-w-6xl mx-auto">
            {/* Step label */}
            <div className="mb-6">
              <div className="tag-pill bg-bg-elevated border border-bg-border text-text-secondary mb-4">
                Step 2
              </div>
              <h2 className="font-display text-2xl md:text-3xl font-700 text-text-primary mb-3">
                Choose a Model & Get Recommendations
              </h2>
            </div>

            {/* Model selector */}
            <div className="mb-8">
              <ModelSelector
                selected={selectedModel}
                onChange={setSelectedModel}
                onRecommend={handleRecommend}
                onCompare={handleCompare}
                loading={loading}
                comparing={comparing}
                hasUser={hasUser}
              />
            </div>

            {/* Results area */}
            {(activeView === "single" || loading || recError) &&
              activeView !== "compare" && (
                <RecommendationPanel
                  recommendations={recommendations}
                  loading={loading}
                  error={recError}
                  modelType={selectedModel}
                />
              )}

            {(activeView === "compare" || comparing || compareError) &&
              activeView !== "single" && (
                <CompareView
                  data={compareData}
                  loading={comparing}
                  error={compareError}
                />
              )}

            {!loading && !comparing && !recommendations && !compareData && !recError && !compareError && (
              <div className="mt-8 flex flex-col items-center justify-center py-20 border border-dashed border-bg-border rounded-2xl text-center">
                <div className="w-16 h-16 rounded-2xl bg-bg-surface border border-bg-border flex items-center justify-center text-3xl mb-4">
                  🎯
                </div>
                <p className="text-text-secondary text-sm max-w-xs">
                  {hasUser
                    ? 'Click "Get Recommendations" or "Compare All" to see results.'
                    : "Select a persona or pick items to get started."}
                </p>
              </div>
            )}
          </div>
        </section>
      </div>

      <Footer />
    </main>
  );
}
