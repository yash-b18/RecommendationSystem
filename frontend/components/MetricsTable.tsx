"use client";

const METRICS = [
  {
    name: "Naive Popularity",
    tag: "Best Recall ✦",
    recall: 0.0046,
    ndcg: 0.0022,
    mrr: 0.0020,
    hit: 0.0046,
    color: "text-brand-purple",
    border: "border-brand-purple/30",
    bg: "bg-brand-purple/8",
    dot: "bg-brand-purple",
    winner: true,
  },
  {
    name: "LightGBM",
    tag: "Classical",
    recall: 0.0026,
    ndcg: 0.0012,
    mrr: 0.0010,
    hit: 0.0026,
    color: "text-brand-orange",
    border: "border-brand-orange/20",
    bg: "bg-brand-orange/5",
    dot: "bg-brand-orange",
    winner: false,
  },
  {
    name: "Two-Tower Neural",
    tag: "Best Ranking",
    recall: 0.0024,
    ndcg: 0.0014,
    mrr: 0.0012,
    hit: 0.0024,
    color: "text-brand-green",
    border: "border-brand-green/20",
    bg: "bg-brand-green/5",
    dot: "bg-brand-green",
    winner: false,
  },
];

function fmt(v: number) {
  return v.toFixed(3);
}

function Bar({ value, max, winner }: { value: number; max: number; winner: boolean }) {
  const pct = (value / max) * 100;
  return (
    <div className="w-full bg-bg-elevated rounded-full h-1.5 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-700 ${winner ? "bg-brand-purple shadow-[0_0_6px_rgba(139,92,246,0.5)]" : "bg-bg-border"}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function MetricsTable() {
  const maxRecall = Math.max(...METRICS.map((m) => m.recall));

  return (
    <section className="py-16 px-6 border-t border-bg-border">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-10">
          <div className="tag-pill bg-bg-elevated border border-bg-border text-text-secondary mb-4 inline-flex">
            Model Comparison
          </div>
          <h2 className="font-display text-2xl md:text-3xl font-700 text-text-primary mb-3">
            How We Chose This Model
          </h2>
          <p className="text-text-secondary text-sm max-w-xl">
            All three models were evaluated on a held-out test set using the leave-last-out protocol.
            Naive Popularity leads on recall; Two-Tower outranks LightGBM on NDCG and MRR.
          </p>
        </div>

        {/* Leaderboard */}
        <div className="flex flex-col gap-3">
          {METRICS.map((m, i) => (
            <div
              key={m.name}
              className={`relative rounded-2xl border p-5 transition-all ${m.border} ${m.bg} ${
                m.winner
                  ? "shadow-[0_0_24px_rgba(139,92,246,0.08)] ring-1 ring-brand-purple/20"
                  : m.name === "Two-Tower Neural"
                  ? "shadow-[0_0_24px_rgba(16,185,129,0.06)] ring-1 ring-brand-green/15"
                  : ""
              }`}
            >
              {/* Rank + name */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs text-text-muted w-4">
                    #{i + 1}
                  </span>
                  <div
                    className={`w-2 h-2 rounded-full ${m.dot} ${m.winner ? "shadow-[0_0_8px_rgba(139,92,246,0.8)]" : ""}`}
                  />
                  <span className={`font-display font-bold text-sm ${m.color}`}>
                    {m.name}
                  </span>
                </div>
                <span
                  className={`tag-pill text-xs border ${m.border} ${m.color} font-mono`}
                >
                  {m.tag}
                </span>
              </div>

              {/* Metrics grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "Recall@10", value: m.recall },
                  { label: "NDCG@10", value: m.ndcg },
                  { label: "MRR", value: m.mrr },
                  { label: "Hit@10", value: m.hit },
                ].map((metric) => (
                  <div key={metric.label}>
                    <div className="text-xs text-text-muted font-mono mb-1">
                      {metric.label}
                    </div>
                    <div className={`text-lg font-bold font-mono ${m.color}`}>
                      {fmt(metric.value)}
                    </div>
                    {metric.label === "Recall@10" && (
                      <div className="mt-1.5">
                        <Bar value={metric.value} max={maxRecall} winner={m.winner} />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Footer note */}
        <p className="mt-5 text-xs text-text-muted font-mono">
          Evaluated on held-out test set · leave-last-out protocol · @K = 10 · 5,000 sampled users
        </p>
        <p className="mt-2 text-xs text-text-muted max-w-xl">
          <span className="text-brand-green font-medium">Note:</span> Naive Popularity wins on Recall@10 due to dataset sparsity — most users have fewer than 5 reviews. Two-Tower uses BPR loss with per-user negatives and outranks LightGBM on NDCG@10 (0.0014 vs 0.0012) and MRR (0.0012 vs 0.0010), indicating better ranking quality when it does hit.
        </p>
      </div>
    </section>
  );
}
