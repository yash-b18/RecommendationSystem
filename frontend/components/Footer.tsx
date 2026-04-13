export function Footer() {
  return (
    <footer className="relative mt-24 border-t border-bg-border">
      {/* Top gradient line */}
      <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-brand-blue/30 to-transparent" />

      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8">
          {/* Brand */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-brand-blue/20 border border-brand-blue/40 flex items-center justify-center">
                <span className="text-brand-blue text-xs font-bold font-mono">R</span>
              </div>
              <span className="font-display font-700 text-text-primary">DeepReads</span>
            </div>
            <p className="text-text-muted text-sm max-w-xs leading-relaxed">
              Explainable multi-stage e-commerce recommendation system built on
              Amazon Reviews 2023.
            </p>
          </div>

          {/* Links */}
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-6">
              <a
                href="https://github.com/yash-b18/RecommendationSystem"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-text-secondary hover:text-brand-blue transition-colors"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path
                    fillRule="evenodd"
                    d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                    clipRule="evenodd"
                  />
                </svg>
                GitHub
              </a>
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-text-secondary hover:text-brand-blue transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                API Docs
              </a>
            </div>
            <div className="flex flex-wrap gap-2">
              {["Amazon Reviews 2023", "LightGBM", "Two-Tower NN", "SHAP", "FastAPI", "Next.js"].map(
                (tag) => (
                  <span
                    key={tag}
                    className="tag-pill bg-bg-elevated border border-bg-border text-text-muted text-[10px]"
                  >
                    {tag}
                  </span>
                )
              )}
            </div>
          </div>
        </div>

        {/* Bottom row */}
        <div className="mt-8 pt-6 border-t border-bg-border flex flex-col sm:flex-row items-center justify-between gap-3">
          <p className="text-xs text-text-muted font-mono">
            Duke University · AIPI 540 Deep Learning Applications · Spring 2025
          </p>
          <p className="text-xs text-text-muted">
            Built with{" "}
            <span className="text-brand-blue">Next.js</span> ·{" "}
            <span className="text-brand-green">FastAPI</span> ·{" "}
            <span className="text-brand-orange">PyTorch</span>
          </p>
        </div>
      </div>
    </footer>
  );
}
