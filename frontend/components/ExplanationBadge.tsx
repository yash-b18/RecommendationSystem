"use client";

import { useState } from "react";

interface ExplanationBadgeProps {
  explanation: string;
}

export function ExplanationBadge({ explanation }: ExplanationBadgeProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs text-text-secondary hover:text-brand-blue transition-colors group"
      >
        <svg
          className="w-3.5 h-3.5 text-brand-blue/60 group-hover:text-brand-blue transition-colors"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
            clipRule="evenodd"
          />
        </svg>
        <span className="font-medium">
          {open ? "Hide" : "Why this?"}{" "}
          <span className="text-text-muted">·</span>{" "}
          <span className="text-brand-blue/80">explanation</span>
        </span>
        <svg
          className={`w-3 h-3 text-text-muted transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {open && (
        <div className="mt-2 p-3 rounded-lg bg-brand-blue/5 border border-brand-blue/20 animate-fade-in">
          <p className="text-xs text-text-secondary italic leading-relaxed font-body">
            <span className="not-italic mr-1">💡</span>
            {explanation}
          </p>
        </div>
      )}
    </div>
  );
}
