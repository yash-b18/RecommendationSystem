import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          base: "#070d1a",
          surface: "#0d1526",
          elevated: "#121e33",
          border: "#1e2d47",
        },
        brand: {
          blue: "#3b82f6",
          "blue-dim": "#1d4ed8",
          green: "#10b981",
          orange: "#f97316",
          purple: "#8b5cf6",
        },
        text: {
          primary: "#f1f5f9",
          secondary: "#94a3b8",
          muted: "#475569",
        },
      },
      fontFamily: {
        display: ["var(--font-syne)", "sans-serif"],
        body: ["var(--font-dm-sans)", "sans-serif"],
        mono: ["var(--font-dm-mono)", "monospace"],
      },
      backgroundImage: {
        "grid-pattern":
          "linear-gradient(rgba(59,130,246,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(59,130,246,0.04) 1px, transparent 1px)",
      },
      backgroundSize: {
        grid: "40px 40px",
      },
      animation: {
        "fade-up": "fadeUp 0.5s ease forwards",
        "fade-in": "fadeIn 0.4s ease forwards",
        pulse2: "pulse2 2s ease-in-out infinite",
        shimmer: "shimmer 1.5s infinite",
        "glow-pulse": "glowPulse 2s ease-in-out infinite",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        pulse2: {
          "0%, 100%": { opacity: "0.6" },
          "50%": { opacity: "1" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        glowPulse: {
          "0%, 100%": { boxShadow: "0 0 8px 0 rgba(59,130,246,0.3)" },
          "50%": { boxShadow: "0 0 20px 4px rgba(59,130,246,0.5)" },
        },
      },
      boxShadow: {
        glow: "0 0 20px rgba(59,130,246,0.25)",
        "glow-green": "0 0 20px rgba(16,185,129,0.25)",
        "glow-orange": "0 0 20px rgba(249,115,22,0.25)",
        "glow-purple": "0 0 20px rgba(139,92,246,0.25)",
        card: "0 1px 3px rgba(0,0,0,0.5), 0 0 0 1px rgba(30,45,71,0.8)",
      },
    },
  },
  plugins: [],
};

export default config;
