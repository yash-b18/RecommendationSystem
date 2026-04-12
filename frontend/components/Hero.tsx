"use client";

import { useEffect, useRef, useState } from "react";
import { MODEL_CONFIGS } from "@/lib/types";

interface HeroProps {
  onGetStarted: () => void;
}

export function Hero({ onGetStarted }: HeroProps) {
  const [mounted, setMounted] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Animated particle network background
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    type Particle = {
      x: number; y: number;
      vx: number; vy: number;
      radius: number;
    };

    const particles: Particle[] = Array.from({ length: 60 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      radius: Math.random() * 1.5 + 0.5,
    }));

    let raf: number;
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(59,130,246,0.4)";
        ctx.fill();
      });

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(59,130,246,${0.12 * (1 - dist / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      raf = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
      {/* Canvas background */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* Grid pattern */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(59,130,246,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(59,130,246,0.04) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      {/* Radial glow center */}
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse, rgba(59,130,246,0.08) 0%, transparent 70%)",
        }}
      />

      {/* Nav bar */}
      <nav className="absolute top-0 inset-x-0 flex items-center justify-between px-8 py-5 z-10">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-brand-blue/20 border border-brand-blue/40 flex items-center justify-center">
            <span className="text-brand-blue text-xs font-bold font-mono">R</span>
          </div>
          <span className="font-display font-700 text-sm tracking-wide text-text-primary">
            RecoIQ
          </span>
        </div>
        <div className="hidden md:flex items-center gap-6 text-sm text-text-secondary">
          <span className="tag-pill bg-bg-elevated border border-bg-border text-text-secondary">
            Amazon Reviews 2023
          </span>
          <span className="tag-pill bg-brand-blue/10 border border-brand-blue/30 text-brand-blue">
            Video Games
          </span>
        </div>
      </nav>

      {/* Hero content */}
      <div className="relative z-10 flex flex-col items-center text-center px-6 max-w-5xl mx-auto">
        {/* Badge */}
        <div
          className={`mb-8 tag-pill bg-bg-elevated border border-bg-border text-text-secondary text-xs tracking-widest uppercase ${
            mounted ? "animate-fade-in" : "opacity-0"
          }`}
        >
          Duke AIPI 540 · Module 3 Project
        </div>

        {/* Main heading */}
        <h1
          className={`font-display text-5xl md:text-7xl font-800 leading-[1.05] tracking-tight mb-6 ${
            mounted ? "animate-fade-up" : "opacity-0"
          }`}
        >
          <span className="gradient-text">RecoIQ</span>
          <br />
          <span className="text-text-primary">Explainable</span>
          <br />
          <span className="text-text-secondary font-400">
            AI Recommendations
          </span>
        </h1>

        {/* Tagline */}
        <p
          className={`text-text-secondary text-lg md:text-xl max-w-2xl leading-relaxed mb-12 ${
            mounted ? "animate-fade-up delay-200" : "opacity-0"
          }`}
        >
          A multi-stage recommendation system that compares three model tiers —
          from naive popularity to deep neural networks — with transparent,
          feature-level explanations for every recommendation.
        </p>

        {/* Model pills */}
        <div
          className={`flex flex-wrap justify-center gap-3 mb-12 ${
            mounted ? "animate-fade-up delay-300" : "opacity-0"
          }`}
        >
          {MODEL_CONFIGS.map((m) => (
            <div
              key={m.id}
              className={`flex items-center gap-2 px-4 py-2 rounded-full border text-sm font-medium transition-all ${m.bgColor} ${m.borderColor} ${m.color}`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-current opacity-70" />
              {m.fullName}
            </div>
          ))}
        </div>

        {/* CTA */}
        <div
          className={`flex items-center gap-4 ${
            mounted ? "animate-fade-up delay-400" : "opacity-0"
          }`}
        >
          <button
            onClick={onGetStarted}
            className="group relative px-8 py-3.5 rounded-xl bg-brand-blue text-white font-semibold text-sm overflow-hidden transition-all hover:bg-brand-blue/90 hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-brand-blue/30"
          >
            <span className="relative z-10">Try the Demo →</span>
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-500" />
          </button>
          <a
            href="https://github.com/yash-b18/RecommendationSystem"
            target="_blank"
            rel="noopener noreferrer"
            className="px-8 py-3.5 rounded-xl border border-bg-border text-text-secondary font-semibold text-sm hover:border-brand-blue/40 hover:text-text-primary transition-all"
          >
            GitHub →
          </a>
        </div>

        {/* Stats row */}
        <div
          className={`mt-20 grid grid-cols-3 gap-8 md:gap-16 ${
            mounted ? "animate-fade-up delay-500" : "opacity-0"
          }`}
        >
          {[
            { value: "3", label: "Model Tiers" },
            { value: "SHAP", label: "Explainability" },
            { value: "NDCG@K", label: "Evaluation Metric" },
          ].map((stat) => (
            <div key={stat.label} className="flex flex-col items-center gap-1">
              <span className="font-display text-2xl font-700 text-brand-blue">
                {stat.value}
              </span>
              <span className="text-xs text-text-muted uppercase tracking-widest">
                {stat.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-pulse2">
        <span className="text-xs text-text-muted uppercase tracking-widest">Scroll</span>
        <div className="w-px h-8 bg-gradient-to-b from-text-muted to-transparent" />
      </div>
    </section>
  );
}
