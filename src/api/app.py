"""
FastAPI application for the recommendation system.

Endpoints:
  GET  /health          — health check + model status
  POST /recommend       — get recommendations from a single model
  POST /compare         — get recommendations from all three models
  GET  /items/popular   — popular items for the UI item picker
  GET  /items/{item_idx} — single item metadata
  GET  /personas        — demo user personas for the UI

Run with:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.recommender import InferenceOrchestrator
from src.api.schemas import (
    CompareRequest,
    CompareResponse,
    HealthResponse,
    Persona,
    PopularItem,
    RecommendRequest,
    RecommendResponse,
    RecommendedItem,
)
from src.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App state (populated at startup)
# ---------------------------------------------------------------------------

class AppState:
    registry = None
    orchestrator: Optional[InferenceOrchestrator] = None
    cfg = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, release on shutdown."""
    logger.info("Loading model artifacts…")
    state.cfg = get_config()

    # Import here to avoid circular imports at module level
    from scripts.serve_utils import ModelRegistry
    state.registry = ModelRegistry.load(state.cfg)
    state.orchestrator = InferenceOrchestrator(state.registry, state.cfg)
    logger.info("API ready.")
    yield
    logger.info("Shutting down API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Explainable E-Commerce Recommendation API",
    description="Multi-stage recommendation system with SHAP and language explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server and production origins
cors_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://*.vercel.app",
]
env_origin = os.getenv("FRONTEND_URL")
if env_origin:
    cors_origins.append(env_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Return API health and model availability."""
    reg = state.registry
    return HealthResponse(
        status="ok",
        models_loaded={
            "naive": getattr(reg, "naive_available", False),
            "classical": getattr(reg, "classical_available", False),
            "deep": getattr(reg, "deep_available", False),
        },
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Return top-K recommendations for a user using the specified model.

    If user_idx is None, the request is treated as a cold-start (new user)
    and the naive popularity model is used as a fallback for all models.
    """
    if state.orchestrator is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    items = state.orchestrator.recommend(
        model=request.model,
        user_idx=request.user_idx,
        top_k=request.top_k,
    )
    return RecommendResponse(
        model_used=request.model,
        user_idx=request.user_idx,
        recommendations=items,
    )


@app.post("/compare", response_model=CompareResponse, tags=["Recommendations"])
def compare(request: CompareRequest) -> CompareResponse:
    """
    Return top-K recommendations from all three models for side-by-side comparison.
    """
    if state.orchestrator is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    orch = state.orchestrator
    return CompareResponse(
        user_idx=request.user_idx,
        naive=orch.recommend("naive", request.user_idx, request.top_k),
        classical=orch.recommend("classical", request.user_idx, request.top_k),
        deep=orch.recommend("deep", request.user_idx, request.top_k),
    )


@app.get("/items/popular", response_model=list[PopularItem], tags=["Items"])
def popular_items(n: int = 50) -> list[PopularItem]:
    """Return the top-N globally popular items for the UI item selection widget."""
    if state.orchestrator is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    raw = state.orchestrator.get_popular_items(n=min(n, 200))
    return [PopularItem(**r) for r in raw]


@app.get("/items/{item_idx}", response_model=PopularItem, tags=["Items"])
def get_item(item_idx: int) -> PopularItem:
    """Return metadata for a single item by its encoded index."""
    if state.registry is None or state.registry.item_features is None:
        raise HTTPException(status_code=503, detail="Item features not loaded.")
    row = state.registry.item_features[
        state.registry.item_features["item_idx"] == item_idx
    ]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Item {item_idx} not found.")
    import pandas as pd
    r = row.iloc[0]
    return PopularItem(
        item_idx=item_idx,
        title=str(r.get("title", f"Item #{item_idx}")),
        category=str(r.get("category", "Unknown")),
        brand=str(r.get("brand", "Unknown")),
        price=float(r["price"]) if pd.notna(r.get("price")) else None,
        avg_rating=round(float(r["item_avg_rating"]), 2) if "item_avg_rating" in r else None,
        num_ratings=int(r["item_num_ratings"]) if "item_num_ratings" in r else None,
    )


@app.get("/personas", response_model=list[Persona], tags=["Demo"])
def personas() -> list[Persona]:
    """
    Return a list of demo user personas for the frontend persona selector.

    Personas are seeded from real users in the training set who have at least
    10 positive interactions, making them representative for demos.
    """
    if state.registry is None or state.registry.train_df is None:
        raise HTTPException(status_code=503, detail="Training data not loaded.")

    import pandas as pd

    train = state.registry.train_df
    pos_threshold = state.cfg.data.positive_rating_threshold

    # Find users with >= 10 positive ratings
    pos_counts = (
        train[train.get("label", train["rating"]) >= pos_threshold]
        .groupby("user_idx")["item_idx"]
        .count()
    )
    eligible = pos_counts[pos_counts >= 10].index.tolist()[:20]

    persona_names = [
        ("Action Gamer", "Loves fast-paced action and shooter games"),
        ("RPG Enthusiast", "Prefers deep story-driven role-playing games"),
        ("Sports Fan", "Follows sports simulation franchises"),
        ("Strategy Thinker", "Enjoys real-time and turn-based strategy"),
        ("Casual Player", "Plays a mix of genres and indie games"),
    ]

    result = []
    for i, uid in enumerate(eligible[:5]):
        liked = train[
            (train["user_idx"] == uid) &
            (train.get("label", train["rating"]) >= pos_threshold)
        ]["item_idx"].tolist()[:6]

        name, desc = persona_names[i % len(persona_names)]
        result.append(Persona(
            persona_id=i + 1,
            name=name,
            description=desc,
            liked_item_idxs=liked,
            user_idx=int(uid),
        ))

    return result
