"""
Pydantic schemas for the FastAPI recommendation API.

Defines request and response models for all endpoints.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Request body for the /recommend endpoint."""
    user_idx: Optional[int] = Field(
        None,
        description="Encoded user index. If None, recommendations are based on liked_items only.",
    )
    liked_items: List[int] = Field(
        default_factory=list,
        description="List of item indices the user has expressed interest in.",
    )
    model: Literal["naive", "classical", "deep"] = Field(
        "deep",
        description="Which model to use for recommendations.",
    )
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return.")


class CompareRequest(BaseModel):
    """Request body for the /compare endpoint (all three models)."""
    user_idx: Optional[int] = None
    liked_items: List[int] = Field(default_factory=list)
    top_k: int = Field(10, ge=1, le=20)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class RecommendedItem(BaseModel):
    """A single recommended product."""
    item_idx: int
    title: str
    category: str
    brand: str
    price: Optional[float]
    avg_rating: Optional[float]
    num_ratings: Optional[int]
    score: float = Field(description="Model relevance score (higher = more relevant).")
    explanation: str = Field(description="Natural language explanation for this recommendation.")


class RecommendResponse(BaseModel):
    """Response from /recommend."""
    model_used: str
    user_idx: Optional[int]
    recommendations: List[RecommendedItem]


class CompareResponse(BaseModel):
    """Response from /compare — all three models side by side."""
    user_idx: Optional[int]
    naive: List[RecommendedItem]
    classical: List[RecommendedItem]
    deep: List[RecommendedItem]


class PopularItem(BaseModel):
    """A popular item for the persona/item selection UI."""
    item_idx: int
    title: str
    category: str
    brand: str
    price: Optional[float]
    avg_rating: Optional[float]
    num_ratings: Optional[int]


class Persona(BaseModel):
    """A demo user persona with pre-selected liked items."""
    persona_id: int
    name: str
    description: str
    liked_item_idxs: List[int]
    user_idx: Optional[int]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: dict[str, bool]
