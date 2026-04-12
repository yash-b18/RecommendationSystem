"""
Inference orchestrator for the API layer.

Takes a ModelRegistry and handles recommendation logic for all three models,
including explanation generation. This is the single point of contact between
the FastAPI routes and the ML models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.schemas import RecommendedItem
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _build_item_response(
    item_idx: int,
    score: float,
    explanation: str,
    item_features: pd.DataFrame,
) -> RecommendedItem:
    """Build a RecommendedItem response object for a given item_idx."""
    row = item_features[item_features["item_idx"] == item_idx]
    if row.empty:
        return RecommendedItem(
            item_idx=item_idx,
            title=f"Item #{item_idx}",
            category="Unknown",
            brand="Unknown",
            price=None,
            avg_rating=None,
            num_ratings=None,
            score=round(float(score), 4),
            explanation=explanation,
        )
    r = row.iloc[0]
    return RecommendedItem(
        item_idx=item_idx,
        title=str(r.get("title", f"Item #{item_idx}")),
        category=str(r.get("category", "Unknown")),
        brand=str(r.get("brand", "Unknown")),
        price=float(r["price"]) if pd.notna(r.get("price")) else None,
        avg_rating=round(float(r["item_avg_rating"]), 2) if "item_avg_rating" in r and pd.notna(r.get("item_avg_rating")) else None,
        num_ratings=int(r["item_num_ratings"]) if "item_num_ratings" in r and pd.notna(r.get("item_num_ratings")) else None,
        score=round(float(score), 4),
        explanation=explanation,
    )


class InferenceOrchestrator:
    """
    Routes recommendation requests to the appropriate model and assembles responses.

    Args:
        registry: Loaded ModelRegistry with all model artifacts.
        cfg: Full Config object.
    """

    def __init__(self, registry, cfg) -> None:
        self.registry = registry
        self.cfg = cfg

    def recommend_naive(
        self,
        user_idx: Optional[int],
        top_k: int = 10,
    ) -> list[RecommendedItem]:
        """
        Recommend using the global popularity model.

        Falls back to top-K globally popular items if user_idx is unknown.
        """
        reg = self.registry
        if not reg.naive_available or reg.naive_model is None:
            return []

        uid = user_idx if user_idx is not None else -1

        if uid in (reg.naive_model.user_seen or {}):
            recs = reg.naive_model.recommend(uid, top_k=top_k)
        else:
            # New user: just return global top-K
            recs = list(reg.naive_model.popularity.head(top_k).items())
            recs = [(int(iid), float(score)) for iid, score in recs]

        items = []
        for item_idx, score in recs:
            explanation = "Recommended based on overall popularity — this item is frequently purchased."
            items.append(_build_item_response(item_idx, score, explanation, reg.item_features))
        return items

    def recommend_classical(
        self,
        user_idx: Optional[int],
        top_k: int = 10,
    ) -> list[RecommendedItem]:
        """Recommend using the LightGBM reranker."""
        from src.features.builder import build_feature_matrix
        from src.features.text_features import build_user_text_profiles

        reg = self.registry
        if not reg.classical_available or reg.lgbm_model is None:
            return []

        # Get candidate pool from naive model
        if reg.naive_available and user_idx is not None:
            candidates = reg.naive_model.recommend(
                user_idx, top_k=self.cfg.classical.candidate_pool_size
            )
        else:
            pop = reg.naive_model.popularity if reg.naive_available else None
            if pop is None:
                return []
            candidates = [(int(iid), float(s)) for iid, s in pop.head(self.cfg.classical.candidate_pool_size).items()]

        if not candidates:
            return []

        candidate_items = [iid for iid, _ in candidates]
        rows = pd.DataFrame({
            "user_idx": user_idx if user_idx is not None else 0,
            "item_idx": candidate_items,
            "label": 0,
        })

        user_profiles = pd.DataFrame({"user_idx": [], "user_text_profile": []})
        vectorizer = reg.feature_artifacts.get("vectorizer")

        if vectorizer is None:
            return []

        feat, _ = build_feature_matrix(
            rows,
            reg.user_features if reg.user_features is not None else pd.DataFrame(),
            reg.item_features if reg.item_features is not None else pd.DataFrame(),
            user_profiles,
            vectorizer,
            feature_set="id_metadata_text_history",
        )

        scores = reg.lgbm_model.predict_proba(feat)
        ranked = sorted(zip(candidate_items, scores.tolist()), key=lambda x: x[1], reverse=True)[:top_k]

        items = []
        for item_idx, score in ranked:
            # Build a simple feature-based explanation
            row = feat[feat["item_idx"] == item_idx]
            try:
                from src.explainability.shap_explainer import SHAPExplainer
                explainer = SHAPExplainer(reg.lgbm_model)
                title_col = reg.item_features[reg.item_features["item_idx"] == item_idx]["title"].values
                title = str(title_col[0]) if len(title_col) > 0 else f"Item #{item_idx}"
                explanation = explainer.explain_as_text(row, title)
            except Exception:
                explanation = f"Recommended based on your interaction patterns and item features."
            items.append(_build_item_response(item_idx, score, explanation, reg.item_features))
        return items

    def recommend_deep(
        self,
        user_idx: Optional[int],
        top_k: int = 10,
    ) -> list[RecommendedItem]:
        """Recommend using the Two-Tower neural model."""
        reg = self.registry
        if not reg.deep_available or reg.deep_trainer is None:
            return []

        if user_idx is None or user_idx >= reg.n_users:
            # Cold-start: fall back to naive
            return self.recommend_naive(user_idx=None, top_k=top_k)

        seen = reg.user_seen.get(user_idx, set())
        recs = reg.deep_trainer.recommend(
            user_idx,
            reg.item_embeddings,
            top_k=top_k,
            exclude_items=seen,
        )

        items = []
        for item_idx, score in recs:
            if reg.language_explainer:
                explanation = reg.language_explainer.explain(user_idx, item_idx, score)
            else:
                explanation = "Recommended based on your interaction history."
            items.append(_build_item_response(item_idx, score, explanation, reg.item_features))
        return items

    def recommend(
        self,
        model: str,
        user_idx: Optional[int],
        top_k: int = 10,
    ) -> list[RecommendedItem]:
        """Route recommendation to the specified model."""
        if model == "naive":
            return self.recommend_naive(user_idx, top_k)
        elif model == "classical":
            return self.recommend_classical(user_idx, top_k)
        elif model == "deep":
            return self.recommend_deep(user_idx, top_k)
        else:
            raise ValueError(f"Unknown model: {model!r}. Choose from naive, classical, deep.")

    def get_popular_items(self, n: int = 50) -> list[dict]:
        """Return the top-N globally popular items for the UI item selector."""
        reg = self.registry
        if not reg.naive_available or reg.naive_model is None or reg.item_features is None:
            return []
        top_items = list(reg.naive_model.popularity.head(n).index)
        result = []
        for iid in top_items:
            row = reg.item_features[reg.item_features["item_idx"] == iid]
            if row.empty:
                continue
            r = row.iloc[0]
            result.append({
                "item_idx": int(iid),
                "title": str(r.get("title", f"Item #{iid}")),
                "category": str(r.get("category", "Unknown")),
                "brand": str(r.get("brand", "Unknown")),
                "price": float(r["price"]) if pd.notna(r.get("price")) else None,
                "avg_rating": round(float(r["item_avg_rating"]), 2) if "item_avg_rating" in r else None,
                "num_ratings": int(r["item_num_ratings"]) if "item_num_ratings" in r else None,
            })
        return result
