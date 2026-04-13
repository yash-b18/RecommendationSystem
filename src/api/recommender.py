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
    title_val = r.get("title")
    category_val = r.get("category")
    brand_val = r.get("brand")
    return RecommendedItem(
        item_idx=item_idx,
        title=str(title_val) if pd.notna(title_val) and str(title_val) != "nan" else f"Item #{item_idx}",
        category=str(category_val) if pd.notna(category_val) and str(category_val) != "nan" else "Unknown",
        brand=str(brand_val) if pd.notna(brand_val) and str(brand_val) != "nan" else "Unknown",
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
        # Pre-compute set of item_idxs that have no title — exclude from all recommendations
        if registry.item_features is not None:
            df = registry.item_features
            untitled_mask = df["title"].isna() | (df["title"].astype(str).str.strip() == "") | (df["title"].astype(str) == "nan")
            self._untitled_items: set[int] = set(df.loc[untitled_mask, "item_idx"].tolist())
        else:
            self._untitled_items = set()
        logger.info("Untitled items excluded from recommendations: %d", len(self._untitled_items))

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
            recs = reg.naive_model.recommend(uid, top_k=top_k + len(self._untitled_items))
            recs = [(iid, s) for iid, s in recs if iid not in self._untitled_items][:top_k]
        else:
            # New user: just return global top-K from titled items only
            pop_titled = reg.naive_model.popularity[
                ~reg.naive_model.popularity.index.isin(self._untitled_items)
            ]
            recs = list(pop_titled.head(top_k).items())
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

        # Get candidate pool from naive model (titled items only)
        if reg.naive_available and user_idx is not None:
            raw_candidates = reg.naive_model.recommend(
                user_idx, top_k=self.cfg.classical.candidate_pool_size + len(self._untitled_items)
            )
            candidates = [(iid, s) for iid, s in raw_candidates if iid not in self._untitled_items][:self.cfg.classical.candidate_pool_size]
        else:
            pop = reg.naive_model.popularity if reg.naive_available else None
            if pop is None:
                return []
            pop_titled = pop[~pop.index.isin(self._untitled_items)]
            candidates = [(int(iid), float(s)) for iid, s in pop_titled.head(self.cfg.classical.candidate_pool_size).items()]

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
        # feat rows are in same order as candidate_items; build index map
        item_to_feat_idx = {iid: i for i, iid in enumerate(candidate_items)}
        ranked = sorted(zip(candidate_items, scores.tolist()), key=lambda x: x[1], reverse=True)[:top_k]

        items = []
        for item_idx, score in ranked:
            try:
                from src.explainability.shap_explainer import SHAPExplainer
                explainer = SHAPExplainer(reg.lgbm_model)
                feat_idx = item_to_feat_idx.get(item_idx, 0)
                row = feat.iloc[[feat_idx]]
                title_col = reg.item_features[reg.item_features["item_idx"] == item_idx]["title"].values
                title = str(title_col[0]) if len(title_col) > 0 and pd.notna(title_col[0]) else f"Item #{item_idx}"
                explanation = explainer.explain_as_text(row, title)
            except Exception:
                explanation = "Recommended based on your interaction patterns and item features."
            items.append(_build_item_response(item_idx, score, explanation, reg.item_features))
        return items

    def recommend_deep(
        self,
        user_idx: Optional[int],
        top_k: int = 10,
        liked_items: list[int] | None = None,
    ) -> list[RecommendedItem]:
        """Recommend using the Two-Tower neural model."""
        reg = self.registry
        if not reg.deep_available or reg.deep_trainer is None:
            return []

        seen = reg.user_seen.get(user_idx, set()) if user_idx is not None else set()

        if user_idx is None or user_idx >= reg.n_users:
            # Cold-start: if liked_items provided, use their mean embedding as proxy user vector
            if liked_items and reg.item_embeddings is not None:
                valid = [i for i in liked_items if 0 <= i < len(reg.item_embeddings)]
                if valid:
                    proxy = reg.item_embeddings[valid].mean(axis=0)
                    norm = np.linalg.norm(proxy)
                    if norm > 0:
                        proxy = proxy / norm
                    scores = reg.item_embeddings @ proxy
                    exclude = set(liked_items) | self._untitled_items
                    excl_arr = np.array([i for i in exclude if 0 <= i < len(scores)], dtype=np.int64)
                    if len(excl_arr):
                        scores[excl_arr] = -np.inf
                    top_indices = np.argpartition(scores, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
                    recs = [(int(i), float(scores[i])) for i in top_indices]
                    items = []
                    for item_idx, score in recs:
                        if reg.language_explainer:
                            explanation = reg.language_explainer.explain_from_items(
                                liked_item_idxs=valid,
                                item_idx=item_idx,
                                score=score,
                            )
                        else:
                            explanation = "Recommended based on the items you selected."
                        items.append(_build_item_response(item_idx, score, explanation, reg.item_features))
                    return items
            # No liked items either — fall back to naive
            return self.recommend_naive(user_idx=None, top_k=top_k)

        recs = reg.deep_trainer.recommend(
            user_idx,
            reg.item_embeddings,
            top_k=top_k,
            exclude_items=seen | self._untitled_items,
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
        liked_items: list[int] | None = None,
    ) -> list[RecommendedItem]:
        """Route recommendation to the specified model."""
        if model == "naive":
            return self.recommend_naive(user_idx, top_k)
        elif model == "classical":
            return self.recommend_classical(user_idx, top_k)
        elif model == "deep":
            return self.recommend_deep(user_idx, top_k, liked_items=liked_items)
        else:
            raise ValueError(f"Unknown model: {model!r}. Choose from naive, classical, deep.")

    def get_popular_items(self, n: int = 50) -> list[dict]:
        """Return the top-N globally popular items for the UI item selector."""
        reg = self.registry
        if not reg.naive_available or reg.naive_model is None or reg.item_features is None:
            return []
        pop_titled = reg.naive_model.popularity[~reg.naive_model.popularity.index.isin(self._untitled_items)]
        top_items = list(pop_titled.head(n).index)
        result = []
        for iid in top_items:
            row = reg.item_features[reg.item_features["item_idx"] == iid]
            if row.empty:
                continue
            r = row.iloc[0]
            t = r.get("title")
            cat = r.get("category")
            brd = r.get("brand")
            result.append({
                "item_idx": int(iid),
                "title": str(t) if pd.notna(t) and str(t) != "nan" else f"Item #{iid}",
                "category": str(cat) if pd.notna(cat) and str(cat) != "nan" else "Unknown",
                "brand": str(brd) if pd.notna(brd) and str(brd) != "nan" else "Unknown",
                "price": float(r["price"]) if pd.notna(r.get("price")) else None,
                "avg_rating": round(float(r["item_avg_rating"]), 2) if "item_avg_rating" in r else None,
                "num_ratings": int(r["item_num_ratings"]) if "item_num_ratings" in r else None,
            })
        return result
