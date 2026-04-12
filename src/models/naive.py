"""
Naive baseline recommendation models.

Implements two baselines:
  1. GlobalPopularityRecommender  — ranks items by total interaction count
  2. CategoryPopularityRecommender — ranks items by popularity within each category,
     falling back to global popularity for users with no category signal

Both support the same .fit() / .recommend() interface as the other models.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class GlobalPopularityRecommender:
    """
    Recommends the globally most popular items that a user has not yet seen.

    Popularity is measured as the total number of interactions per item in the
    training set (frequency, not rating).
    """

    def __init__(self) -> None:
        self.popularity: pd.Series | None = None   # item_idx → score
        self.user_seen: dict[int, set[int]] = {}

    def fit(self, train_df: pd.DataFrame) -> "GlobalPopularityRecommender":
        """
        Compute item popularity counts and per-user seen sets.

        Args:
            train_df: Training interactions with [user_idx, item_idx].

        Returns:
            self
        """
        self.popularity = (
            train_df.groupby("item_idx")["item_idx"]
            .count()
            .rename("score")
            .sort_values(ascending=False)
        )
        self.user_seen = (
            train_df.groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )
        logger.info("GlobalPopularity fitted on %d items", len(self.popularity))
        return self

    def recommend(self, user_idx: int, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Return top-K unseen items for a user.

        Args:
            user_idx: Encoded user index.
            top_k: Number of recommendations to return.

        Returns:
            List of (item_idx, score) tuples sorted descending by score.
        """
        if self.popularity is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        seen = self.user_seen.get(user_idx, set())
        recs = []
        for iid, score in self.popularity.items():
            if iid not in seen:
                recs.append((int(iid), float(score)))
                if len(recs) == top_k:
                    break
        return recs

    def recommend_batch(
        self, user_idxs: list[int], top_k: int = 10
    ) -> dict[int, list[tuple[int, float]]]:
        """Recommend for a list of users."""
        return {u: self.recommend(u, top_k) for u in user_idxs}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved GlobalPopularityRecommender → %s", path)

    @classmethod
    def load(cls, path: Path) -> "GlobalPopularityRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


class CategoryPopularityRecommender:
    """
    Recommends the most popular items within a user's preferred category.

    Falls back to global popularity for users with no category signal or
    when the category has fewer items than top_k.
    """

    def __init__(self) -> None:
        self.category_popularity: dict[str, pd.Series] = {}
        self.global_popularity: pd.Series | None = None
        self.user_seen: dict[int, set[int]] = {}
        self.user_top_category: dict[int, str] = {}
        self.item_categories: dict[int, str] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        item_features: pd.DataFrame,
        user_features: pd.DataFrame,
        positive_threshold: float = 4.0,
    ) -> "CategoryPopularityRecommender":
        """
        Fit category-level popularity statistics.

        Args:
            train_df: Training interactions.
            item_features: Per-item features including [item_idx, category].
            user_features: Per-user features including [user_idx, user_top_category].
            positive_threshold: Rating threshold for positive interactions.

        Returns:
            self
        """
        merged = train_df.merge(
            item_features[["item_idx", "category"]], on="item_idx", how="left"
        )
        merged["category"] = merged["category"].fillna("Unknown")

        # Global popularity fallback
        self.global_popularity = (
            merged.groupby("item_idx")["item_idx"]
            .count()
            .rename("score")
            .sort_values(ascending=False)
        )

        # Per-category popularity
        for cat, grp in merged.groupby("category"):
            self.category_popularity[cat] = (
                grp.groupby("item_idx")["item_idx"]
                .count()
                .rename("score")
                .sort_values(ascending=False)
            )

        # User seen items and top categories
        self.user_seen = (
            train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )
        self.user_top_category = dict(
            zip(user_features["user_idx"], user_features["user_top_category"])
        )
        self.item_categories = dict(
            zip(item_features["item_idx"], item_features["category"])
        )
        logger.info(
            "CategoryPopularity fitted on %d categories", len(self.category_popularity)
        )
        return self

    def recommend(self, user_idx: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Return top-K category-aware recommendations for a user."""
        if self.global_popularity is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        seen = self.user_seen.get(user_idx, set())
        user_cat = self.user_top_category.get(user_idx, "Unknown")

        cat_pop = self.category_popularity.get(user_cat, self.global_popularity)
        recs = [
            (int(iid), float(score))
            for iid, score in cat_pop.items()
            if iid not in seen
        ]
        if len(recs) < top_k:
            # Supplement with global popularity
            global_recs = [
                (int(iid), float(score))
                for iid, score in self.global_popularity.items()
                if iid not in seen and (int(iid), float(score)) not in recs
            ]
            recs = recs + global_recs

        return recs[:top_k]

    def recommend_batch(
        self, user_idxs: list[int], top_k: int = 10
    ) -> dict[int, list[tuple[int, float]]]:
        return {u: self.recommend(u, top_k) for u in user_idxs}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved CategoryPopularityRecommender → %s", path)

    @classmethod
    def load(cls, path: Path) -> "CategoryPopularityRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)
