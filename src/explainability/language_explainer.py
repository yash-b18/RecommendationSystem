"""
Metadata-grounded language explanations for the Two-Tower neural model.

Generates factual, template-based recommendation explanations that are
grounded in observable data (user history, item metadata) — not hallucinated.

The explanation is constructed by:
  1. Identifying the user's dominant categories and brands from training history
  2. Comparing these against the recommended item's metadata
  3. Filling a natural language template based on the match signals

This approach provides transparency without requiring a generative LLM.
"""

from __future__ import annotations

import pandas as pd


def _most_common(values: list[str], n: int = 3) -> list[str]:
    """Return the top-n most frequent values in a list."""
    from collections import Counter
    return [v for v, _ in Counter(values).most_common(n)]


class LanguageExplainer:
    """
    Generates template-based explanations for Two-Tower recommendations.

    Args:
        train_df: Training interactions with [user_idx, item_idx, rating].
        item_features: Item metadata with [item_idx, title, category, brand].
        positive_threshold: Minimum rating to count as positive.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        item_features: pd.DataFrame,
        positive_threshold: float = 4.0,
    ) -> None:
        # Build user history from positive interactions
        positives = train_df[train_df["rating"] >= positive_threshold].copy()
        positives = positives.merge(
            item_features[["item_idx", "title", "category", "brand"]],
            on="item_idx",
            how="left",
        )

        self._user_categories: dict[int, list[str]] = (
            positives.groupby("user_idx")["category"]
            .apply(lambda x: x.fillna("Unknown").tolist())
            .to_dict()
        )
        self._user_brands: dict[int, list[str]] = (
            positives.groupby("user_idx")["brand"]
            .apply(lambda x: x.fillna("Unknown").tolist())
            .to_dict()
        )
        self._user_titles: dict[int, list[str]] = (
            positives.groupby("user_idx")["title"]
            .apply(lambda x: x.fillna("").tolist())
            .to_dict()
        )

        self._item_meta: dict[int, dict] = (
            item_features.set_index("item_idx")[["title", "category", "brand"]]
            .fillna("Unknown")
            .to_dict(orient="index")
        )

    def explain(
        self,
        user_idx: int,
        item_idx: int,
        score: float,
    ) -> str:
        """
        Generate a natural language explanation for recommending item_idx to user_idx.

        Args:
            user_idx: Encoded user index.
            item_idx: Encoded item index.
            score: Model similarity score (higher = more relevant).

        Returns:
            Explanation string grounded in user history and item metadata.
        """
        item_meta = self._item_meta.get(item_idx, {})
        item_title = str(item_meta.get("title", "this item"))[:60]
        item_category = str(item_meta.get("category", "Unknown"))
        item_brand = str(item_meta.get("brand", "Unknown"))

        user_categories = self._user_categories.get(user_idx, [])
        user_brands = self._user_brands.get(user_idx, [])
        user_titles = self._user_titles.get(user_idx, [])

        top_user_cats = _most_common(user_categories, 2)
        top_user_brands = _most_common(user_brands, 2)

        reasons = []

        # Category match
        if item_category in top_user_cats:
            reasons.append(f"you frequently enjoy {item_category} products")

        # Brand match
        if item_brand != "Unknown" and item_brand in top_user_brands:
            reasons.append(f"you have liked {item_brand} items before")

        # Category affinity (partial)
        elif item_category != "Unknown" and any(item_category in cat for cat in user_categories):
            reasons.append(f"it falls in a category you explore ({item_category})")

        # Title similarity (basic keyword overlap)
        item_words = set(item_title.lower().split())
        history_words = set(
            w for t in user_titles[:10] for w in t.lower().split()
        )
        overlap = item_words & history_words - {"the", "a", "an", "and", "of", "for", "with", ""}
        if len(overlap) >= 2:
            reasons.append(f"it shares themes with items you liked ('{', '.join(list(overlap)[:3])}')")

        # Fallback
        if not reasons:
            if top_user_cats:
                reasons.append(f"it complements your interest in {', '.join(top_user_cats)}")
            else:
                reasons.append("it is highly rated and aligns with your interaction patterns")

        reasons_str = "; ".join(reasons)
        return f'Recommended because {reasons_str}.'

    def explain_batch(
        self,
        user_idx: int,
        recommendations: list[tuple[int, float]],
    ) -> list[str]:
        """
        Generate explanations for a list of (item_idx, score) recommendations.

        Args:
            user_idx: Encoded user index.
            recommendations: List of (item_idx, score) tuples.

        Returns:
            List of explanation strings, one per recommendation.
        """
        return [
            self.explain(user_idx, item_idx, score)
            for item_idx, score in recommendations
        ]
