"""
User-level feature engineering.

Computes aggregate statistics over a user's interaction history:
  - avg_rating, rating_std
  - num_ratings (activity level)
  - avg_review_recency (days since first interaction, normalised)
  - top_categories (mode category in rated items)
  - category_affinity vector (fraction of positive ratings per category)

All features are derived from training data only to prevent leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_user_features(
    train_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Compute per-user aggregate features from the training split.

    Args:
        train_df: Training interactions with columns [user_idx, item_idx, rating, timestamp].
        metadata_df: Item metadata with columns [item_idx, category, brand, price].
        positive_threshold: Rating threshold for a positive interaction.

    Returns:
        DataFrame indexed by user_idx with one row per user.
    """
    # Merge metadata into interactions
    merged = train_df.merge(
        metadata_df[["item_idx", "category", "brand", "price"]],
        on="item_idx",
        how="left",
    )
    merged["is_positive"] = (merged["rating"] >= positive_threshold).astype(int)

    # Basic rating stats
    rating_stats = (
        merged.groupby("user_idx")["rating"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "user_avg_rating", "std": "user_rating_std", "count": "user_num_ratings"})
    )
    rating_stats["user_rating_std"] = rating_stats["user_rating_std"].fillna(0.0)

    # Positive interaction count
    pos_count = (
        merged[merged["is_positive"] == 1]
        .groupby("user_idx")["item_idx"]
        .count()
        .rename("user_num_positives")
    )

    # Most frequent category among positively-rated items
    pos_merged = merged[merged["is_positive"] == 1].copy()
    top_category = (
        pos_merged.groupby("user_idx")["category"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown")
        .rename("user_top_category")
    )

    # Number of unique categories interacted with
    n_categories = (
        merged.groupby("user_idx")["category"]
        .nunique()
        .rename("user_n_categories")
    )

    # Recency: mean timestamp normalised to [0, 1]
    if merged["timestamp"].notna().any():
        ts_min = merged["timestamp"].min()
        ts_max = merged["timestamp"].max()
        ts_range = max(ts_max - ts_min, 1)
        recency = (
            merged.groupby("user_idx")["timestamp"]
            .mean()
            .apply(lambda t: (t - ts_min) / ts_range)
            .rename("user_recency")
        )
    else:
        recency = pd.Series(0.5, index=rating_stats.index, name="user_recency")

    # Average price of positively rated items
    avg_price = (
        pos_merged.groupby("user_idx")["price"]
        .mean()
        .rename("user_avg_item_price")
    )

    user_features = (
        rating_stats
        .join(pos_count, how="left")
        .join(top_category, how="left")
        .join(n_categories, how="left")
        .join(recency, how="left")
        .join(avg_price, how="left")
    )

    user_features["user_num_positives"] = user_features["user_num_positives"].fillna(0)
    user_features["user_top_category"] = user_features["user_top_category"].fillna("Unknown")
    user_features["user_avg_item_price"] = user_features["user_avg_item_price"].fillna(
        user_features["user_avg_item_price"].median()
    )

    logger.info("Computed user features for %d users, %d features",
                len(user_features), len(user_features.columns))
    return user_features.reset_index()
