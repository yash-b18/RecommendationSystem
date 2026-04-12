"""
Item-level feature engineering.

Computes per-item statistics from the training set:
  - popularity (interaction count)
  - avg_rating, rating_std
  - positive_rate (fraction of ratings >= threshold)
  - price (from metadata, normalised)
  - category (label-encoded)
  - brand (label-encoded)
  - title text (raw; used downstream in text_features.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_item_features(
    train_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Compute per-item aggregate features from the training split.

    Args:
        train_df: Training interactions with [user_idx, item_idx, rating].
        metadata_df: Item metadata with [item_idx, title, category, brand, price].
        positive_threshold: Rating threshold for a positive interaction.

    Returns:
        Tuple of (item_features_df, category_encoder, brand_encoder).
    """
    # Rating statistics from interactions
    rating_stats = (
        train_df.groupby("item_idx")["rating"]
        .agg(["mean", "std", "count"])
        .rename(columns={
            "mean": "item_avg_rating",
            "std": "item_rating_std",
            "count": "item_num_ratings",
        })
    )
    rating_stats["item_rating_std"] = rating_stats["item_rating_std"].fillna(0.0)

    # Positive rate
    pos_rate = (
        train_df.groupby("item_idx")["rating"]
        .apply(lambda x: (x >= positive_threshold).mean())
        .rename("item_positive_rate")
    )

    # Normalised popularity (log scale for heavy tail)
    rating_stats["item_popularity"] = np.log1p(rating_stats["item_num_ratings"])

    item_stats = rating_stats.join(pos_rate, how="left").reset_index()

    # Merge with metadata
    meta_cols = ["item_idx", "title", "category", "brand", "price"]
    meta_cols = [c for c in meta_cols if c in metadata_df.columns]
    item_features = item_stats.merge(metadata_df[meta_cols], on="item_idx", how="left")

    # Encode categorical fields
    cat_enc = LabelEncoder()
    brand_enc = LabelEncoder()

    item_features["category"] = item_features["category"].fillna("Unknown")
    item_features["brand"] = item_features["brand"].fillna("Unknown")
    item_features["category_idx"] = cat_enc.fit_transform(item_features["category"])
    item_features["brand_idx"] = brand_enc.fit_transform(item_features["brand"])

    # Normalise price: fill NaN with median, then scale to [0, 1]
    price_median = item_features["price"].median()
    item_features["price"] = item_features["price"].fillna(price_median)
    price_max = item_features["price"].max()
    if price_max > 0:
        item_features["price_norm"] = item_features["price"] / price_max
    else:
        item_features["price_norm"] = 0.0

    logger.info("Computed item features for %d items, %d features",
                len(item_features), len(item_features.columns))
    return item_features, cat_enc, brand_enc
