"""
Data preprocessing for Amazon Reviews 2023.

Loads raw JSONL files, filters low-activity users and items,
encodes user/item IDs as sequential integers, and saves clean
parquet files to data/processed/.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import DataConfig, PROCESSED_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_raw_reviews(path: Path) -> pd.DataFrame:
    """
    Load raw reviews JSONL into a DataFrame, keeping only the columns we need.

    Expected columns: user_id, asin (=parent_asin if available), rating, timestamp.

    Args:
        path: Path to raw reviews JSONL.

    Returns:
        DataFrame with columns [user_id, item_id, rating, timestamp].
    """
    logger.info("Loading reviews from %s", path)
    df = pd.read_json(path, lines=True)

    # Use parent_asin when available as the canonical item identifier
    if "parent_asin" in df.columns:
        df["item_id"] = df["parent_asin"].fillna(df["asin"])
    else:
        df["item_id"] = df["asin"]

    keep = ["user_id", "item_id", "rating", "timestamp"]
    df = df[keep].copy()

    # Normalise types
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    # Drop rows with missing essentials
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    df = df.drop_duplicates(subset=["user_id", "item_id"])
    logger.info("Loaded %d reviews (%d users, %d items)",
                len(df), df["user_id"].nunique(), df["item_id"].nunique())
    return df


def load_raw_metadata(path: Path) -> pd.DataFrame:
    """
    Load raw metadata JSONL into a DataFrame.

    Args:
        path: Path to raw metadata JSONL.

    Returns:
        DataFrame with item-level attributes.
    """
    logger.info("Loading metadata from %s", path)
    df = pd.read_json(path, lines=True)

    if "parent_asin" in df.columns:
        df["item_id"] = df["parent_asin"].fillna(df.get("asin", df["parent_asin"]))
    elif "asin" in df.columns:
        df["item_id"] = df["asin"]
    else:
        raise ValueError("Metadata JSONL has neither 'asin' nor 'parent_asin' column.")

    # Flatten categories: take first listed category as primary
    if "categories" in df.columns:
        df["category"] = df["categories"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown"
        )
    else:
        df["category"] = "Unknown"

    # Normalise price to float
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = float("nan")

    keep_cols = ["item_id", "title", "category", "price"]
    # Include brand/store if present
    for col in ("store", "brand"):
        if col in df.columns:
            df["brand"] = df[col].fillna("Unknown")
            break
    else:
        df["brand"] = "Unknown"
    keep_cols.append("brand")

    # Include description for text features
    for col in ("description", "features"):
        if col in df.columns:
            df["description"] = df[col].apply(
                lambda x: " ".join(x) if isinstance(x, list) else str(x)
            )
            break
    else:
        df["description"] = ""
    keep_cols.append("description")

    df = df[[c for c in keep_cols if c in df.columns]].drop_duplicates("item_id")
    logger.info("Loaded metadata for %d items", len(df))
    return df


def filter_interactions(
    df: pd.DataFrame,
    min_user_reviews: int,
    min_item_reviews: int,
) -> pd.DataFrame:
    """
    Iteratively filter users and items below minimum interaction counts.

    Args:
        df: Interactions DataFrame.
        min_user_reviews: Minimum reviews per user.
        min_item_reviews: Minimum reviews per item.

    Returns:
        Filtered DataFrame.
    """
    prev_size = len(df) + 1
    while len(df) < prev_size:
        prev_size = len(df)
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user_reviews].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= min_item_reviews].index)]

    logger.info("After filtering: %d reviews, %d users, %d items",
                len(df), df["user_id"].nunique(), df["item_id"].nunique())
    return df


def encode_ids(
    df: pd.DataFrame,
    user_encoder_path: Path,
    item_encoder_path: Path,
) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Encode user_id and item_id as zero-based integers and save encoders.

    Args:
        df: Interactions DataFrame with string user_id and item_id.
        user_encoder_path: Where to save the user LabelEncoder.
        item_encoder_path: Where to save the item LabelEncoder.

    Returns:
        Tuple of (encoded_df, user_encoder, item_encoder).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    df = df.copy()
    df["user_idx"] = user_enc.fit_transform(df["user_id"])
    df["item_idx"] = item_enc.fit_transform(df["item_id"])

    with open(user_encoder_path, "wb") as f:
        pickle.dump(user_enc, f)
    with open(item_encoder_path, "wb") as f:
        pickle.dump(item_enc, f)

    logger.info("Encoded %d users and %d items", len(user_enc.classes_), len(item_enc.classes_))
    return df, user_enc, item_enc


def preprocess(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline: load → filter → encode → save.

    Args:
        cfg: DataConfig.

    Returns:
        Tuple of (interactions_df, metadata_df).
    """
    interactions = load_raw_reviews(cfg.raw_reviews_path)
    metadata = load_raw_metadata(cfg.raw_metadata_path)

    # Optional debug subsetting before filtering
    if cfg.debug:
        top_users = interactions["user_id"].value_counts().head(cfg.debug_n_users).index
        interactions = interactions[interactions["user_id"].isin(top_users)].copy()
        logger.info("Debug mode: trimmed to %d users", interactions["user_id"].nunique())

    interactions = filter_interactions(
        interactions, cfg.min_user_reviews, cfg.min_item_reviews
    )

    interactions, user_enc, item_enc = encode_ids(
        interactions, cfg.user_encoder_path, cfg.item_encoder_path
    )

    # Keep only metadata for items present after filtering
    valid_items = set(interactions["item_id"].unique())
    metadata = metadata[metadata["item_id"].isin(valid_items)].copy()

    # Add item_idx to metadata
    item_id_to_idx = dict(zip(user_enc.classes_, range(len(user_enc.classes_))))
    item_id_to_idx = dict(zip(item_enc.classes_, range(len(item_enc.classes_))))
    metadata["item_idx"] = metadata["item_id"].map(item_id_to_idx)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(cfg.interactions_path, index=False)
    metadata.to_parquet(cfg.metadata_path, index=False)
    logger.info("Saved interactions → %s", cfg.interactions_path)
    logger.info("Saved metadata     → %s", cfg.metadata_path)

    return interactions, metadata
