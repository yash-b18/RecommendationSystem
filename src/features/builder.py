"""
Feature matrix builder.

Assembles the full user-item feature matrix for the classical (LightGBM) model
by joining user features, item features, and interaction-level features.

Feature tiers (for the ablation experiment):
  id_only              : user_idx, item_idx (encoded as floats)
  id_metadata          : + item stats, category, brand, price
  id_metadata_text     : + user-item title cosine similarity
  id_metadata_text_history : + user history aggregates

The `feature_set` parameter controls which tier is used.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DataConfig, PROCESSED_DIR
from src.features.item_features import compute_item_features
from src.features.text_features import (
    build_item_tfidf,
    build_user_text_profiles,
    compute_text_similarity,
)
from src.features.user_features import compute_user_features
from src.utils.logging import get_logger

logger = get_logger(__name__)

FEATURE_SETS = ["id_only", "id_metadata", "id_metadata_text", "id_metadata_text_history"]


def build_feature_matrix(
    df: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    user_profiles: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    feature_set: str = "id_metadata_text_history",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a feature matrix for a given interaction DataFrame.

    Args:
        df: Interactions DataFrame with [user_idx, item_idx, label].
        user_features: Per-user features (output of compute_user_features).
        item_features: Per-item features (output of compute_item_features).
        user_profiles: User text profiles (output of build_user_text_profiles).
        vectorizer: Fitted TF-IDF vectorizer.
        feature_set: One of FEATURE_SETS.

    Returns:
        Tuple of (feature_df, feature_column_names).
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")

    feat = df[["user_idx", "item_idx"]].copy()
    feature_cols: list[str] = []

    # --- Tier 1: ID-only ---
    feat["user_idx_f"] = feat["user_idx"].astype(float)
    feat["item_idx_f"] = feat["item_idx"].astype(float)
    feature_cols += ["user_idx_f", "item_idx_f"]

    if feature_set == "id_only":
        feat["label"] = df["label"].values
        return feat[feature_cols + ["label"]], feature_cols

    # --- Tier 2: + metadata ---
    item_meta_cols = [
        "item_avg_rating", "item_rating_std", "item_num_ratings",
        "item_popularity", "item_positive_rate",
        "category_idx", "brand_idx", "price_norm",
    ]
    item_meta_cols = [c for c in item_meta_cols if c in item_features.columns]

    feat = feat.merge(
        item_features[["item_idx"] + item_meta_cols],
        on="item_idx", how="left",
    )
    feature_cols += item_meta_cols

    # Category match: does item's category == user's top category?
    if "user_top_category" in user_features.columns and "category" in item_features.columns:
        uid_to_top_cat = dict(zip(user_features["user_idx"], user_features["user_top_category"]))
        iid_to_cat = dict(zip(item_features["item_idx"], item_features["category"]))
        feat["category_match"] = [
            1 if uid_to_top_cat.get(u, "") == iid_to_cat.get(i, "__") else 0
            for u, i in zip(feat["user_idx"], feat["item_idx"])
        ]
        feature_cols.append("category_match")

    if feature_set == "id_metadata":
        feat["label"] = df["label"].values
        return feat[feature_cols + ["label"]], feature_cols

    # --- Tier 3: + text similarity ---
    text_sims = compute_text_similarity(user_profiles, item_features, vectorizer, feat)
    feat["text_similarity"] = text_sims.values
    feature_cols.append("text_similarity")

    if feature_set == "id_metadata_text":
        feat["label"] = df["label"].values
        return feat[feature_cols + ["label"]], feature_cols

    # --- Tier 4: + user history aggregates ---
    user_hist_cols = [
        "user_avg_rating", "user_rating_std", "user_num_ratings",
        "user_num_positives", "user_n_categories", "user_recency",
        "user_avg_item_price",
    ]
    user_hist_cols = [c for c in user_hist_cols if c in user_features.columns]

    feat = feat.merge(
        user_features[["user_idx"] + user_hist_cols],
        on="user_idx", how="left",
    )
    feature_cols += user_hist_cols

    feat["label"] = df["label"].values
    feat = feat.fillna(0.0)
    return feat[feature_cols + ["label"]], feature_cols


def run_feature_pipeline(cfg: DataConfig) -> None:
    """
    Build and save feature matrices for train / val / test splits.

    Also saves user features, item features, and the TF-IDF vectorizer.

    Args:
        cfg: DataConfig.
    """
    logger.info("Loading processed data…")
    train_df = pd.read_parquet(cfg.train_path)
    val_df = pd.read_parquet(cfg.val_path)
    test_df = pd.read_parquet(cfg.test_path)
    metadata_df = pd.read_parquet(cfg.metadata_path)

    # Ensure label column exists in val/test
    for split in (val_df, test_df):
        if "label" not in split.columns:
            split["label"] = (split["rating"] >= cfg.positive_rating_threshold).astype(int)

    logger.info("Computing user features…")
    user_features = compute_user_features(train_df, metadata_df, cfg.positive_rating_threshold)

    logger.info("Computing item features…")
    item_features, cat_enc, brand_enc = compute_item_features(
        train_df, metadata_df, cfg.positive_rating_threshold
    )

    logger.info("Building TF-IDF and user text profiles…")
    tfidf_matrix, vectorizer = build_item_tfidf(item_features)
    user_profiles = build_user_text_profiles(train_df, item_features, cfg.positive_rating_threshold)

    logger.info("Building feature matrices (full tier)…")
    train_feat, feature_cols = build_feature_matrix(
        train_df, user_features, item_features, user_profiles, vectorizer,
        feature_set="id_metadata_text_history",
    )
    val_feat, _ = build_feature_matrix(
        val_df, user_features, item_features, user_profiles, vectorizer,
        feature_set="id_metadata_text_history",
    )
    test_feat, _ = build_feature_matrix(
        test_df, user_features, item_features, user_profiles, vectorizer,
        feature_set="id_metadata_text_history",
    )

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    user_features.to_parquet(cfg.user_features_path, index=False)
    item_features.to_parquet(cfg.item_features_path, index=False)
    train_feat.to_parquet(cfg.train_features_path, index=False)
    val_feat.to_parquet(cfg.val_features_path, index=False)
    test_feat.to_parquet(cfg.test_features_path, index=False)

    artifacts_path = PROCESSED_DIR / "feature_artifacts.pkl"
    with open(artifacts_path, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "cat_enc": cat_enc,
            "brand_enc": brand_enc,
            "feature_cols": feature_cols,
        }, f)

    logger.info("Feature pipeline complete. Saved to %s", PROCESSED_DIR)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)
