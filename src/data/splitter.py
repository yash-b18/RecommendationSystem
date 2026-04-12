"""
Train / validation / test splitting for recommendation data.

Uses a leave-last-out strategy per user:
  - The most recent positive interaction per user → test set
  - The second most recent → validation set
  - All remaining → training set

This mirrors standard practice in recommender system evaluation
(e.g., BERT4Rec, LightGCN papers) and avoids temporal leakage.

Negative sampling is also handled here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DataConfig
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def split_leave_last_out(
    interactions: pd.DataFrame,
    positive_threshold: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train / val / test using leave-last-out per user.

    Only positive interactions (rating >= threshold) are used for val/test.
    Training set includes all remaining interactions (positive and negative).

    Args:
        interactions: DataFrame with columns [user_id, item_id, user_idx, item_idx,
                      rating, timestamp].
        positive_threshold: Rating threshold for a positive interaction.
        random_seed: For reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    set_seed(random_seed)

    positives = interactions[interactions["rating"] >= positive_threshold].copy()

    # Sort by timestamp; fall back to index if timestamp is missing
    if positives["timestamp"].notna().all():
        positives = positives.sort_values(["user_idx", "timestamp"])
    else:
        positives = positives.sort_values("user_idx")

    # Assign rank within each user (last = most recent)
    positives["_rank"] = positives.groupby("user_idx").cumcount(ascending=False)

    test_mask = positives["_rank"] == 0
    val_mask = positives["_rank"] == 1

    test_df = positives[test_mask].drop(columns=["_rank"])
    val_df = positives[val_mask].drop(columns=["_rank"])

    # Training = everything not in test or val — vectorized via tuple key set
    held_keys = set(
        zip(test_df["user_idx"], test_df["item_idx"])
    ) | set(
        zip(val_df["user_idx"], val_df["item_idx"])
    )
    pair_keys = pd.Series(
        list(zip(interactions["user_idx"], interactions["item_idx"])),
        index=interactions.index,
    )
    train_df = interactions[~pair_keys.isin(held_keys)].copy()

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def sample_negatives(
    train_df: pd.DataFrame,
    n_items: int,
    n_negatives: int,
    positive_threshold: float,
    random_seed: int,
) -> pd.DataFrame:
    """
    Append sampled negative interactions to the training set.

    For each positive interaction, sample `n_negatives` items the user
    has not interacted with. Assigns rating = 0 and label = 0.

    Args:
        train_df: Training interactions (may include low-rated items).
        n_items: Total number of unique items.
        n_negatives: Number of negatives to sample per positive.
        positive_threshold: Rating threshold for positive label.
        random_seed: For reproducibility.

    Returns:
        DataFrame with both positive and negative rows, plus a 'label' column.
    """
    set_seed(random_seed)
    rng = np.random.default_rng(random_seed)

    train_df = train_df.copy()
    train_df["label"] = (train_df["rating"] >= positive_threshold).astype(int)

    positives = train_df[train_df["label"] == 1]

    # Build per-user set of seen items for fast exclusion
    user_seen: dict[int, set[int]] = (
        train_df.groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    all_items = np.arange(n_items)

    # Vectorized: iterate per-user (not per-row) — O(n_users) instead of O(n_positives)
    neg_user_idxs: list[np.ndarray] = []
    neg_item_idxs: list[np.ndarray] = []

    pos_counts = positives.groupby("user_idx").size()

    for uid, n_pos in pos_counts.items():
        seen = user_seen.get(uid, set())
        seen_arr = np.fromiter(seen, dtype=np.int64, count=len(seen))
        candidates = np.setdiff1d(all_items, seen_arr, assume_unique=True)
        if len(candidates) == 0:
            continue
        n_sample = min(n_negatives * n_pos, len(candidates))
        sampled = rng.choice(candidates, size=n_sample, replace=False)
        neg_user_idxs.append(np.full(n_sample, uid, dtype=np.int64))
        neg_item_idxs.append(sampled)

    neg_df = pd.DataFrame({
        "user_idx": np.concatenate(neg_user_idxs) if neg_user_idxs else np.array([], dtype=np.int64),
        "item_idx": np.concatenate(neg_item_idxs) if neg_item_idxs else np.array([], dtype=np.int64),
        "rating": 0.0,
        "label": 0,
        "user_id": "",
        "item_id": "",
        "timestamp": 0,
    })
    result = pd.concat([train_df, neg_df], ignore_index=True)
    logger.info(
        "After negative sampling: %d rows (%d pos, %d neg)",
        len(result),
        (result["label"] == 1).sum(),
        (result["label"] == 0).sum(),
    )
    return result


def run_split(cfg: DataConfig, interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full split pipeline and save parquet files.

    Args:
        cfg: DataConfig.
        interactions: Preprocessed interactions DataFrame.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    n_items = interactions["item_idx"].nunique()

    train_df, val_df, test_df = split_leave_last_out(
        interactions,
        positive_threshold=cfg.positive_rating_threshold,
        random_seed=cfg.random_seed,
    )

    train_df = sample_negatives(
        train_df,
        n_items=n_items,
        n_negatives=cfg.negative_samples_per_positive,
        positive_threshold=cfg.positive_rating_threshold,
        random_seed=cfg.random_seed,
    )

    train_df.to_parquet(cfg.train_path, index=False)
    val_df.to_parquet(cfg.val_path, index=False)
    test_df.to_parquet(cfg.test_path, index=False)

    logger.info("Saved train → %s", cfg.train_path)
    logger.info("Saved val   → %s", cfg.val_path)
    logger.info("Saved test  → %s", cfg.test_path)

    return train_df, val_df, test_df
