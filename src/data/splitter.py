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

    # Training = everything not in test or val (identified by user+item pair)
    test_keys = set(zip(test_df["user_idx"], test_df["item_idx"]))
    val_keys = set(zip(val_df["user_idx"], val_df["item_idx"]))

    def _not_in_held_out(row: pd.Series) -> bool:
        k = (row["user_idx"], row["item_idx"])
        return k not in test_keys and k not in val_keys

    train_mask = interactions.apply(_not_in_held_out, axis=1)
    train_df = interactions[train_mask].copy()

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

    neg_rows = []
    all_items = np.arange(n_items)

    for _, row in positives.iterrows():
        uid = int(row["user_idx"])
        seen = user_seen.get(uid, set())
        candidates = np.setdiff1d(all_items, list(seen))
        if len(candidates) == 0:
            continue
        sampled = rng.choice(
            candidates,
            size=min(n_negatives, len(candidates)),
            replace=False,
        )
        for iid in sampled:
            neg_rows.append({
                "user_idx": uid,
                "item_idx": int(iid),
                "rating": 0.0,
                "label": 0,
                "user_id": row.get("user_id", ""),
                "item_id": "",
                "timestamp": row.get("timestamp", 0),
            })

    neg_df = pd.DataFrame(neg_rows)
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
