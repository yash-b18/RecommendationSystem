"""
PyTorch Dataset classes for the Two-Tower neural recommender.

Provides:
  - InteractionDataset  — for training (user_idx, item_idx, label)
  - InferenceDataset    — for building item tower embeddings at inference time
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """
    Dataset of (user_idx, item_idx, label) triples for training the Two-Tower model.

    Optionally attaches item feature vectors when `item_features` is provided.

    Args:
        df: DataFrame with columns [user_idx, item_idx, label].
        item_features: Optional numpy array of shape (n_items, n_item_features).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        item_features: np.ndarray | None = None,
    ) -> None:
        self.user_idxs = torch.LongTensor(df["user_idx"].values)
        self.item_idxs = torch.LongTensor(df["item_idx"].values)
        self.labels = torch.FloatTensor(df["label"].values)
        self.item_features = (
            torch.FloatTensor(item_features) if item_features is not None else None
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {
            "user_idx": self.user_idxs[idx],
            "item_idx": self.item_idxs[idx],
            "label": self.labels[idx],
        }
        if self.item_features is not None:
            sample["item_features"] = self.item_features[self.item_idxs[idx]]
        return sample


class BPRDataset(Dataset):
    """
    Bayesian Personalized Ranking dataset: (user, pos_item, neg_item) triples.

    For each positive interaction, a random negative item is paired.
    If the DataFrame has pre-sampled negatives (label=0 rows from train split),
    those are used. Otherwise (e.g., val split with only positives), random
    item indices are sampled globally.

    Args:
        df: DataFrame with columns [user_idx, item_idx, label].
        item_features: Optional numpy array of shape (n_items, n_item_features).
        n_items: Total number of items (required when df has no negatives).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        item_features: np.ndarray | None = None,
        n_items: int | None = None,
    ) -> None:
        label_col = "label" if "label" in df.columns else None
        pos_df = (
            df[df["label"] == 1].reset_index(drop=True)
            if label_col
            else df.reset_index(drop=True)
        )
        neg_df = (
            df[df["label"] == 0].reset_index(drop=True)
            if label_col
            else pd.DataFrame()
        )

        n_pos = len(pos_df)
        rng = np.random.default_rng(42)
        sampled_negs = np.empty(n_pos, dtype=np.int64)

        if len(neg_df) > 0:
            # Per-user pairing: for each user's positive, sample from THAT user's negatives.
            # This ensures user-specific gradients (critical for non-collapsed user tower).
            user_neg_map: dict[int, np.ndarray] = {
                uid: grp["item_idx"].values
                for uid, grp in neg_df.groupby("user_idx")
            }
            pos_users = pos_df["user_idx"].values
            for i, uid in enumerate(pos_users):
                neg_pool = user_neg_map.get(uid)
                if neg_pool is not None and len(neg_pool) > 0:
                    sampled_negs[i] = neg_pool[rng.integers(0, len(neg_pool))]
                else:
                    # Fallback: global random item
                    assert n_items is not None
                    sampled_negs[i] = rng.integers(0, n_items)
        else:
            # No pre-sampled negatives (val/test split) — sample globally
            assert n_items is not None, "n_items required when df has no label=0 rows"
            sampled_negs = rng.integers(0, n_items, size=n_pos).astype(np.int64)

        self.user_idxs = torch.LongTensor(pos_df["user_idx"].values)
        self.pos_item_idxs = torch.LongTensor(pos_df["item_idx"].values)
        self.neg_item_idxs = torch.LongTensor(sampled_negs)

        # Store full item feature matrix for indexed lookup
        self.item_features = (
            torch.FloatTensor(item_features) if item_features is not None else None
        )

    def __len__(self) -> int:
        return len(self.user_idxs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pos_iid = self.pos_item_idxs[idx]
        neg_iid = self.neg_item_idxs[idx]
        sample: dict[str, torch.Tensor] = {
            "user_idx": self.user_idxs[idx],
            "pos_item_idx": pos_iid,
            "neg_item_idx": neg_iid,
        }
        if self.item_features is not None:
            sample["pos_item_features"] = self.item_features[pos_iid]
            sample["neg_item_features"] = self.item_features[neg_iid]
        return sample


class InferenceDataset(Dataset):
    """
    Dataset over all items for computing item tower embeddings in batch.

    Args:
        n_items: Total number of items.
        item_features: Optional numpy array of shape (n_items, n_item_features).
    """

    def __init__(
        self,
        n_items: int,
        item_features: np.ndarray | None = None,
    ) -> None:
        self.item_idxs = torch.arange(n_items, dtype=torch.long)
        self.item_features = (
            torch.FloatTensor(item_features) if item_features is not None else None
        )

    def __len__(self) -> int:
        return len(self.item_idxs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {"item_idx": self.item_idxs[idx]}
        if self.item_features is not None:
            sample["item_features"] = self.item_features[idx]
        return sample
