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
