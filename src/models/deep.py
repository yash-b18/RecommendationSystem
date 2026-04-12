"""
Deep learning model: Two-Tower Neural Recommender.

Architecture:
  User Tower  : user_id embedding → MLP → L2-normalised user vector
  Item Tower  : item_id embedding + metadata features (+ optional title tokens)
                → MLP → L2-normalised item vector
  Score       : dot product of user and item vectors
  Training    : binary cross-entropy with in-batch negative sampling

At inference:
  1. Pre-compute all item tower embeddings (once).
  2. For each user, compute user embedding on-the-fly.
  3. Score against all item embeddings via dot product.
  4. Return top-K items.

Explainability:
  Provides a metadata-grounded language explanation (no hallucination)
  via the explainability module.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import InferenceDataset, InteractionDataset
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Generic multi-layer perceptron with ReLU activations and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UserTower(nn.Module):
    """
    Encodes a user into a fixed-size embedding vector.

    Args:
        n_users: Vocabulary size for user IDs.
        embedding_dim: ID embedding dimension.
        output_dim: Final embedding dimension.
        hidden_dims: MLP hidden layer sizes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_users: int,
        embedding_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_users, embedding_dim, padding_idx=0)
        self.mlp = MLP(embedding_dim, hidden_dims, output_dim, dropout)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(user_idx)
        out = self.mlp(emb)
        return nn.functional.normalize(out, dim=-1)


class ItemTower(nn.Module):
    """
    Encodes an item into a fixed-size embedding vector.

    Optionally incorporates item metadata features alongside the ID embedding.

    Args:
        n_items: Vocabulary size for item IDs.
        embedding_dim: ID embedding dimension.
        output_dim: Final embedding dimension.
        hidden_dims: MLP hidden layer sizes.
        dropout: Dropout probability.
        n_meta_features: Number of continuous metadata features (0 = ID-only).
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float,
        n_meta_features: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.n_meta = n_meta_features
        input_dim = embedding_dim + n_meta_features
        self.mlp = MLP(input_dim, hidden_dims, output_dim, dropout)

    def forward(
        self,
        item_idx: torch.Tensor,
        item_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.embedding(item_idx)
        if self.n_meta > 0 and item_features is not None:
            emb = torch.cat([emb, item_features], dim=-1)
        out = self.mlp(emb)
        return nn.functional.normalize(out, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Full Two-Tower model with user and item towers.

    Args:
        n_users: Number of users.
        n_items: Number of items.
        embedding_dim: Embedding size for ID lookup tables.
        output_dim: Shared embedding space dimension.
        hidden_dims: MLP hidden layer sizes.
        dropout: Dropout probability.
        n_meta_features: Item metadata feature count.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        output_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        n_meta_features: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.user_tower = UserTower(n_users, embedding_dim, output_dim, hidden_dims, dropout)
        self.item_tower = ItemTower(n_items, embedding_dim, output_dim, hidden_dims, dropout, n_meta_features)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        item_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute dot-product similarity scores for (user, item) pairs.

        Returns:
            Tensor of shape (batch_size,) with similarity scores in [-1, 1].
        """
        u_emb = self.user_tower(user_idx)
        i_emb = self.item_tower(item_idx, item_features)
        return (u_emb * i_emb).sum(dim=-1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TwoTowerTrainer:
    """
    Handles training, validation, and inference for the Two-Tower model.

    Args:
        model: TwoTowerModel instance.
        device: Torch device string.
        lr: Learning rate.
        weight_decay: L2 regularisation.
        patience: Early stopping patience (epochs).
        model_path: Where to save the best model checkpoint.
    """

    def __init__(
        self,
        model: TwoTowerModel,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        model_path: Path = Path("models/two_tower.pt"),
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        self.patience = patience
        self.model_path = model_path
        self._best_val_loss = float("inf")
        self._patience_counter = 0

    def _step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run a single forward pass and return loss."""
        user_idx = batch["user_idx"].to(self.device)
        item_idx = batch["item_idx"].to(self.device)
        labels = batch["label"].to(self.device)
        item_features = batch.get("item_features")
        if item_features is not None:
            item_features = item_features.to(self.device)

        scores = self.model(user_idx, item_idx, item_features)
        return self.criterion(scores, labels)

    def train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch and return mean loss."""
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            self.optimizer.zero_grad()
            loss = self._step(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> float:
        """Evaluate on validation set and return mean loss."""
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            loss = self._step(batch)
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> list[dict[str, float]]:
        """
        Train the model with early stopping.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Maximum epochs.

        Returns:
            List of per-epoch dicts with 'train_loss' and 'val_loss'.
        """
        history = []
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            logger.info(
                "Epoch %d/%d — train_loss: %.4f  val_loss: %.4f",
                epoch, epochs, train_loss, val_loss,
            )

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                self._save_checkpoint()
                logger.info("  → New best val_loss. Checkpoint saved.")
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

        # Restore best checkpoint
        self._load_checkpoint()
        return history

    def _save_checkpoint(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)

    def _load_checkpoint(self) -> None:
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))

    @torch.no_grad()
    def build_item_embeddings(
        self,
        n_items: int,
        item_features: np.ndarray | None = None,
        batch_size: int = 2048,
    ) -> np.ndarray:
        """
        Pre-compute item tower embeddings for all items.

        Args:
            n_items: Total number of items.
            item_features: Optional metadata features array (n_items, n_meta).
            batch_size: Inference batch size.

        Returns:
            Numpy array of shape (n_items, output_dim).
        """
        self.model.eval()
        dataset = InferenceDataset(n_items, item_features)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_embeddings = []
        for batch in loader:
            item_idx = batch["item_idx"].to(self.device)
            feat = batch.get("item_features")
            if feat is not None:
                feat = feat.to(self.device)
            emb = self.model.item_tower(item_idx, feat)
            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)

    @torch.no_grad()
    def get_user_embedding(
        self,
        user_idx: int,
    ) -> np.ndarray:
        """
        Compute the user embedding for a single user.

        Args:
            user_idx: Encoded user integer index.

        Returns:
            Numpy array of shape (output_dim,).
        """
        self.model.eval()
        uid_tensor = torch.LongTensor([user_idx]).to(self.device)
        emb = self.model.user_tower(uid_tensor)
        return emb.squeeze(0).cpu().numpy()

    def recommend(
        self,
        user_idx: int,
        item_embeddings: np.ndarray,
        top_k: int = 10,
        exclude_items: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Recommend top-K items for a user using pre-computed item embeddings.

        Args:
            user_idx: Encoded user index.
            item_embeddings: Pre-computed (n_items, dim) item embeddings.
            top_k: Number of recommendations.
            exclude_items: Item indices to exclude (e.g., already seen).

        Returns:
            List of (item_idx, score) sorted descending.
        """
        user_emb = self.get_user_embedding(user_idx)
        scores = item_embeddings @ user_emb          # (n_items,)

        if exclude_items:
            for iid in exclude_items:
                if 0 <= iid < len(scores):
                    scores[iid] = -np.inf

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(int(i), float(scores[i])) for i in top_indices]


def build_item_feature_matrix(
    item_features: "pd.DataFrame",  # noqa: F821
    meta_cols: list[str],
) -> np.ndarray:
    """
    Build a dense numpy array of item metadata features for the item tower.

    Args:
        item_features: Item features DataFrame sorted by item_idx.
        meta_cols: Columns to include as continuous features.

    Returns:
        Float32 array of shape (n_items, len(meta_cols)).
    """
    import pandas as pd
    df = item_features.sort_values("item_idx")
    available = [c for c in meta_cols if c in df.columns]
    arr = df[available].fillna(0.0).values.astype(np.float32)
    return arr
