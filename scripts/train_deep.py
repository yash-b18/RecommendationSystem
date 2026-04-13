"""
Script: train_deep.py

Trains the Two-Tower neural recommender and pre-computes item embeddings.

Outputs:
  models/two_tower.pt              (best checkpoint)
  models/item_embeddings.npy       (pre-computed item tower outputs)
  models/item_index.pkl            (item_idx → item metadata lookup)
  data/outputs/figures/training_curve.png

Usage:
    python scripts/train_deep.py
    python scripts/train_deep.py --debug
    python scripts/train_deep.py --device cuda
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import FIGURES_DIR, MODELS_DIR, ensure_dirs, get_config
from src.data.dataset import BPRDataset
from src.models.deep import TwoTowerModel, TwoTowerTrainer, build_item_feature_matrix
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

# Metadata features passed to the item tower
ITEM_META_COLS = [
    "item_avg_rating", "item_num_ratings", "item_popularity",
    "item_positive_rate", "price_norm", "category_idx", "brand_idx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Two-Tower neural recommender.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu', 'cuda', 'mps'. Auto-detected if not set.")
    return parser.parse_args()


def detect_device(override: str | None) -> str:
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def plot_training_curve(history: list[dict], output_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train loss", linewidth=2, color="#3b82f6")
    ax.plot(epochs, val_loss, label="Val loss", linewidth=2, color="#f97316", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BPR Loss")
    ax.set_title("Two-Tower Training Curve (BPR)")
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Training curve saved → %s", output_path)


def main() -> None:
    args = parse_args()
    cfg = get_config(debug=args.debug)
    ensure_dirs(cfg)
    set_seed(cfg.data.random_seed)

    device = detect_device(args.device)
    logger.info("Using device: %s", device)

    logger.info("Loading data…")
    train_df = pd.read_parquet(cfg.data.train_path)
    val_df = pd.read_parquet(cfg.data.val_path)
    item_features = pd.read_parquet(cfg.data.item_features_path)

    # Ensure label column
    if "label" not in train_df.columns:
        train_df["label"] = (train_df["rating"] >= cfg.data.positive_rating_threshold).astype(int)
    if "label" not in val_df.columns:
        val_df["label"] = (val_df["rating"] >= cfg.data.positive_rating_threshold).astype(int)

    n_users = int(train_df["user_idx"].max() + 1)
    n_items = int(item_features["item_idx"].max() + 1)
    logger.info("Dataset: %d users, %d items", n_users, n_items)

    # Build item metadata feature matrix
    item_feat_matrix = build_item_feature_matrix(item_features, ITEM_META_COLS)
    n_meta = item_feat_matrix.shape[1]
    logger.info("Item metadata features: %d", n_meta)

    # BPR Datasets: pairs each positive with a sampled negative
    # train_df has pre-sampled negatives (label=0); val_df may only have positives
    train_ds = BPRDataset(train_df, item_feat_matrix)
    val_ds = BPRDataset(val_df, item_feat_matrix, n_items=n_items)
    logger.info("BPR dataset — train pairs: %d, val pairs: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.deep.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.deep.batch_size,
        shuffle=False,
        num_workers=0,
    )

    logger.info("Building Two-Tower model…")
    model = TwoTowerModel(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=cfg.deep.embedding_dim,
        output_dim=cfg.deep.embedding_dim,
        hidden_dims=cfg.deep.hidden_dims,
        dropout=cfg.deep.dropout,
        n_meta_features=n_meta,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", total_params)

    trainer = TwoTowerTrainer(
        model=model,
        device=device,
        lr=cfg.deep.learning_rate,
        weight_decay=cfg.deep.weight_decay,
        patience=cfg.deep.patience,
        model_path=cfg.deep.model_path,
    )

    logger.info("Training for up to %d epochs…", cfg.deep.epochs)
    history = trainer.fit(train_loader, val_loader, epochs=cfg.deep.epochs)

    # Plot training curve
    plot_training_curve(history, FIGURES_DIR / "training_curve.png")

    # Pre-compute and save item embeddings
    logger.info("Pre-computing item embeddings…")
    item_embeddings = trainer.build_item_embeddings(
        n_items=n_items,
        item_features=item_feat_matrix,
        batch_size=2048,
    )
    np.save(cfg.deep.item_embeddings_path, item_embeddings)
    logger.info("Item embeddings saved → %s  shape: %s",
                cfg.deep.item_embeddings_path, item_embeddings.shape)

    # Save item index (item_idx → item_id + metadata) for explanation lookups
    item_index = item_features.set_index("item_idx")[["title", "category", "brand", "price"]].to_dict(orient="index")
    with open(cfg.deep.item_index_path, "wb") as f:
        pickle.dump(item_index, f)
    logger.info("Item index saved → %s", cfg.deep.item_index_path)

    logger.info("Two-Tower training complete.")


if __name__ == "__main__":
    main()
