"""
Script: train_classical.py

Trains the LightGBM reranker on the full feature set and saves the model.

Outputs:
  models/lgbm_model.pkl
  data/outputs/figures/feature_importance.png

Usage:
    python scripts/train_classical.py
    python scripts/train_classical.py --debug
    python scripts/train_classical.py --feature-set id_metadata   # ablation tier
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, ensure_dirs, get_config
from src.features.builder import FEATURE_SETS, build_feature_matrix
from src.models.classical import LightGBMReranker
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM reranker.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--feature-set",
        type=str,
        default="id_metadata_text_history",
        choices=FEATURE_SETS,
        help="Feature tier to use for training.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="lgbm_model",
        help="Stem of the output .pkl filename (in models/).",
    )
    return parser.parse_args()


def plot_feature_importance(model: LightGBMReranker, output_path: Path) -> None:
    """Save a horizontal bar chart of feature importances."""
    imp = model.feature_importance()
    top = imp.head(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#3b82f6")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("LightGBM Feature Importance (Top 20)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved → %s", output_path)


def main() -> None:
    args = parse_args()
    cfg = get_config(debug=args.debug)
    ensure_dirs(cfg)

    logger.info("Loading pre-built feature matrices…")
    # Try to load pre-built features; fall back to rebuilding
    if cfg.data.train_features_path.exists():
        train_feat = pd.read_parquet(cfg.data.train_features_path)
        val_feat = pd.read_parquet(cfg.data.val_features_path)
    else:
        logger.warning("Pre-built features not found. Loading raw splits and rebuilding…")
        train_df = pd.read_parquet(cfg.data.train_path)
        val_df = pd.read_parquet(cfg.data.val_path)
        user_features = pd.read_parquet(cfg.data.user_features_path)
        item_features = pd.read_parquet(cfg.data.item_features_path)

        artifacts_path = PROCESSED_DIR / "feature_artifacts.pkl"
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)

        user_profiles_path = PROCESSED_DIR / "user_profiles.parquet"
        if user_profiles_path.exists():
            user_profiles = pd.read_parquet(user_profiles_path)
        else:
            from src.features.text_features import build_user_text_profiles
            user_profiles = build_user_text_profiles(train_df, item_features)

        train_feat, _ = build_feature_matrix(
            train_df, user_features, item_features, user_profiles,
            artifacts["vectorizer"], feature_set=args.feature_set,
        )
        val_feat, _ = build_feature_matrix(
            val_df, user_features, item_features, user_profiles,
            artifacts["vectorizer"], feature_set=args.feature_set,
        )

    # Determine feature columns (everything except 'label' and index cols)
    exclude = {"label", "user_idx", "item_idx", "user_id", "item_id", "rating", "timestamp"}
    feature_cols = [c for c in train_feat.columns if c not in exclude]

    logger.info("Training LightGBM on %d features, %d rows…", len(feature_cols), len(train_feat))
    model = LightGBMReranker(
        n_estimators=cfg.classical.n_estimators,
        learning_rate=cfg.classical.learning_rate,
        num_leaves=cfg.classical.num_leaves,
        max_depth=cfg.classical.max_depth,
        min_child_samples=cfg.classical.min_child_samples,
        subsample=cfg.classical.subsample,
        colsample_bytree=cfg.classical.colsample_bytree,
        reg_alpha=cfg.classical.reg_alpha,
        reg_lambda=cfg.classical.reg_lambda,
        n_jobs=cfg.classical.n_jobs,
        random_state=cfg.data.random_seed,
        early_stopping_rounds=cfg.classical.early_stopping_rounds,
    )
    model.fit(train_feat, val_feat, feature_cols)

    # Save model
    model_path = MODELS_DIR / f"{args.output_name}.pkl"
    model.save(model_path)

    # Feature importance plot
    plot_feature_importance(model, FIGURES_DIR / "feature_importance.png")

    logger.info("LightGBM training complete.")


if __name__ == "__main__":
    main()
