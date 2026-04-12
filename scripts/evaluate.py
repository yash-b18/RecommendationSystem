"""
Script: evaluate.py

Evaluates all three trained models on the test set and saves:
  - data/outputs/metrics.csv
  - data/outputs/metrics_table.md
  - data/outputs/figures/metrics_comparison.png

Prerequisites: all three models must be trained.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --debug
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, ensure_dirs, get_config
from src.evaluation.evaluator import evaluate_all_models
from src.features.builder import build_feature_matrix
from src.models.classical import LightGBMReranker
from src.models.deep import TwoTowerModel, TwoTowerTrainer, build_item_feature_matrix
from src.models.naive import GlobalPopularityRecommender
from src.utils.logging import get_logger

logger = get_logger(__name__)

ITEM_META_COLS = [
    "item_avg_rating", "item_num_ratings", "item_popularity",
    "item_positive_rate", "price_norm", "category_idx", "brand_idx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all recommendation models.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def load_deep_model(cfg, n_users: int, n_items: int, n_meta: int, device: str):
    """Load Two-Tower model and item embeddings."""
    import torch
    from src.models.deep import TwoTowerModel, TwoTowerTrainer

    model = TwoTowerModel(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=cfg.deep.embedding_dim,
        output_dim=cfg.deep.embedding_dim,
        hidden_dims=cfg.deep.hidden_dims,
        dropout=cfg.deep.dropout,
        n_meta_features=n_meta,
    )
    model.load_state_dict(torch.load(cfg.deep.model_path, map_location=device))
    trainer = TwoTowerTrainer(model=model, device=device, model_path=cfg.deep.model_path)
    return trainer


def main() -> None:
    args = parse_args()
    cfg = get_config(debug=args.debug)
    ensure_dirs(cfg)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading data…")
    train_df = pd.read_parquet(cfg.data.train_path)
    test_df = pd.read_parquet(cfg.data.test_path)
    item_features = pd.read_parquet(cfg.data.item_features_path)
    user_features = pd.read_parquet(cfg.data.user_features_path)

    # ---- Naive model ----
    logger.info("Loading naive model…")
    naive_model = GlobalPopularityRecommender.load(MODELS_DIR / "naive_baseline.pkl")
    naive_fn = lambda uid: naive_model.recommend(uid, top_k=args.top_k)

    # ---- Classical model ----
    logger.info("Loading LightGBM model…")
    lgbm_model = LightGBMReranker.load(MODELS_DIR / "lgbm_model.pkl")

    # Build test features for classical model
    artifacts_path = PROCESSED_DIR / "feature_artifacts.pkl"
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    # Pre-build a full test feature lookup: (user_idx, item_idx) → feature row
    # For evaluation we score each user's candidate pool from naive retrieval
    def classical_fn(uid: int) -> list[tuple[int, float]]:
        candidates = naive_model.recommend(uid, top_k=cfg.classical.candidate_pool_size)
        if not candidates:
            return []
        candidate_items = [iid for iid, _ in candidates]
        rows = pd.DataFrame({"user_idx": uid, "item_idx": candidate_items})
        rows["label"] = 0  # placeholder
        feat, _ = build_feature_matrix(
            rows, user_features, item_features,
            pd.DataFrame({"user_idx": [], "user_text_profile": []}),
            artifacts["vectorizer"],
            feature_set="id_metadata_text_history",
        )
        scores = lgbm_model.predict_proba(feat)
        ranked = sorted(zip(candidate_items, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:args.top_k]

    # ---- Deep model ----
    n_users = int(train_df["user_idx"].max() + 1)
    n_items = int(item_features["item_idx"].max() + 1)
    item_feat_matrix = build_item_feature_matrix(item_features, ITEM_META_COLS)
    n_meta = item_feat_matrix.shape[1]

    logger.info("Loading Two-Tower model…")
    trainer = load_deep_model(cfg, n_users, n_items, n_meta, device)
    item_embeddings = np.load(cfg.deep.item_embeddings_path)

    user_seen: dict[int, set[int]] = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    def deep_fn(uid: int) -> list[tuple[int, float]]:
        seen = user_seen.get(uid, set())
        return trainer.recommend(uid, item_embeddings, top_k=args.top_k, exclude_items=seen)

    # ---- Evaluate ----
    logger.info("Running evaluation on %d test users…", test_df["user_idx"].nunique())
    results = evaluate_all_models(
        test_df=test_df,
        train_df=train_df,
        naive_recommend_fn=naive_fn,
        classical_recommend_fn=classical_fn,
        deep_recommend_fn=deep_fn,
        k_values=cfg.eval.k_values,
        top_k=args.top_k,
        output_dir=OUTPUTS_DIR,
    )

    print("\n" + results.to_string(index=False))
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
