"""
Script: error_analysis.py

Identifies poor recommendations (mispredictions) for the deep model and
exports a detailed CSV for report writing.

For each of the 5+ selected users:
  - User interaction history (titles of positively-rated items)
  - Model's predicted top-K items
  - Actual relevant item from test set
  - Model score for the relevant item
  - Rank of the relevant item in the predicted list
  - Suggested root cause

Output:
  data/outputs/error_analysis.csv

Usage:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --debug --n-examples 5
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, OUTPUTS_DIR, ensure_dirs, get_config
from src.models.deep import TwoTowerModel, TwoTowerTrainer, build_item_feature_matrix
from src.utils.logging import get_logger

logger = get_logger(__name__)

ITEM_META_COLS = [
    "item_avg_rating", "item_num_ratings", "item_popularity",
    "item_positive_rate", "price_norm", "category_idx", "brand_idx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export error analysis for the deep model.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n-examples", type=int, default=5,
                        help="Number of misprediction examples to export.")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def find_worst_users(
    test_df: pd.DataFrame,
    trainer: TwoTowerTrainer,
    item_embeddings: np.ndarray,
    user_seen: dict[int, set[int]],
    top_k: int,
    n_examples: int,
) -> list[dict]:
    """
    Find users where the model fails most (high rank of relevant item).

    Args:
        test_df: Test split (one positive per user).
        trainer: Fitted TwoTowerTrainer.
        item_embeddings: Pre-computed item embeddings.
        user_seen: Per-user seen item sets from training.
        top_k: Recommendation depth.
        n_examples: Number of worst-case examples to return.

    Returns:
        List of dicts with user analysis info.
    """
    gt_map = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    records = []

    for uid, gt_items in gt_map.items():
        try:
            user_emb = trainer.get_user_embedding(uid)
            scores = item_embeddings @ user_emb
            exclude = user_seen.get(uid, set())
            for i in exclude:
                if 0 <= i < len(scores):
                    scores[i] = -np.inf

            ranked_items = np.argsort(scores)[::-1]

            # Find rank of the ground-truth item
            gt_item = gt_items[0]
            rank = int(np.where(ranked_items == gt_item)[0][0]) + 1  # 1-indexed
            gt_score = float(scores[gt_item])
            top_k_items = ranked_items[:top_k].tolist()
            top_k_scores = scores[top_k_items].tolist()

            records.append({
                "user_idx": uid,
                "gt_item": gt_item,
                "gt_rank": rank,
                "gt_score": gt_score,
                "top_k_items": top_k_items,
                "top_k_scores": top_k_scores,
                "missed": gt_item not in top_k_items,
            })
        except Exception as exc:
            logger.warning("Error for user %d: %s", uid, exc)

    # Sort by rank descending (worst first)
    records.sort(key=lambda r: r["gt_rank"], reverse=True)
    return records[:n_examples]


def build_analysis_rows(
    records: list[dict],
    train_df: pd.DataFrame,
    item_features: pd.DataFrame,
    item_index: dict,
) -> pd.DataFrame:
    """
    Format misprediction records into a human-readable DataFrame.

    Args:
        records: Output of find_worst_users.
        train_df: Training interactions for user history.
        item_features: Item feature DataFrame.
        item_index: Dict item_idx → {title, category, brand}.

    Returns:
        DataFrame with one row per misprediction example.
    """
    iid_to_title = dict(zip(item_features["item_idx"], item_features.get("title", pd.Series())))
    iid_to_cat = dict(zip(item_features["item_idx"], item_features.get("category", pd.Series())))

    rows = []
    for i, rec in enumerate(records, start=1):
        uid = rec["user_idx"]

        # User history: titles of positively-rated items in training
        user_hist = train_df[
            (train_df["user_idx"] == uid) & (train_df.get("label", train_df["rating"]) >= 1)
        ]["item_idx"].tolist()
        history_titles = [str(iid_to_title.get(iid, f"item_{iid}")) for iid in user_hist[:5]]

        # Ground truth item
        gt_iid = rec["gt_item"]
        gt_title = str(iid_to_title.get(gt_iid, f"item_{gt_iid}"))
        gt_category = str(iid_to_cat.get(gt_iid, "Unknown"))

        # Top-K predicted items
        top_predicted = [
            str(iid_to_title.get(iid, f"item_{iid}"))
            for iid in rec["top_k_items"][:5]
        ]
        top_categories = [
            str(iid_to_cat.get(iid, "Unknown"))
            for iid in rec["top_k_items"][:5]
        ]

        # Simple root cause heuristic
        if rec["gt_rank"] > 100:
            root_cause = "Popularity bias: item is niche/low-interaction and overshadowed by popular items"
        elif gt_category not in top_categories:
            root_cause = f"Category mismatch: user history does not strongly signal '{gt_category}'"
        else:
            root_cause = "Embedding space does not capture fine-grained preference within category"

        rows.append({
            "example_id": i,
            "user_idx": uid,
            "user_history_sample": " | ".join(history_titles),
            "gt_item_title": gt_title,
            "gt_item_category": gt_category,
            "gt_rank_in_model": rec["gt_rank"],
            "gt_model_score": round(rec["gt_score"], 4),
            "top_5_predicted_titles": " | ".join(top_predicted),
            "top_5_predicted_categories": " | ".join(top_categories),
            "root_cause": root_cause,
            "proposed_mitigation": (
                "Increase diversity penalty in candidate generation; "
                "consider popularity-debiasing in training loss"
                if "popularity" in root_cause.lower()
                else "Add stronger category-affinity signal to item tower features"
            ),
        })

    return pd.DataFrame(rows)


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

    n_users = int(train_df["user_idx"].max() + 1)
    n_items = int(item_features["item_idx"].max() + 1)
    item_feat_matrix = build_item_feature_matrix(item_features, ITEM_META_COLS)
    n_meta = item_feat_matrix.shape[1]

    logger.info("Loading Two-Tower model…")
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

    item_embeddings = np.load(cfg.deep.item_embeddings_path)

    with open(cfg.deep.item_index_path, "rb") as f:
        item_index = pickle.load(f)

    user_seen = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    logger.info("Finding %d worst-case mispredictions…", args.n_examples)
    records = find_worst_users(
        test_df, trainer, item_embeddings, user_seen, args.top_k, args.n_examples
    )

    analysis_df = build_analysis_rows(records, train_df, item_features, item_index)

    output_path = OUTPUTS_DIR / "error_analysis.csv"
    analysis_df.to_csv(output_path, index=False)
    logger.info("Error analysis saved → %s", output_path)
    print(analysis_df[["example_id", "gt_item_title", "gt_rank_in_model", "root_cause"]].to_string(index=False))


if __name__ == "__main__":
    main()
