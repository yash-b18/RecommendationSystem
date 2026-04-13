"""
Script: run_experiment.py

Feature Ablation Experiment.

Compares LightGBM performance across four feature tiers:
  1. id_only              — encoded user/item IDs
  2. id_metadata          — + item stats, category, brand, price
  3. id_metadata_text     — + user-item title cosine similarity
  4. id_metadata_text_history — + user history aggregates (full set)

For each tier: train LightGBM → evaluate on test → record NDCG@10 and Recall@10.

Outputs:
  data/outputs/experiment/ablation_results.csv
  data/outputs/experiment/ablation_ndcg.png
  data/outputs/experiment/ablation_recall.png

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --debug
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, ensure_dirs, get_config
from src.evaluation.metrics import compute_all_metrics
from src.features.builder import FEATURE_SETS, build_feature_matrix
from src.features.text_features import build_user_text_profiles
from src.models.classical import LightGBMReranker
from src.models.naive import GlobalPopularityRecommender
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature ablation experiment.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--k", type=int, default=10, help="Primary K for the experiment table.")
    return parser.parse_args()


def train_and_evaluate_tier(
    feature_set: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    user_profiles: pd.DataFrame,
    vectorizer,
    naive_model: GlobalPopularityRecommender,
    cfg,
    candidate_pool_size: int = 100,
    k: int = 10,
) -> dict[str, float]:
    """Train a LightGBM model for a given feature tier and evaluate it."""
    logger.info("  Feature set: %s", feature_set)

    # Build feature matrices for this tier
    train_feat, feat_cols = build_feature_matrix(
        train_df, user_features, item_features, user_profiles, vectorizer,
        feature_set=feature_set,
    )
    val_feat, _ = build_feature_matrix(
        val_df, user_features, item_features, user_profiles, vectorizer,
        feature_set=feature_set,
    )

    # Train
    model = LightGBMReranker(
        n_estimators=200 if cfg.data.debug else 500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=cfg.data.random_seed,
        early_stopping_rounds=30,
    )
    model.fit(train_feat, val_feat, feat_cols)

    # Evaluate using candidate pool from naive model
    gt_map = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    all_recs: list[list[int]] = []
    all_gts: list[list[int]] = []

    for uid, gt in gt_map.items():
        candidates = naive_model.recommend(uid, top_k=candidate_pool_size)
        if not candidates:
            all_recs.append([])
            all_gts.append(gt)
            continue
        candidate_items = [iid for iid, _ in candidates]
        rows = pd.DataFrame({"user_idx": uid, "item_idx": candidate_items, "label": 0})
        feat, _ = build_feature_matrix(
            rows, user_features, item_features, user_profiles, vectorizer,
            feature_set=feature_set,
        )
        scores = model.predict_proba(feat)
        ranked = sorted(zip(candidate_items, scores.tolist()), key=lambda x: x[1], reverse=True)
        all_recs.append([i for i, _ in ranked[:k]])
        all_gts.append(gt)

    metrics = compute_all_metrics(all_recs, all_gts, k_values=[k])
    logger.info("    NDCG@%d=%.4f  Recall@%d=%.4f", k, metrics[f"NDCG@{k}"], k, metrics[f"Recall@{k}"])
    return metrics


def plot_ablation(results_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(results_df["feature_set"], results_df[metric], color="#3b82f6", edgecolor="white")
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Feature Set")
    ax.set_ylabel(metric)
    ax.set_title(f"Feature Ablation Experiment — {metric}")
    ax.set_ylim(0, results_df[metric].max() * 1.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Ablation plot saved → %s", output_path)


def main() -> None:
    args = parse_args()
    cfg = get_config(debug=args.debug)
    ensure_dirs(cfg)

    exp_dir = cfg.experiment.output_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data…")
    train_df = pd.read_parquet(cfg.data.train_path)
    val_df = pd.read_parquet(cfg.data.val_path)
    test_df = pd.read_parquet(cfg.data.test_path)
    item_features = pd.read_parquet(cfg.data.item_features_path)
    user_features = pd.read_parquet(cfg.data.user_features_path)

    for split in (val_df, test_df):
        if "label" not in split.columns:
            split["label"] = (split["rating"] >= cfg.data.positive_rating_threshold).astype(int)

    # Load feature artifacts
    with open(PROCESSED_DIR / "feature_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    vectorizer = artifacts["vectorizer"]

    user_profiles = build_user_text_profiles(train_df, item_features, cfg.data.positive_rating_threshold)

    # Load naive model for candidate generation
    naive_model = GlobalPopularityRecommender.load(MODELS_DIR / "naive_baseline.pkl")

    logger.info("=== Feature Ablation Experiment ===")
    rows = []
    for fset in FEATURE_SETS:
        metrics = train_and_evaluate_tier(
            feature_set=fset,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            user_features=user_features,
            item_features=item_features,
            user_profiles=user_profiles,
            vectorizer=vectorizer,
            naive_model=naive_model,
            cfg=cfg,
            candidate_pool_size=cfg.classical.candidate_pool_size,
            k=args.k,
        )
        rows.append({"feature_set": fset, **metrics})

    results_df = pd.DataFrame(rows)
    results_path = exp_dir / "ablation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Ablation results saved → %s", results_path)

    # Plots
    ndcg_col = f"NDCG@{args.k}"
    recall_col = f"Recall@{args.k}"
    if ndcg_col in results_df.columns:
        plot_ablation(results_df, ndcg_col, exp_dir / "ablation_ndcg.png")
    if recall_col in results_df.columns:
        plot_ablation(results_df, recall_col, exp_dir / "ablation_recall.png")

    print("\n=== Ablation Results ===")
    print(results_df.to_string(index=False))

    # Interpretation summary
    summary_path = exp_dir / "ablation_interpretation.md"
    best_row = results_df.loc[results_df[ndcg_col].idxmax()] if ndcg_col in results_df else None
    summary_lines = [
        "# Feature Ablation Experiment — Interpretation\n",
        f"Metric: {ndcg_col} on test set.\n",
        "## Results\n",
        results_df.to_markdown(index=False),
        "",
        "## Key Findings\n",
    ]
    if best_row is not None:
        summary_lines.append(
            f"- Best feature set: **{best_row['feature_set']}** "
            f"({ndcg_col} = {best_row[ndcg_col]:.4f})"
        )
    summary_path.write_text("\n".join(summary_lines))
    logger.info("Interpretation saved → %s", summary_path)
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
