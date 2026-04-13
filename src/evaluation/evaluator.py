"""
Unified evaluator for all three recommendation models.

Given a test split and trained models, generates recommendations for each
test user and computes the full metric suite.

Also generates comparison plots saved to data/outputs/figures/.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.evaluation.metrics import compute_all_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


RecommendFn = Callable[[int], list[tuple[int, float]]]


def evaluate_model(
    recommend_fn: RecommendFn,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k_values: list[int] = (5, 10, 20),
    top_k: int = 20,
) -> dict[str, float]:
    """
    Evaluate a recommendation function on the test set.

    For each test user:
      - Ground truth = items in test_df for that user
      - Exclude items already seen in train_df

    Args:
        recommend_fn: Function (user_idx) → list of (item_idx, score).
        test_df: Test split with [user_idx, item_idx] (one positive per user).
        train_df: Training split to build seen-items exclusion sets.
        k_values: K cutoffs.
        top_k: Maximum K to retrieve from the recommender.

    Returns:
        Dict of metric names → mean values across users.
    """
    # Build per-user ground truths and seen sets
    gt_map: dict[int, list[int]] = (
        test_df.groupby("user_idx")["item_idx"]
        .apply(list)
        .to_dict()
    )

    all_recs: list[list[int]] = []
    all_gts: list[list[int]] = []

    test_users = list(gt_map.keys())
    for uid in test_users:
        try:
            recs = recommend_fn(uid)
            rec_items = [item_idx for item_idx, _ in recs]
        except Exception as exc:
            logger.warning("Recommendation failed for user %d: %s", uid, exc)
            rec_items = []

        all_recs.append(rec_items)
        all_gts.append(gt_map[uid])

    metrics = compute_all_metrics(all_recs, all_gts, k_values=list(k_values))
    return metrics


def evaluate_all_models(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    naive_recommend_fn: RecommendFn,
    classical_recommend_fn: RecommendFn,
    deep_recommend_fn: RecommendFn,
    k_values: list[int] = (5, 10, 20),
    top_k: int = 20,
    output_dir: Path = Path("data/outputs"),
) -> pd.DataFrame:
    """
    Evaluate all three models and produce comparison artifacts.

    Args:
        test_df: Test split.
        train_df: Training split.
        naive_recommend_fn: Callable for naive model.
        classical_recommend_fn: Callable for classical model.
        deep_recommend_fn: Callable for deep model.
        k_values: K cutoffs.
        top_k: Retrieval depth.
        output_dir: Directory to save metrics CSV and plots.

    Returns:
        DataFrame with one row per model and metric columns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_fns = {
        "Naive (Popularity)": naive_recommend_fn,
        "Classical (LightGBM)": classical_recommend_fn,
        "Deep (Two-Tower)": deep_recommend_fn,
    }

    rows = []
    for model_name, fn in model_fns.items():
        logger.info("Evaluating %s…", model_name)
        metrics = evaluate_model(fn, test_df, train_df, k_values=k_values, top_k=top_k)
        row = {"Model": model_name, **metrics}
        rows.append(row)
        for k, v in metrics.items():
            logger.info("  %s: %.4f", k, v)

    results_df = pd.DataFrame(rows)

    # Save CSV
    csv_path = output_dir / "metrics.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Metrics saved → %s", csv_path)

    # Save markdown table
    md_table = tabulate(
        results_df.round(4), headers="keys", tablefmt="github", showindex=False
    )
    md_path = output_dir / "metrics_table.md"
    md_path.write_text(f"# Evaluation Results\n\n{md_table}\n")
    logger.info("Metrics table saved → %s", md_path)

    # Plot NDCG@K comparison
    _plot_metric_comparison(results_df, k_values, output_dir / "figures" / "metrics_comparison.png")

    return results_df


def _plot_metric_comparison(
    results_df: pd.DataFrame,
    k_values: list[int],
    output_path: Path,
) -> None:
    """Plot Recall@K and NDCG@K bar charts for all models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = results_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25
    colors = ["#3b82f6", "#f97316", "#10b981"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric_prefix, ylabel in zip(
        axes,
        ["NDCG@", "Recall@"],
        ["NDCG@K", "Recall@K"],
    ):
        for i, k in enumerate(k_values):
            col = f"{metric_prefix}{k}"
            if col in results_df.columns:
                values = results_df[col].values
                bars = ax.bar(x + i * width, values, width, label=f"K={k}", color=colors[i % 3])
                for bar, v in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8,
                    )
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=10, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.set_ylim(0, 1)

    plt.suptitle("Model Comparison — Test Set Metrics", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Metrics comparison plot saved → %s", output_path)
