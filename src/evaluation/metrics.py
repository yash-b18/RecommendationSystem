"""
Recommendation evaluation metrics.

All metrics work on ranked lists:
  - Recall@K
  - Hit Rate@K  (1 if any relevant item in top-K)
  - NDCG@K      (Normalised Discounted Cumulative Gain)
  - MRR         (Mean Reciprocal Rank)

Each function accepts per-user lists of recommendations and ground truths
and returns the mean over all users.
"""

from __future__ import annotations

import numpy as np


def recall_at_k(
    recommendations: list[list[int]],
    ground_truths: list[list[int]],
    k: int,
) -> float:
    """
    Compute mean Recall@K across all users.

    Recall@K = |relevant ∩ top-K| / |relevant|

    Args:
        recommendations: Ranked list of item indices per user (descending relevance).
        ground_truths: List of relevant item indices per user.
        k: Cutoff.

    Returns:
        Mean Recall@K (float in [0, 1]).
    """
    scores = []
    for recs, gt in zip(recommendations, ground_truths):
        if not gt:
            continue
        hits = len(set(recs[:k]) & set(gt))
        scores.append(hits / len(gt))
    return float(np.mean(scores)) if scores else 0.0


def hit_rate_at_k(
    recommendations: list[list[int]],
    ground_truths: list[list[int]],
    k: int,
) -> float:
    """
    Compute mean Hit Rate@K (= Recall@K with |relevant|=1).

    Hit Rate@K = 1 if any relevant item appears in top-K recommendations, else 0.

    Args:
        recommendations: Ranked item lists per user.
        ground_truths: Relevant item lists per user.
        k: Cutoff.

    Returns:
        Mean Hit Rate@K (float in [0, 1]).
    """
    scores = []
    for recs, gt in zip(recommendations, ground_truths):
        if not gt:
            continue
        hit = int(bool(set(recs[:k]) & set(gt)))
        scores.append(hit)
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(
    recommendations: list[list[int]],
    ground_truths: list[list[int]],
    k: int,
) -> float:
    """
    Compute mean NDCG@K across all users.

    NDCG@K uses binary relevance (1 if item in ground truth, else 0).

    Args:
        recommendations: Ranked item lists per user.
        ground_truths: Relevant item lists per user.
        k: Cutoff.

    Returns:
        Mean NDCG@K (float in [0, 1]).
    """
    def dcg(hits: list[int], k: int) -> float:
        return sum(h / np.log2(i + 2) for i, h in enumerate(hits[:k]))

    scores = []
    for recs, gt in zip(recommendations, ground_truths):
        if not gt:
            continue
        gt_set = set(gt)
        gains = [1 if item in gt_set else 0 for item in recs[:k]]
        ideal = sorted(gains, reverse=True)
        idcg = dcg(ideal, k)
        if idcg == 0:
            scores.append(0.0)
        else:
            scores.append(dcg(gains, k) / idcg)
    return float(np.mean(scores)) if scores else 0.0


def mean_reciprocal_rank(
    recommendations: list[list[int]],
    ground_truths: list[list[int]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR = mean(1 / rank_of_first_relevant_item)
    If no relevant item is found, reciprocal rank = 0.

    Args:
        recommendations: Ranked item lists per user.
        ground_truths: Relevant item lists per user.

    Returns:
        MRR (float in [0, 1]).
    """
    scores = []
    for recs, gt in zip(recommendations, ground_truths):
        if not gt:
            continue
        gt_set = set(gt)
        rr = 0.0
        for rank, item in enumerate(recs, start=1):
            if item in gt_set:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return float(np.mean(scores)) if scores else 0.0


def compute_all_metrics(
    recommendations: list[list[int]],
    ground_truths: list[list[int]],
    k_values: list[int] = (5, 10, 20),
) -> dict[str, float]:
    """
    Compute the full metric suite for a set of recommendations.

    Args:
        recommendations: Ranked item lists per user.
        ground_truths: Relevant item lists per user.
        k_values: List of K cutoffs to evaluate.

    Returns:
        Dict mapping metric names to float values, e.g.:
        {"Recall@10": 0.23, "HitRate@10": 0.41, "NDCG@10": 0.19, "MRR": 0.28}
    """
    results: dict[str, float] = {}
    for k in k_values:
        results[f"Recall@{k}"] = recall_at_k(recommendations, ground_truths, k)
        results[f"HitRate@{k}"] = hit_rate_at_k(recommendations, ground_truths, k)
        results[f"NDCG@{k}"] = ndcg_at_k(recommendations, ground_truths, k)
    results["MRR"] = mean_reciprocal_rank(recommendations, ground_truths)
    return results
