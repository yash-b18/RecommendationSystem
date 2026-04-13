"""
Script: train_baseline.py

Trains and saves the naive baseline recommenders:
  1. GlobalPopularityRecommender
  2. CategoryPopularityRecommender

Outputs:
  models/naive_baseline.pkl   (GlobalPopularity)
  models/naive_category.pkl   (CategoryPopularity)

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import MODELS_DIR, ensure_dirs, get_config
from src.models.naive import CategoryPopularityRecommender, GlobalPopularityRecommender
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train naive baseline recommenders.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(debug=args.debug)
    ensure_dirs(cfg)

    logger.info("Loading training data…")
    train_df = pd.read_parquet(cfg.data.train_path)
    item_features = pd.read_parquet(cfg.data.item_features_path)
    user_features = pd.read_parquet(cfg.data.user_features_path)

    logger.info("Training GlobalPopularityRecommender…")
    global_model = GlobalPopularityRecommender()
    global_model.fit(train_df)
    global_model.save(MODELS_DIR / "naive_baseline.pkl")

    logger.info("Training CategoryPopularityRecommender…")
    cat_model = CategoryPopularityRecommender()
    cat_model.fit(
        train_df,
        item_features,
        user_features,
        positive_threshold=cfg.data.positive_rating_threshold,
    )
    cat_model.save(MODELS_DIR / "naive_category.pkl")

    logger.info("Naive baselines saved to %s", MODELS_DIR)


if __name__ == "__main__":
    main()
