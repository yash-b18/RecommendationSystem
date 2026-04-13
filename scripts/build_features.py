"""
Script: build_features.py

Runs the full feature engineering pipeline, producing:
  - data/processed/user_features.parquet
  - data/processed/item_features.parquet
  - data/processed/train_features.parquet
  - data/processed/val_features.parquet
  - data/processed/test_features.parquet
  - data/processed/feature_artifacts.pkl  (vectorizer, encoders)

Prerequisites: run make_dataset.py first.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ensure_dirs, get_config
from src.features.builder import run_feature_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature matrices for the recommendation models.")
    parser.add_argument("--debug", action="store_true", help="Use config for small dataset.")
    parser.add_argument("--category", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(debug=args.debug, category=args.category)
    ensure_dirs(cfg)

    if not cfg.data.train_path.exists():
        logger.error("Train split not found at %s. Run make_dataset.py first.", cfg.data.train_path)
        sys.exit(1)

    run_feature_pipeline(cfg.data)
    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    main()
