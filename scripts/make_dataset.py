"""
Script: make_dataset.py

Downloads the Amazon Reviews 2023 dataset from HuggingFace and
preprocesses it into clean parquet files ready for feature engineering.

Usage (inside activated venv):
    python scripts/make_dataset.py
    python scripts/make_dataset.py --debug        # fast subset
    python scripts/make_dataset.py --category Electronics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ensure_dirs, get_config
from src.data.downloader import download_metadata, download_reviews
from src.data.preprocessor import preprocess
from src.data.splitter import run_split
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess Amazon Reviews 2023.")
    parser.add_argument("--debug", action="store_true", help="Use a small data subset.")
    parser.add_argument("--category", type=str, default=None,
                        help="Amazon product category (default: from config).")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if raw files already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(debug=args.debug, category=args.category)
    ensure_dirs(cfg)

    logger.info("=== Stage 1: Download ===")
    if args.skip_download and cfg.raw_reviews_path.exists() and cfg.raw_metadata_path.exists():
        logger.info("Raw files already exist — skipping download.")
    else:
        download_reviews(cfg.data)
        download_metadata(cfg.data)

    logger.info("=== Stage 2: Preprocess ===")
    interactions, metadata = preprocess(cfg.data)

    logger.info("=== Stage 3: Split ===")
    run_split(cfg.data, interactions)

    logger.info("Dataset pipeline complete.")
    logger.info("  interactions : %s", cfg.data.interactions_path)
    logger.info("  metadata     : %s", cfg.data.metadata_path)
    logger.info("  train        : %s", cfg.data.train_path)
    logger.info("  val          : %s", cfg.data.val_path)
    logger.info("  test         : %s", cfg.data.test_path)


if __name__ == "__main__":
    main()
