"""
Dataset downloader for Amazon Reviews 2023.

Downloads the Video Games (or configured category) reviews and metadata
from HuggingFace using the `datasets` library, then saves raw JSONL files
to data/raw/.

Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from datasets import load_dataset

from src.config import DataConfig, RAW_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _hf_token() -> str | None:
    """Return HF token from environment if set."""
    return os.getenv("HF_TOKEN") or None


def _save_jsonl(records: Iterator[dict], path: Path, max_rows: int | None = None) -> int:
    """
    Stream records and save as JSONL.

    Args:
        records: Iterable of dicts.
        path: Output file path.
        max_rows: If set, stop after this many rows (for debug mode).

    Returns:
        Number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, default=str) + "\n")
            count += 1
            if max_rows and count >= max_rows:
                break
            if count % 50_000 == 0:
                logger.info("  written %d rows…", count)
    return count


def download_reviews(cfg: DataConfig) -> Path:
    """
    Download review interactions for the configured category.

    Args:
        cfg: DataConfig with category and path settings.

    Returns:
        Path to the saved JSONL file.
    """
    split_name = f"raw_review_{cfg.category}"
    logger.info("Downloading reviews: %s / %s", cfg.hf_dataset_name, split_name)

    # Streaming avoids loading the full dataset into memory
    ds = load_dataset(
        cfg.hf_dataset_name,
        split_name,
        split="full",
        streaming=True,
        token=_hf_token(),
        trust_remote_code=True,
    )

    max_rows = 200_000 if cfg.debug else None
    n = _save_jsonl(iter(ds), cfg.raw_reviews_path, max_rows=max_rows)
    logger.info("Saved %d reviews → %s", n, cfg.raw_reviews_path)
    return cfg.raw_reviews_path


def download_metadata(cfg: DataConfig) -> Path:
    """
    Download item metadata for the configured category.

    Args:
        cfg: DataConfig with category and path settings.

    Returns:
        Path to the saved JSONL file.
    """
    split_name = f"raw_meta_{cfg.category}"
    logger.info("Downloading metadata: %s / %s", cfg.hf_dataset_name, split_name)

    ds = load_dataset(
        cfg.hf_dataset_name,
        split_name,
        split="full",
        streaming=True,
        token=_hf_token(),
        trust_remote_code=True,
    )

    max_rows = 50_000 if cfg.debug else None
    n = _save_jsonl(iter(ds), cfg.raw_metadata_path, max_rows=max_rows)
    logger.info("Saved %d metadata rows → %s", n, cfg.raw_metadata_path)
    return cfg.raw_metadata_path
