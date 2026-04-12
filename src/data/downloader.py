"""
Dataset downloader for Amazon Reviews 2023.

Streams JSONL files directly from HuggingFace Hub via HfFileSystem,
bypassing the deprecated custom loading-script API.

File layout on Hub:
  reviews : datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{Category}.jsonl
  metadata: datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_{Category}.jsonl

Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import HfFileSystem

from src.config import DataConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

_HF_BASE = "datasets/McAuley-Lab/Amazon-Reviews-2023/raw"


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or None


def _stream_jsonl(
    fs: HfFileSystem,
    hf_path: str,
    output_path: Path,
    max_rows: int | None,
) -> int:
    """
    Stream a JSONL file from HuggingFace and write it locally.

    Args:
        fs: Authenticated HfFileSystem.
        hf_path: Remote path on HuggingFace.
        output_path: Local destination.
        max_rows: Stop after this many rows (debug mode). None = all.

    Returns:
        Number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with fs.open(hf_path, "r", encoding="utf-8") as src, \
         output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            dst.write(line + "\n")
            count += 1
            if count % 50_000 == 0:
                logger.info("  written %d rows…", count)
            if max_rows and count >= max_rows:
                break
    return count


def download_reviews(cfg: DataConfig) -> Path:
    """
    Download review interactions for the configured category.

    Args:
        cfg: DataConfig with category and path settings.

    Returns:
        Path to the saved JSONL file.
    """
    hf_path = f"{_HF_BASE}/review_categories/{cfg.category}.jsonl"

    if cfg.raw_reviews_path.exists():
        logger.info("Reviews already downloaded → %s (skipping)", cfg.raw_reviews_path)
        return cfg.raw_reviews_path

    logger.info("Downloading reviews from %s", hf_path)
    fs = HfFileSystem(token=_hf_token())
    max_rows = 5_000_000 if cfg.debug else None
    n = _stream_jsonl(fs, hf_path, cfg.raw_reviews_path, max_rows)
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
    hf_path = f"{_HF_BASE}/meta_categories/meta_{cfg.category}.jsonl"

    if cfg.raw_metadata_path.exists():
        logger.info("Metadata already downloaded → %s (skipping)", cfg.raw_metadata_path)
        return cfg.raw_metadata_path

    logger.info("Downloading metadata from %s", hf_path)
    fs = HfFileSystem(token=_hf_token())
    max_rows = 2_000_000 if cfg.debug else None
    n = _stream_jsonl(fs, hf_path, cfg.raw_metadata_path, max_rows)
    logger.info("Saved %d metadata rows → %s", n, cfg.raw_metadata_path)
    return cfg.raw_metadata_path
