"""
Download model and data artifacts from HuggingFace Hub at startup.

Run once before starting the API server. Files are cached locally so
subsequent restarts skip the download entirely.

Usage:
    python scripts/download_artifacts.py

Environment variables:
    HF_REPO_ID   — HuggingFace model repo, e.g. "yashb18/deepreads-artifacts"
    HF_TOKEN     — Optional HF token (only needed if repo is private)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Artifact manifest
# Each entry: (repo_filename, local_path)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_MAP: list[tuple[str, Path]] = [
    # ── Models ──────────────────────────────────────────────────────────────
    ("models/two_tower.pt",          ROOT / "models" / "two_tower.pt"),
    ("models/item_embeddings.npy",   ROOT / "models" / "item_embeddings.npy"),
    ("models/item_index.pkl",        ROOT / "models" / "item_index.pkl"),
    ("models/naive_baseline.pkl",    ROOT / "models" / "naive_baseline.pkl"),
    ("models/naive_category.pkl",    ROOT / "models" / "naive_category.pkl"),
    ("models/lgbm_model.pkl",        ROOT / "models" / "lgbm_model.pkl"),
    # ── Processed data ──────────────────────────────────────────────────────
    ("data/processed/item_features.parquet",  ROOT / "data" / "processed" / "item_features.parquet"),
    ("data/processed/user_features.parquet",  ROOT / "data" / "processed" / "user_features.parquet"),
    ("data/processed/train.parquet",          ROOT / "data" / "processed" / "train.parquet"),
    ("data/processed/feature_artifacts.pkl",  ROOT / "data" / "processed" / "feature_artifacts.pkl"),
    ("data/processed/item_encoder.pkl",       ROOT / "data" / "processed" / "item_encoder.pkl"),
    ("data/processed/user_encoder.pkl",       ROOT / "data" / "processed" / "user_encoder.pkl"),
]


def main() -> None:
    repo_id = os.environ.get("HF_REPO_ID", "")
    if not repo_id:
        print("ERROR: HF_REPO_ID environment variable is not set.", file=sys.stderr)
        print("  Set it to your HuggingFace model repo, e.g.:", file=sys.stderr)
        print("    export HF_REPO_ID=yourname/deepreads-artifacts", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("HF_TOKEN") or None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface-hub", file=sys.stderr)
        sys.exit(1)

    missing = [(repo_fn, local) for repo_fn, local in ARTIFACT_MAP if not local.exists()]

    if not missing:
        print("All artifacts already present — skipping download.")
        return

    print(f"Downloading {len(missing)} artifact(s) from {repo_id} …")

    for repo_filename, local_path in missing:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  ↓  {repo_filename}")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=repo_filename,
                repo_type="model",
                token=token,
                local_dir=ROOT,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download places files under local_dir/<filename>
            # which matches local_path exactly, so nothing to move.
            _ = downloaded
        except Exception as exc:
            print(f"ERROR: Failed to download {repo_filename}: {exc}", file=sys.stderr)
            sys.exit(1)

    print("Download complete.")


if __name__ == "__main__":
    main()
