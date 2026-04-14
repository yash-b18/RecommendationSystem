"""
Upload model and data artifacts to a HuggingFace Hub model repository.

Run this ONCE from your local machine before deploying the Space.

Usage:
    HF_REPO_ID=yourname/deepreads-artifacts python scripts/upload_artifacts.py

Environment variables:
    HF_REPO_ID   — target HF model repo (will be created if it doesn't exist)
    HF_TOKEN     — your HF write token (from https://huggingface.co/settings/tokens)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_MAP: list[tuple[Path, str]] = [
    # (local_path, repo_filename)
    # ── Models ──────────────────────────────────────────────────────────────
    (ROOT / "models" / "two_tower.pt",          "models/two_tower.pt"),
    (ROOT / "models" / "item_embeddings.npy",   "models/item_embeddings.npy"),
    (ROOT / "models" / "item_index.pkl",        "models/item_index.pkl"),
    (ROOT / "models" / "naive_baseline.pkl",    "models/naive_baseline.pkl"),
    (ROOT / "models" / "naive_category.pkl",    "models/naive_category.pkl"),
    (ROOT / "models" / "lgbm_model.pkl",        "models/lgbm_model.pkl"),
    # ── Processed data ──────────────────────────────────────────────────────
    (ROOT / "data" / "processed" / "item_features.parquet",  "data/processed/item_features.parquet"),
    (ROOT / "data" / "processed" / "user_features.parquet",  "data/processed/user_features.parquet"),
    (ROOT / "data" / "processed" / "train.parquet",          "data/processed/train.parquet"),
    (ROOT / "data" / "processed" / "feature_artifacts.pkl",  "data/processed/feature_artifacts.pkl"),
    (ROOT / "data" / "processed" / "item_encoder.pkl",       "data/processed/item_encoder.pkl"),
    (ROOT / "data" / "processed" / "user_encoder.pkl",       "data/processed/user_encoder.pkl"),
]


def main() -> None:
    repo_id = os.environ.get("HF_REPO_ID", "")
    if not repo_id:
        print("ERROR: Set HF_REPO_ID, e.g.:")
        print("  export HF_REPO_ID=yourname/deepreads-artifacts")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN") or None
    if not token:
        print("WARNING: HF_TOKEN not set. Will try using cached credentials.")
        print("  Get a write token at https://huggingface.co/settings/tokens")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface-hub")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        print(f"Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"ERROR creating repo: {e}")
        sys.exit(1)

    missing = [(local, repo_fn) for local, repo_fn in ARTIFACT_MAP if not local.exists()]
    if missing:
        print(f"WARNING: {len(missing)} local file(s) not found and will be skipped:")
        for local, _ in missing:
            print(f"  {local}")

    to_upload = [(local, repo_fn) for local, repo_fn in ARTIFACT_MAP if local.exists()]
    total = len(to_upload)
    print(f"\nUploading {total} file(s) to {repo_id} …\n")

    for i, (local_path, repo_filename) in enumerate(to_upload, 1):
        size_mb = local_path.stat().st_size / 1_048_576
        print(f"  [{i}/{total}] {repo_filename}  ({size_mb:.1f} MB)")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_filename,
                repo_id=repo_id,
                repo_type="model",
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            sys.exit(1)

    print(f"\nDone! All artifacts uploaded to:")
    print(f"  https://huggingface.co/{repo_id}")
    print(f"\nNext: set HF_REPO_ID={repo_id} as a Space secret.")


if __name__ == "__main__":
    main()
