"""
Central configuration for the recommendation system project.

All paths, hyperparameters, and dataset settings are defined here.
Scripts should import from this module rather than hardcode values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset download and processing settings."""

    # HuggingFace dataset identifier
    hf_dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023"

    # Category to use. Smaller categories are faster for development.
    # Options: "Video_Games", "Books", "Electronics", "Clothing_Shoes_and_Jewelry"
    category: str = "Books"

    # Minimum number of reviews a user must have to be included
    min_user_reviews: int = 5

    # Minimum number of reviews an item must have to be included
    min_item_reviews: int = 5

    # Threshold above which a rating is considered a positive interaction
    positive_rating_threshold: float = 4.0

    # Number of negative samples per positive interaction
    negative_samples_per_positive: int = 4

    # Fraction of data for validation and test splits
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    # Random seed for reproducibility
    random_seed: int = 42

    # Debug mode: use a tiny subset for fast iteration
    debug: bool = False
    debug_n_users: int = 500

    # Paths (resolved at runtime)
    raw_reviews_path: Path = field(default_factory=lambda: RAW_DIR / "reviews.jsonl")
    raw_metadata_path: Path = field(default_factory=lambda: RAW_DIR / "metadata.jsonl")
    interactions_path: Path = field(default_factory=lambda: PROCESSED_DIR / "interactions.parquet")
    metadata_path: Path = field(default_factory=lambda: PROCESSED_DIR / "metadata.parquet")
    train_path: Path = field(default_factory=lambda: PROCESSED_DIR / "train.parquet")
    val_path: Path = field(default_factory=lambda: PROCESSED_DIR / "val.parquet")
    test_path: Path = field(default_factory=lambda: PROCESSED_DIR / "test.parquet")
    user_features_path: Path = field(default_factory=lambda: PROCESSED_DIR / "user_features.parquet")
    item_features_path: Path = field(default_factory=lambda: PROCESSED_DIR / "item_features.parquet")
    train_features_path: Path = field(default_factory=lambda: PROCESSED_DIR / "train_features.parquet")
    val_features_path: Path = field(default_factory=lambda: PROCESSED_DIR / "val_features.parquet")
    test_features_path: Path = field(default_factory=lambda: PROCESSED_DIR / "test_features.parquet")
    user_encoder_path: Path = field(default_factory=lambda: PROCESSED_DIR / "user_encoder.pkl")
    item_encoder_path: Path = field(default_factory=lambda: PROCESSED_DIR / "item_encoder.pkl")


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

@dataclass
class NaiveConfig:
    """Configuration for the naive baseline models."""
    top_k: int = 10
    model_path: Path = field(default_factory=lambda: MODELS_DIR / "naive_baseline.pkl")


@dataclass
class ClassicalConfig:
    """Configuration for the LightGBM classical model."""

    # LightGBM hyperparameters
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    n_jobs: int = -1
    random_state: int = 42

    # Training
    early_stopping_rounds: int = 50
    verbose_eval: int = 100

    # Top-K candidates to score per user (from naive retrieval)
    candidate_pool_size: int = 500

    # Model artifact
    model_path: Path = field(default_factory=lambda: MODELS_DIR / "lgbm_model.pkl")
    feature_names_path: Path = field(default_factory=lambda: MODELS_DIR / "lgbm_feature_names.pkl")
    shap_values_path: Path = field(default_factory=lambda: OUTPUTS_DIR / "shap_values.npy")


@dataclass
class DeepConfig:
    """Configuration for the Two-Tower neural recommender."""

    # Architecture
    # hidden_dims=[] → single linear projection (no hidden layers), avoids MLP collapse
    # on sparse book data. Deep MLPs converge to a constant function with sparse gradients.
    embedding_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [])
    dropout: float = 0.1
    use_metadata: bool = True
    use_text: bool = True

    # Text encoding
    text_embedding_dim: int = 64
    max_title_tokens: int = 32

    # Training
    batch_size: int = 1024
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 4
    patience: int = 8
    random_seed: int = 42

    # Inference
    top_k: int = 10
    candidate_pool_size: int = 200

    # Model artifacts
    model_path: Path = field(default_factory=lambda: MODELS_DIR / "two_tower.pt")
    item_embeddings_path: Path = field(default_factory=lambda: MODELS_DIR / "item_embeddings.npy")
    item_index_path: Path = field(default_factory=lambda: MODELS_DIR / "item_index.pkl")


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Evaluation settings."""
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    primary_k: int = 10
    metrics_output_path: Path = field(default_factory=lambda: OUTPUTS_DIR / "metrics.csv")
    metrics_table_path: Path = field(default_factory=lambda: OUTPUTS_DIR / "metrics_table.md")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Feature ablation experiment settings."""

    experiment_name: str = "feature_ablation"

    # Feature sets to compare in ablation
    ablation_sets: List[str] = field(default_factory=lambda: [
        "id_only",
        "id_metadata",
        "id_metadata_text",
        "id_metadata_text_history",
    ])

    output_dir: Path = field(default_factory=lambda: OUTPUTS_DIR / "experiment")


# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

@dataclass
class APIConfig:
    """FastAPI backend configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    top_k: int = 10
    default_model: str = "deep"   # "naive" | "classical" | "deep"
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "https://*.vercel.app",
    ])


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Aggregates all sub-configs. Import this in scripts."""
    data: DataConfig = field(default_factory=DataConfig)
    naive: NaiveConfig = field(default_factory=NaiveConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    deep: DeepConfig = field(default_factory=DeepConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    api: APIConfig = field(default_factory=APIConfig)


def get_config(debug: bool = False, category: Optional[str] = None) -> Config:
    """
    Return a Config instance, optionally overriding debug mode or category.

    Args:
        debug: If True, use a small subset of data for fast iteration.
        category: Override the Amazon product category.

    Returns:
        Fully constructed Config.
    """
    cfg = Config()
    if debug:
        cfg.data.debug = True
        cfg.data.debug_n_users = 100_000  # high cap — download size already controlled
        cfg.deep.epochs = 10
        cfg.deep.batch_size = 1024
        cfg.classical.n_estimators = 200
    if category:
        cfg.data.category = category
    return cfg


# ---------------------------------------------------------------------------
# Ensure output directories exist when this module is imported
# ---------------------------------------------------------------------------

def ensure_dirs(cfg: Optional[Config] = None) -> None:
    """Create all necessary directories if they do not exist."""
    dirs = [RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR]
    if cfg:
        dirs.append(cfg.experiment.output_dir)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
