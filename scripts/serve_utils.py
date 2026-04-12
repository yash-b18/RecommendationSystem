"""
Model loading utilities for the inference server.

Provides a single ModelRegistry class that loads all trained artifacts
once at startup and exposes them to the API layer.

Usage:
    from scripts.serve_utils import ModelRegistry
    registry = ModelRegistry.load(cfg)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.config import Config, MODELS_DIR, PROCESSED_DIR
from src.explainability.language_explainer import LanguageExplainer
from src.models.classical import LightGBMReranker
from src.models.deep import TwoTowerModel, TwoTowerTrainer, build_item_feature_matrix
from src.models.naive import GlobalPopularityRecommender
from src.utils.logging import get_logger

logger = get_logger(__name__)

ITEM_META_COLS = [
    "item_avg_rating", "item_num_ratings", "item_popularity",
    "item_positive_rate", "price_norm", "category_idx", "brand_idx",
]


@dataclass
class ModelRegistry:
    """
    Holds all loaded model artifacts for the inference API.

    All attributes are populated by ModelRegistry.load(cfg).
    """
    naive_model: Optional[GlobalPopularityRecommender] = None
    lgbm_model: Optional[LightGBMReranker] = None
    deep_trainer: Optional[TwoTowerTrainer] = None
    item_embeddings: Optional[np.ndarray] = None
    item_features: Optional[pd.DataFrame] = None
    user_features: Optional[pd.DataFrame] = None
    train_df: Optional[pd.DataFrame] = None
    feature_artifacts: dict = field(default_factory=dict)
    item_index: dict = field(default_factory=dict)
    language_explainer: Optional[LanguageExplainer] = None
    user_seen: dict[int, set[int]] = field(default_factory=dict)
    n_users: int = 0
    n_items: int = 0

    # Which models are available
    naive_available: bool = False
    classical_available: bool = False
    deep_available: bool = False

    @classmethod
    def load(cls, cfg: Config) -> "ModelRegistry":
        """
        Load all available model artifacts from disk.

        Missing artifacts are skipped with a warning (so partial deployments work).

        Args:
            cfg: Full Config object.

        Returns:
            Populated ModelRegistry.
        """
        registry = cls()

        # --- Shared data ---
        try:
            registry.item_features = pd.read_parquet(cfg.data.item_features_path)
            registry.n_items = int(registry.item_features["item_idx"].max() + 1)
        except Exception as e:
            logger.warning("Could not load item_features: %s", e)

        try:
            registry.user_features = pd.read_parquet(cfg.data.user_features_path)
        except Exception as e:
            logger.warning("Could not load user_features: %s", e)

        try:
            registry.train_df = pd.read_parquet(cfg.data.train_path)
            registry.n_users = int(registry.train_df["user_idx"].max() + 1)
            registry.user_seen = (
                registry.train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
            )
        except Exception as e:
            logger.warning("Could not load train_df: %s", e)

        try:
            fa_path = PROCESSED_DIR / "feature_artifacts.pkl"
            with open(fa_path, "rb") as f:
                registry.feature_artifacts = pickle.load(f)
        except Exception as e:
            logger.warning("Could not load feature_artifacts: %s", e)

        try:
            with open(cfg.deep.item_index_path, "rb") as f:
                registry.item_index = pickle.load(f)
        except Exception as e:
            logger.warning("Could not load item_index: %s", e)

        # --- Naive model ---
        naive_path = MODELS_DIR / "naive_baseline.pkl"
        if naive_path.exists():
            try:
                registry.naive_model = GlobalPopularityRecommender.load(naive_path)
                registry.naive_available = True
                logger.info("Naive model loaded.")
            except Exception as e:
                logger.warning("Failed to load naive model: %s", e)

        # --- Classical model ---
        lgbm_path = cfg.classical.model_path
        if lgbm_path.exists():
            try:
                registry.lgbm_model = LightGBMReranker.load(lgbm_path)
                registry.classical_available = True
                logger.info("LightGBM model loaded.")
            except Exception as e:
                logger.warning("Failed to load LightGBM model: %s", e)

        # --- Deep model ---
        if cfg.deep.model_path.exists() and cfg.deep.item_embeddings_path.exists():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                n_meta = 0
                if registry.item_features is not None:
                    item_feat_matrix = build_item_feature_matrix(
                        registry.item_features, ITEM_META_COLS
                    )
                    n_meta = item_feat_matrix.shape[1]
                else:
                    item_feat_matrix = None

                model = TwoTowerModel(
                    n_users=registry.n_users,
                    n_items=registry.n_items,
                    embedding_dim=cfg.deep.embedding_dim,
                    output_dim=cfg.deep.embedding_dim,
                    hidden_dims=cfg.deep.hidden_dims,
                    dropout=0.0,          # no dropout at inference
                    n_meta_features=n_meta,
                )
                model.load_state_dict(
                    torch.load(cfg.deep.model_path, map_location=device)
                )
                registry.deep_trainer = TwoTowerTrainer(
                    model=model, device=device, model_path=cfg.deep.model_path
                )
                registry.item_embeddings = np.load(cfg.deep.item_embeddings_path)
                registry.deep_available = True
                logger.info("Two-Tower model loaded. Item embeddings shape: %s",
                            registry.item_embeddings.shape)
            except Exception as e:
                logger.warning("Failed to load deep model: %s", e)

        # --- Language explainer (for deep model) ---
        if registry.train_df is not None and registry.item_features is not None:
            try:
                registry.language_explainer = LanguageExplainer(
                    registry.train_df,
                    registry.item_features,
                    positive_threshold=cfg.data.positive_rating_threshold,
                )
                logger.info("Language explainer initialised.")
            except Exception as e:
                logger.warning("Failed to initialise language explainer: %s", e)

        logger.info(
            "ModelRegistry ready — naive: %s | classical: %s | deep: %s",
            registry.naive_available,
            registry.classical_available,
            registry.deep_available,
        )
        return registry
