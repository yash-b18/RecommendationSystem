"""
Classical ML model: LightGBM-based recommendation reranker.

Formulates recommendation as binary classification:
  label = 1 if the user interacted positively with the item, else 0.

At inference, for each user we score a candidate pool of items (retrieved
by the naive popularity model) and return the top-K by predicted probability.

Explainability is provided via:
  - Global feature importance (gain-based)
  - Per-prediction SHAP values (computed by the explainability module)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class LightGBMReranker:
    """
    LightGBM binary classifier used as a recommendation reranker.

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Gradient boosting learning rate.
        num_leaves: Maximum number of leaves per tree.
        max_depth: Maximum tree depth (-1 = unlimited).
        min_child_samples: Minimum samples per leaf.
        subsample: Row sub-sampling ratio.
        colsample_bytree: Column sub-sampling ratio.
        reg_alpha: L1 regularisation.
        reg_lambda: L2 regularisation.
        n_jobs: Parallel threads.
        random_state: Random seed.
        early_stopping_rounds: Early stopping patience.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        n_jobs: int = -1,
        random_state: int = 42,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            objective="binary",
            metric="binary_logloss",
            verbose=-1,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names: list[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> "LightGBMReranker":
        """
        Train the LightGBM classifier.

        Args:
            train_df: Training feature matrix with a 'label' column.
            val_df: Validation feature matrix for early stopping.
            feature_cols: Column names to use as features.

        Returns:
            self
        """
        self.feature_names = feature_cols
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df["label"].values
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df["label"].values

        self.model = lgb.LGBMClassifier(**self.params)
        callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                     lgb.log_evaluation(100)]
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
            feature_name=feature_cols,
        )
        logger.info(
            "LightGBM trained for %d rounds (best: %d)",
            self.params["n_estimators"],
            self.model.best_iteration_,
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return positive-class probabilities for rows in df.

        Args:
            df: Feature DataFrame with columns matching feature_names.

        Returns:
            1-D numpy array of probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        X = df[self.feature_names].values.astype(np.float32)
        return self.model.predict_proba(X)[:, 1]

    def recommend(
        self,
        user_idx: int,
        candidate_item_idxs: list[int],
        feature_df: pd.DataFrame,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Score a set of candidate items for a user and return the top-K.

        Args:
            user_idx: Encoded user index.
            candidate_item_idxs: Pool of item indices to score.
            feature_df: Pre-built feature rows for (user_idx, item_idx) pairs.
                        Must contain all feature_names columns.
            top_k: Number of recommendations.

        Returns:
            List of (item_idx, score) sorted descending.
        """
        scores = self.predict_proba(feature_df)
        ranked = sorted(
            zip(candidate_item_idxs, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Return a DataFrame of feature importances sorted descending.

        Args:
            importance_type: 'gain' or 'split'.

        Returns:
            DataFrame with columns [feature, importance].
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        importances = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved LightGBMReranker → %s", path)

    @classmethod
    def load(cls, path: Path) -> "LightGBMReranker":
        with open(path, "rb") as f:
            return pickle.load(f)
