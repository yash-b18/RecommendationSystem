"""
SHAP-based explainability for the LightGBM reranker.

Computes per-prediction SHAP values to identify which features contributed
most to a given recommendation. This is used by the API to generate
feature-level explanations returned to the frontend.

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
           NeurIPS 2017. https://arxiv.org/abs/1705.07874
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import shap

from src.models.classical import LightGBMReranker
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureExplanation(NamedTuple):
    """Container for a single feature's SHAP contribution."""
    feature: str
    shap_value: float
    feature_value: float
    direction: str   # "positive" or "negative"


class SHAPExplainer:
    """
    Wraps a LightGBMReranker with SHAP TreeExplainer for per-prediction explanations.

    Args:
        model: Trained LightGBMReranker.
    """

    def __init__(self, model: LightGBMReranker) -> None:
        if model.model is None:
            raise RuntimeError("Model must be fitted before creating a SHAPExplainer.")
        self.model = model
        self.explainer = shap.TreeExplainer(model.model)
        logger.info("SHAPExplainer initialised for %d features", len(model.feature_names))

    def explain_row(
        self,
        feature_row: pd.DataFrame,
        top_n: int = 5,
    ) -> list[FeatureExplanation]:
        """
        Compute SHAP values for a single user-item feature row.

        Args:
            feature_row: Single-row DataFrame with model feature columns.
            top_n: Number of top contributing features to return.

        Returns:
            List of FeatureExplanation sorted by |shap_value| descending.
        """
        X = feature_row[self.model.feature_names].values.astype(np.float32)
        shap_values = self.explainer.shap_values(X)

        # For binary classification, shap_values may be a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]      # positive class SHAP values for the single row
        else:
            sv = shap_values[0]

        explanations = []
        for feat_name, sv_val, feat_val in zip(self.model.feature_names, sv, X[0]):
            explanations.append(FeatureExplanation(
                feature=feat_name,
                shap_value=float(sv_val),
                feature_value=float(feat_val),
                direction="positive" if sv_val > 0 else "negative",
            ))

        explanations.sort(key=lambda e: abs(e.shap_value), reverse=True)
        return explanations[:top_n]

    def explain_as_text(
        self,
        feature_row: pd.DataFrame,
        item_title: str,
        top_n: int = 3,
    ) -> str:
        """
        Generate a human-readable explanation string for a recommendation.

        Args:
            feature_row: Single-row feature DataFrame.
            item_title: Title of the recommended item.
            top_n: Number of features to mention in the explanation.

        Returns:
            Natural language explanation string.
        """
        explanations = self.explain_row(feature_row, top_n=top_n)

        # Map feature names to human-readable descriptions
        feature_descriptions = {
            "item_avg_rating": "high average rating",
            "item_num_ratings": "many reviews",
            "item_popularity": "high popularity",
            "item_positive_rate": "high positive review rate",
            "category_match": "matches your preferred category",
            "text_similarity": "similar to items you liked",
            "user_avg_rating": "your rating history",
            "user_num_ratings": "your activity level",
            "category_idx": "category relevance",
            "brand_idx": "brand relevance",
            "price_norm": "price range match",
        }

        pos_features = [
            feature_descriptions.get(e.feature, e.feature)
            for e in explanations
            if e.direction == "positive"
        ][:top_n]

        if not pos_features:
            return f"Recommended based on general popularity patterns."

        features_str = ", ".join(pos_features)
        return f'"{item_title[:60]}" is recommended because: {features_str}.'
