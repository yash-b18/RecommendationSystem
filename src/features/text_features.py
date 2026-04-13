"""
Text-based feature engineering.

Computes:
  1. Item title TF-IDF vectors (for similarity against user's text profile)
  2. User text profile: concatenation of titles of positively-rated items
  3. User-item text cosine similarity feature

These are used in the "text" ablation tier of the classical model.

For the deep model, we use a lightweight word-index encoding of item titles
instead of full sentence embeddings (to keep training feasible on CPU).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_item_tfidf(
    item_features: pd.DataFrame,
    max_features: int = 5_000,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Fit TF-IDF on item titles and return the dense (n_items, max_features) matrix.

    Args:
        item_features: DataFrame with columns [item_idx, title].
        max_features: Vocabulary size.

    Returns:
        Tuple of (tfidf_matrix, fitted_vectorizer).
    """
    titles = item_features["title"].fillna("").tolist()
    vec = TfidfVectorizer(max_features=max_features, sublinear_tf=True, strip_accents="unicode")
    tfidf_matrix = vec.fit_transform(titles).toarray().astype(np.float32)
    logger.info("TF-IDF matrix shape: %s", tfidf_matrix.shape)
    return tfidf_matrix, vec


def build_user_text_profiles(
    train_df: pd.DataFrame,
    item_features: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Build a text profile per user by concatenating the titles of positively-rated items.

    Args:
        train_df: Training interactions.
        item_features: Item features with [item_idx, title].
        positive_threshold: Rating threshold for positive interactions.

    Returns:
        DataFrame with columns [user_idx, user_text_profile].
    """
    positives = train_df[train_df["rating"] >= positive_threshold].copy()
    positives = positives.merge(item_features[["item_idx", "title"]], on="item_idx", how="left")
    positives["title"] = positives["title"].fillna("")

    profiles = (
        positives.groupby("user_idx")["title"]
        .apply(lambda titles: " ".join(titles))
        .reset_index()
        .rename(columns={"title": "user_text_profile"})
    )
    return profiles


def compute_text_similarity(
    user_profiles: pd.DataFrame,
    item_features: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    interaction_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute cosine similarity between each user's text profile and each candidate item title.

    This is called at feature-building time for each user-item pair in the dataset.

    Args:
        user_profiles: DataFrame with [user_idx, user_text_profile].
        item_features: DataFrame with [item_idx, title].
        vectorizer: Fitted TF-IDF vectorizer.
        interaction_df: DataFrame with [user_idx, item_idx] pairs to compute similarity for.

    Returns:
        Series of cosine similarity scores, indexed like interaction_df.
    """
    # Build lookup dicts
    uid_to_profile = dict(zip(user_profiles["user_idx"], user_profiles["user_text_profile"]))
    iid_to_title = dict(zip(item_features["item_idx"], item_features["title"].fillna("")))

    user_idxs = interaction_df["user_idx"].values
    item_idxs = interaction_df["item_idx"].values

    unique_users = np.unique(user_idxs)
    unique_items = np.unique(item_idxs)

    # Vectorise unique texts in batch
    user_texts = [uid_to_profile.get(u, "") for u in unique_users]
    item_texts = [iid_to_title.get(i, "") for i in unique_items]

    user_vecs = vectorizer.transform(user_texts).toarray().astype(np.float32)
    item_vecs = vectorizer.transform(item_texts).toarray().astype(np.float32)

    # Map back to per-row indices
    user_idx_map = {u: idx for idx, u in enumerate(unique_users)}
    item_idx_map = {i: idx for idx, i in enumerate(unique_items)}

    sims = []
    for uid, iid in zip(user_idxs, item_idxs):
        u_vec = user_vecs[user_idx_map[uid]].reshape(1, -1)
        i_vec = item_vecs[item_idx_map[iid]].reshape(1, -1)
        sim = float(cosine_similarity(u_vec, i_vec)[0, 0])
        sims.append(sim)

    return pd.Series(sims, index=interaction_df.index, name="text_similarity")


def build_item_title_tokens(
    item_features: pd.DataFrame,
    vocab_size: int = 10_000,
    max_len: int = 32,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Convert item titles to fixed-length integer token sequences for the deep model.

    Args:
        item_features: DataFrame with [item_idx, title] sorted by item_idx.
        vocab_size: Maximum vocabulary size.
        max_len: Maximum title token length (truncate/pad).

    Returns:
        Tuple of (token_matrix of shape (n_items, max_len), word2idx dict).
    """
    from collections import Counter
    import re

    titles = item_features.sort_values("item_idx")["title"].fillna("").tolist()

    # Build vocabulary
    word_counts: Counter = Counter()
    for title in titles:
        tokens = re.findall(r"\w+", title.lower())
        word_counts.update(tokens)

    vocab = ["<PAD>", "<UNK>"] + [w for w, _ in word_counts.most_common(vocab_size - 2)]
    word2idx = {w: i for i, w in enumerate(vocab)}

    token_matrix = np.zeros((len(titles), max_len), dtype=np.int32)
    for row_idx, title in enumerate(titles):
        tokens = re.findall(r"\w+", title.lower())[:max_len]
        for col_idx, tok in enumerate(tokens):
            token_matrix[row_idx, col_idx] = word2idx.get(tok, 1)  # 1 = <UNK>

    logger.info("Title token matrix: %s, vocab size: %d", token_matrix.shape, len(word2idx))
    return token_matrix, word2idx
