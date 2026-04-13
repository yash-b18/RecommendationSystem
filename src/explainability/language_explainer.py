"""
Metadata-grounded language explanations for the Two-Tower neural model.

Generates factual, template-based recommendation explanations grounded in
observable data (user history, item metadata) — not hallucinated.

For the Amazon Books dataset, explanations are built from:
  - Author match (brand field contains "Author Name (Author)")
  - Item popularity and rating tier
  - Title keyword overlap with user history
  - Price affinity
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd


_BOOK_STOPWORDS = {
    "the", "a", "an", "and", "of", "for", "with", "in", "on", "at", "to",
    "by", "from", "is", "it", "its", "this", "that", "as", "or", "are",
    "be", "was", "but", "not", "you", "your", "my", "we", "our", "their",
    "book", "books", "edition", "volume", "series", "novel", "story",
    "guide", "complete", "new", "great", "good", "best", "first", "second",
    "one", "two", "three", "four", "five", "no", "so", "do", "if", "all",
    "i", "s", "t", "re", "ve", "d", "ll",
}


def _extract_author(brand: str) -> str | None:
    """
    Extract clean author name from Amazon brand strings like
    'Michael Chabon (Author)' or 'Stephen King (Author), Joe Hill (Author)'.
    Returns the first author's name, or None if not parseable.
    """
    if not brand or brand in ("Unknown", "nan"):
        return None
    # Match "Name (Author)" pattern — take first author listed
    match = re.search(r"([^,]+?)\s*\(Author\)", brand)
    if match:
        return match.group(1).strip()
    return None


def _rating_tier(avg_rating: float, num_ratings: int) -> str | None:
    """Return a human-readable quality signal, or None if no useful signal."""
    if num_ratings < 10:
        return None
    # avg_rating is stored as mean of 1-5 ratings normalised to 0-1 in some
    # datasets — but here it appears to be raw sum / count so values > 1 occur.
    # Treat it as a raw average out of 5.
    if avg_rating >= 4.2 and num_ratings >= 50:
        return f"highly rated ({avg_rating:.1f}★, {num_ratings:,} reviews)"
    if avg_rating >= 3.8 and num_ratings >= 200:
        return f"popular ({num_ratings:,} reviews)"
    if num_ratings >= 500:
        return f"widely read ({num_ratings:,} reviews)"
    return None


def _title_keywords(title: str) -> set[str]:
    words = re.findall(r"[a-z]+", title.lower())
    return {w for w in words if w not in _BOOK_STOPWORDS and len(w) > 3}


class LanguageExplainer:
    """
    Generates template-based explanations for Two-Tower recommendations.

    Args:
        train_df: Training interactions with [user_idx, item_idx, rating].
        item_features: Item metadata with [item_idx, title, category, brand,
                       item_avg_rating, item_num_ratings, price].
        positive_threshold: Minimum rating to count as positive.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        item_features: pd.DataFrame,
        positive_threshold: float = 4.0,
    ) -> None:
        needed = ["item_idx", "title", "brand", "item_avg_rating", "item_num_ratings", "price"]
        available = [c for c in needed if c in item_features.columns]
        meta = item_features[available].copy()

        # Extract author from brand
        if "brand" in meta.columns:
            meta["author"] = meta["brand"].fillna("").apply(_extract_author)
        else:
            meta["author"] = None

        self._item_meta: dict[int, dict] = meta.set_index("item_idx").to_dict(orient="index")

        # Build per-user history from positive interactions only
        positives = train_df[train_df["rating"] >= positive_threshold].copy()
        positives = positives.merge(
            meta[["item_idx", "author", "title", "price"]].rename(columns={}),
            on="item_idx",
            how="left",
        )

        self._user_authors: dict[int, list[str]] = (
            positives.dropna(subset=["author"])
            .groupby("user_idx")["author"]
            .apply(list)
            .to_dict()
        )
        self._user_titles: dict[int, list[str]] = (
            positives.groupby("user_idx")["title"]
            .apply(lambda x: x.fillna("").tolist())
            .to_dict()
        )
        self._user_prices: dict[int, list[float]] = (
            positives.dropna(subset=["price"])
            .groupby("user_idx")["price"]
            .apply(list)
            .to_dict()
            if "price" in positives.columns else {}
        )

    def explain(
        self,
        user_idx: int,
        item_idx: int,
        score: float,
    ) -> str:
        """
        Generate a natural language explanation for recommending item_idx to user_idx.
        """
        meta = self._item_meta.get(item_idx, {})
        item_title = str(meta.get("title") or "this book")
        item_author = meta.get("author")
        avg_rating = meta.get("item_avg_rating")
        num_ratings = meta.get("item_num_ratings")
        item_price = meta.get("price")

        user_authors = self._user_authors.get(user_idx, [])
        user_titles = self._user_titles.get(user_idx, [])
        user_prices = self._user_prices.get(user_idx, [])

        top_authors = [a for a, _ in Counter(user_authors).most_common(5)]

        reasons: list[str] = []

        # 1. Author match — strongest signal
        if item_author and item_author in top_authors:
            count = user_authors.count(item_author)
            if count >= 2:
                reasons.append(f"you've enjoyed {count} books by {item_author}")
            else:
                reasons.append(f"you previously liked a book by {item_author}")

        # 2. Title keyword overlap with history
        item_words = _title_keywords(item_title)
        history_words: set[str] = set()
        for t in user_titles[:20]:
            history_words |= _title_keywords(t)
        shared = item_words & history_words
        if len(shared) >= 2:
            sample = sorted(shared, key=len, reverse=True)[:3]
            reasons.append(f"it shares themes with your history (\"{', '.join(sample)}\")")

        # 3. Rating / popularity signal
        if avg_rating is not None and num_ratings is not None:
            tier = _rating_tier(float(avg_rating), int(num_ratings))
            if tier:
                reasons.append(f"it is {tier}")

        # 4. Price affinity
        if item_price and user_prices:
            avg_user_price = sum(user_prices) / len(user_prices)
            if abs(item_price - avg_user_price) <= avg_user_price * 0.3:
                reasons.append(f"it fits your typical price range (${item_price:.0f})")

        # 5. Fallback — author + score confidence
        if not reasons:
            if item_author:
                reasons.append(f"written by {item_author}")
            if avg_rating and num_ratings and int(num_ratings) >= 20:
                reasons.append(f"rated {float(avg_rating):.1f}★ by {int(num_ratings):,} readers")
            if not reasons:
                reasons.append("it matches your reading patterns based on neural similarity")

        return "Recommended because " + "; ".join(reasons) + "."
