"""Encoding of (messy, partially missing) categorical study metadata into integer arrays.

Missing values are encoded as -1.  Rare categories are merged into "__other__" so the
conditional probability tables stay estimable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

OTHER = "__other__"
_MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "unknown", "not available", "nd"}


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return isinstance(v, str) and v.strip().lower() in _MISSING_STRINGS


class MetadataCodec:
    """Maps a DataFrame of categorical columns <-> integer matrix (missing = -1)."""

    def __init__(self, min_count: int = 5, max_categories: int = 30):
        self.min_count = min_count
        self.max_categories = max_categories
        self.columns: list[str] = []
        self.categories: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame, columns: list[str] | None = None) -> "MetadataCodec":
        self.columns = list(columns or df.columns)
        for c in self.columns:
            s = df[c].map(lambda v: None if _is_missing(v) else str(v).strip())
            counts = s.value_counts()
            keep = [k for k, n in counts.items() if n >= self.min_count][: self.max_categories]
            cats = sorted(keep)
            if len(cats) < len(counts):  # something was merged
                cats.append(OTHER)
            if len(cats) < 2:  # degenerate column: keep a dummy 2nd state
                cats = cats + ["__unused__"]
            self.categories[c] = cats
        return self

    @property
    def card(self) -> list[int]:
        return [len(self.categories[c]) for c in self.columns]

    def encode_value(self, col: str, v) -> int:
        if _is_missing(v):
            return -1
        v = str(v).strip()
        cats = self.categories[col]
        if v in cats:
            return cats.index(v)
        return cats.index(OTHER) if OTHER in cats else -1

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = np.full((len(df), len(self.columns)), -1, dtype=np.int64)
        for j, c in enumerate(self.columns):
            if c in df.columns:
                X[:, j] = [self.encode_value(c, v) for v in df[c].values]
        return X

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        out = {}
        for j, c in enumerate(self.columns):
            cats = self.categories[c]
            out[c] = [cats[i] if i >= 0 else None for i in X[:, j]]
        return pd.DataFrame(out)

    def evidence_from_dict(self, d: dict) -> dict[int, int]:
        ev = {}
        for k, v in d.items():
            if k not in self.columns:
                raise KeyError(f"unknown metadata field {k!r}")
            s = self.encode_value(k, v)
            if s >= 0:
                ev[self.columns.index(k)] = s
        return ev
