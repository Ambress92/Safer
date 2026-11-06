"""Dataset loading and preprocessing for fairness benchmarks.

Each dataset subclass loads and preprocesses its data in __init__, producing:
  - X: numeric feature matrix (not standardized — call prepare_split for that)
  - y: binary labels
  - A: DataFrame of categorical protected attributes (not in X)

Datasets are Hydra-instantiable via ``_target_`` in YAML configs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FairnessDataset:
    """Unified dataset format for fairness experiments."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        A: pd.DataFrame,
        feature_names: list[str],
        protected_attrs: list[str],
        name: str,
        intersection_level: int = 3,
    ) -> None:
        self.X = X  # (n, d) numeric features
        self.y = y  # (n,) binary labels {0, 1}
        self.A = A  # (n, k) protected attributes as categorical strings
        self.feature_names = feature_names
        self.protected_attrs = protected_attrs
        self.name = name
        self.intersection_level = intersection_level


def _bin_age(age: pd.Series, bins: list[int], labels: list[str]) -> pd.Series:
    """Bin a numeric age column into categorical labels."""
    return pd.cut(age, bins=bins, labels=labels, right=False).astype(str)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first column name that exists in df from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def prepare_split(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize features: fit on train, transform both.

    Returns (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    return X_train_s, X_test_s, scaler
