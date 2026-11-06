"""AIF360 ExponentiatedGradientReduction wrapper for intersectional fairness.

Wraps AIF360's sklearn-compatible ExponentiatedGradientReduction with
DemographicParity constraints across intersectional subgroups.

Unlike the Fairlearn wrapper, AIF360's API expects protected attributes as
columns in the feature DataFrame and handles sensitive feature extraction
internally via ``prot_attr``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from aif360.sklearn.inprocessing import ExponentiatedGradientReduction


@dataclass
class AIF360ExponGradConfig:
    """Configuration for the AIF360 ExponentiatedGradientReduction."""

    eps: float = 0.01
    max_iter: int = 50
    nu: float | None = None
    eta0: float = 2.0
    drop_prot_attr: bool = True


def build_intersectional_column(
    A: pd.DataFrame,
    level: int = 3,
) -> pd.Series:
    """Build a single intersectional column from protected attributes.

    Args:
        A: DataFrame of categorical protected attributes, shape (n, k).
        level: Maximum intersection level (1, 2, or 3).

    Returns:
        Series of composite group labels, shape (n,).
    """
    cols = list(A.columns)
    max_level = min(level, len(cols))
    parts = [A[col].astype(str) for col in cols[:max_level]]
    return parts[0].str.cat(parts[1:], sep="&") if len(parts) > 1 else parts[0]


class AIF360ExponGradWrapper:
    """Wrapper around AIF360's ExponentiatedGradientReduction.

    By default uses LogisticRegression as the base estimator. A custom
    sklearn-compatible estimator can be passed via ``base_estimator``.
    """

    _PROT_ATTR_NAME = "_intersect_group"

    def __init__(
        self,
        config: AIF360ExponGradConfig | None = None,
        base_estimator: object | None = None,
    ):
        self.config = config or AIF360ExponGradConfig()
        self.base_estimator = base_estimator
        self.model_: ExponentiatedGradientReduction | None = None

    def _make_df(self, X: np.ndarray, intersect_col: pd.Series | None = None) -> pd.DataFrame:
        """Create DataFrame with optional protected attribute column.

        AIF360 expects the protected attribute as a column in X for both
        fit() and predict(). When drop_prot_attr=True it drops the column
        before passing to the base estimator.
        """
        X_df = pd.DataFrame(X)
        if intersect_col is not None:
            X_df[self._PROT_ATTR_NAME] = intersect_col.values
        else:
            # For predict: add a dummy column so AIF360 can drop it
            X_df[self._PROT_ATTR_NAME] = ""
        return X_df

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        A_train: pd.DataFrame,
        intersection_level: int = 3,
    ) -> AIF360ExponGradWrapper:
        """Fit the AIF360 ExponentiatedGradientReduction model.

        Args:
            X_train: Training features, shape (n, d).
            y_train: Training labels, shape (n,).
            A_train: Protected attributes DataFrame.
            intersection_level: Maximum intersection level for group construction.

        Returns:
            self
        """
        cfg = self.config
        intersect_col = build_intersectional_column(A_train, level=intersection_level)
        X_df = self._make_df(X_train, intersect_col)

        if self.base_estimator is not None:
            base_estimator = self.base_estimator
        else:
            base_estimator = LogisticRegression(
                solver="liblinear", max_iter=1000, random_state=0,
            )

        self.model_ = ExponentiatedGradientReduction(
            prot_attr=self._PROT_ATTR_NAME,
            estimator=base_estimator,
            constraints="DemographicParity",
            eps=cfg.eps,
            max_iter=cfg.max_iter,
            nu=cfg.nu,
            eta0=cfg.eta0,
            drop_prot_attr=cfg.drop_prot_attr,
        )
        self.model_.fit(X_df, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for class 1.

        Args:
            X: Features, shape (n, d).

        Returns:
            y_prob: Predicted probabilities, shape (n,).
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_df = self._make_df(X)
        proba = self.model_.predict_proba(X_df)
        return proba[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels.

        Args:
            X: Features, shape (n, d).

        Returns:
            y_pred: Binary predictions, shape (n,).
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_df = self._make_df(X)
        return self.model_.predict(X_df).astype(int)
