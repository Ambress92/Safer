"""Fairlearn ExponentiatedGradient wrapper for intersectional fairness baseline.

Wraps sklearn's LogisticRegression with Fairlearn's ExponentiatedGradient
reduction using DemographicParity constraints across intersectional subgroups.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression



@dataclass
class ExponGradConfig:
    """Configuration for the ExponentiatedGradient baseline.

    Attributes:
        epsilon: Fairness constraint tolerance.
        max_iter: Maximum number of iterations.
        nu: Convergence threshold.
        constraint: Fairness constraint type. One of:
            ``"dp"`` — DemographicParity (default),
            ``"tpr"`` — TruePositiveRateParity (EO, TPR only),
            ``"eo"`` — EqualizedOdds (EO, both TPR and FPR).
    """

    epsilon: float = 0.02
    max_iter: int = 50
    nu: float = 1e-6
    constraint: str = "dp"


def build_sensitive_features(
    A: pd.DataFrame,
    level: int = 3,
) -> pd.Series:
    """Build a single composite sensitive-feature column from protected attributes.

    Creates all intersectional subgroups up to the specified level and encodes
    each sample with its finest-grained group label. For Fairlearn, each unique
    value in the resulting Series is treated as a separate group.

    Args:
        A: DataFrame of categorical protected attributes, shape (n, k).
        level: Maximum intersection level (1, 2, or 3).

    Returns:
        Series of composite group labels, shape (n,).
    """
    cols = list(A.columns)
    max_level = min(level, len(cols))

    # Use the highest-level intersection that includes all columns
    # Fairlearn enforces parity across all unique values of the sensitive feature
    parts = []
    for col in cols[:max_level]:
        parts.append(A[col].astype(str))

    return parts[0].str.cat(parts[1:], sep="&") if len(parts) > 1 else parts[0]


class ExponGradWrapper:
    """Wrapper around Fairlearn's ExponentiatedGradient for intersectional fairness.

    By default uses LogisticRegression as the base estimator. A custom
    sklearn-compatible estimator can be passed via ``base_estimator`` (e.g.
    :class:`~asbr.baselines.pytorch_mlp_wrapper.PyTorchMLPWrapper`).
    """

    def __init__(
        self,
        config: ExponGradConfig | None = None,
        base_estimator: object | None = None,
    ):
        self.config = config or ExponGradConfig()
        self.base_estimator = base_estimator
        self.model_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        A_train: pd.DataFrame,
        intersection_level: int = 3,
    ) -> ExponGradWrapper:
        """Fit the ExponentiatedGradient model.

        Args:
            X_train: Training features, shape (n, d).
            y_train: Training labels, shape (n,).
            A_train: Protected attributes DataFrame.
            intersection_level: Maximum intersection level for group construction.

        Returns:
            self
        """
        from fairlearn.reductions import (
            ExponentiatedGradient,
            DemographicParity,
            EqualizedOdds,
            TruePositiveRateParity,
        )

        sensitive = build_sensitive_features(A_train, level=intersection_level)

        if self.base_estimator is not None:
            base_estimator = self.base_estimator
        else:
            base_estimator = LogisticRegression(
                solver="liblinear", max_iter=1000, random_state=0,
            )

        c = self.config.constraint
        if c == "tpr":
            constraint = TruePositiveRateParity()
        elif c == "eo":
            constraint = EqualizedOdds()
        elif c == "dp":
            constraint = DemographicParity()
        else:
            raise ValueError(
                f"Unknown constraint {c!r}. Expected 'dp', 'tpr', or 'eo'."
            )

        self.model_ = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=constraint,
            eps=self.config.epsilon,
            max_iter=self.config.max_iter,
            nu=self.config.nu,
        )
        self.model_.fit(X_train, y_train, sensitive_features=sensitive)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for class 1 by averaging mixture components.

        This is the naive mixture-average estimator.  It collapses the
        score distribution and artificially deflates AUC; prefer
        :meth:`predict_proba_sampled` for AUC-based evaluation.

        Args:
            X: Features, shape (n, d).

        Returns:
            y_prob: Mixture-averaged predicted probabilities, shape (n,).
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        weights = np.asarray(self.model_.weights_)
        probas = np.stack(
            [est.predict_proba(X)[:, 1] for est in self.model_.predictors_],
            axis=1,
        )  # (n, K)
        return probas @ (weights / weights.sum())

    def predict_proba_sampled(self, X: np.ndarray, seed: int | None = None) -> np.ndarray:
        """Predict probabilities by sampling one component from the mixture.

        Draws a single base classifier from the mixture distribution (according
        to the learned weights) and returns its calibrated probabilities.  This
        gives an unbiased estimate of the randomized classifier's AUC:

            E_θ[AUC(score(·; θ))]

        which is the correct AUC for a randomized predictor, unlike the AUC of
        the mixture-averaged score returned by :meth:`predict_proba`.

        Args:
            X: Features, shape (n, d).
            seed: Optional RNG seed for reproducibility.  If None, uses the
                current numpy random state.

        Returns:
            y_prob: Predicted probabilities from the sampled component, shape (n,).
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        rng = np.random.RandomState(seed)
        weights = np.asarray(self.model_.weights_)
        weights = weights / weights.sum()
        idx = rng.choice(len(self.model_.predictors_), p=weights)
        return self.model_.predictors_[idx].predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels.

        Args:
            X: Features, shape (n, d).

        Returns:
            y_pred: Binary predictions, shape (n,).
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model_.predict(X).astype(int)
