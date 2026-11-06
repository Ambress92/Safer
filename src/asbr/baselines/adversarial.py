"""Fairlearn AdversarialFairnessClassifier wrapper for adversarial debiasing baseline.

Wraps Fairlearn's AdversarialFairnessClassifier (Zhang et al. 2018) which trains
a predictor and adversary in a minimax game — the predictor tries to be accurate
while preventing the adversary from predicting the sensitive attribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from asbr.baselines.expgrad import build_sensitive_features


@dataclass
class AdversarialDebiasingConfig:
    """Configuration for the AdversarialFairnessClassifier baseline."""

    alpha: float = 1.0
    predictor_model: list | None = None
    adversary_model: list | None = None
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 256
    constraints: str = "demographic_parity"
    random_state: int | None = 0


class AdversarialDebiasingWrapper:
    """Wrapper around Fairlearn's AdversarialFairnessClassifier.

    Uses the PyTorch backend for minimax adversarial training to enforce
    fairness constraints across intersectional subgroups.
    """

    def __init__(self, config: AdversarialDebiasingConfig | None = None):
        self.config = config or AdversarialDebiasingConfig()
        self.model_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        A_train: pd.DataFrame,
        intersection_level: int = 3,
    ) -> AdversarialDebiasingWrapper:
        """Fit the adversarial debiasing model.

        Args:
            X_train: Training features, shape (n, d).
            y_train: Training labels, shape (n,).
            A_train: Protected attributes DataFrame.
            intersection_level: Maximum intersection level for group construction.

        Returns:
            self
        """
        sensitive = build_sensitive_features(A_train, level=intersection_level)

        from fairlearn.adversarial import AdversarialFairnessClassifier

        cfg = self.config
        adversary_model = cfg.adversary_model if cfg.adversary_model is not None else [3, "relu"]

        self.model_ = AdversarialFairnessClassifier(
            predictor_model=cfg.predictor_model,
            adversary_model=adversary_model,
            alpha=cfg.alpha,
            learning_rate=cfg.learning_rate,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            constraints=cfg.constraints,
            random_state=cfg.random_state,
        )
        self.model_.fit(X_train, y_train, sensitive_features=sensitive)
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
        # _raw_predict returns sigmoid outputs of shape (n, 1)
        raw = self.model_._raw_predict(X)
        return np.asarray(raw).ravel()

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
