"""Adversarial Debiasing method (Fairlearn AdversarialFairnessClassifier)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asbr.experiments.config import TrainingConfig
from asbr.methods.base import TrainedMethod


class AdversarialDebiasingMethod(TrainedMethod):
    """Adversarial Debiasing via Fairlearn's AdversarialFairnessClassifier."""

    def __init__(
        self,
        name: str = "AdvDebiasing",
        alpha: float = 1.0,
        learning_rate: float = 0.001,
        epochs: int | None = None,
        batch_size: int | None = None,
        constraints: str = "demographic_parity",
        adversary_model: list | None = None,
        random_state: int | None = 0,
    ):
        self.name = name
        self._alpha = alpha
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._constraints = constraints
        self._adversary_model = adversary_model
        self._random_state = random_state

    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_vl: np.ndarray,
        y_vl: np.ndarray,
        A_tr: pd.DataFrame,
        intersection_level: int,
        training_config: TrainingConfig,
    ) -> None:
        # Build predictor architecture from training_config.hidden_dims
        # e.g. (64, 32) => [64, "relu", 32, "relu"]
        predictor_model: list = []
        for dim in training_config.hidden_dims:
            predictor_model.append(dim)
            predictor_model.append("relu")

        epochs = self._epochs if self._epochs is not None else training_config.max_epochs
        batch_size = self._batch_size if self._batch_size is not None else training_config.batch_size

        from asbr.baselines.adversarial import AdversarialDebiasingConfig, AdversarialDebiasingWrapper
        self._wrapper = AdversarialDebiasingWrapper(
            AdversarialDebiasingConfig(
                alpha=self._alpha,
                predictor_model=predictor_model,
                adversary_model=self._adversary_model,
                learning_rate=self._learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                constraints=self._constraints,
                random_state=self._random_state,
            )
        )
        self._wrapper.fit(X_tr, y_tr, A_tr, intersection_level=intersection_level)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        y_prob = self._wrapper.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        return y_prob, y_pred
