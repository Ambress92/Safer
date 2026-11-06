"""ExponentiatedGradient method variants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asbr.baselines.expgrad import ExponGradConfig, ExponGradWrapper
from asbr.baselines.pytorch_mlp_wrapper import PyTorchMLPWrapper
from asbr.experiments.config import TrainingConfig
from asbr.methods.base import TrainedMethod


class ExponGradMethod(TrainedMethod):
    """Fairlearn ExponentiatedGradient with LogisticRegression base.

    Args:
        name: Method name shown in result CSVs.
        epsilon: Fairness constraint tolerance.
        lambd: Mirrors epsilon so the Hydra sweep subdir works uniformly; unused.
        max_iter: Maximum ExponGrad iterations.
        nu: Convergence threshold.
        constraint: Fairness constraint — ``"dp"`` (DemographicParity, default),
            ``"tpr"`` (TruePositiveRateParity), or ``"eo"`` (EqualizedOdds).
    """

    def __init__(
        self,
        name: str = "ExponGrad",
        epsilon: float = 0.02,
        lambd: float | None = None,  # mirrors epsilon; passed by Hydra, unused here
        max_iter: int = 50,
        nu: float = 1e-6,
        constraint: str = "dp",
    ):
        self.name = name
        self._epsilon = epsilon
        self._max_iter = max_iter
        self._nu = nu
        self._constraint = constraint

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
        self._wrapper = ExponGradWrapper(
            ExponGradConfig(
                epsilon=self._epsilon,
                max_iter=self._max_iter,
                nu=self._nu,
                constraint=self._constraint,
            )
        )
        self._wrapper.fit(X_tr, y_tr, A_tr, intersection_level=intersection_level)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        y_prob = self._wrapper.predict_proba_sampled(X)
        y_pred = (y_prob >= threshold).astype(int)
        return y_prob, y_pred


class ExponGradMLPMethod(TrainedMethod):
    """Fairlearn ExponentiatedGradient with PyTorch MLP base estimator."""

    def __init__(
        self,
        name: str = "ExponGrad-MLP",
        mlp_max_epochs: int = 20,
        epsilon: float = 0.02,
        lambd: float | None = None,  # mirrors epsilon; passed by Hydra, unused here
        max_iter: int = 50,
        nu: float = 1e-6,
    ):
        self.name = name
        self._mlp_max_epochs = mlp_max_epochs
        self._epsilon = epsilon
        self._max_iter = max_iter
        self._nu = nu

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
        base_estimator = PyTorchMLPWrapper(
            input_dim=X_tr.shape[1],
            hidden_dims=training_config.hidden_dims,
            dropout=training_config.dropout,
            lr=training_config.lr,
            max_epochs=self._mlp_max_epochs,
            batch_size=training_config.batch_size,
        )
        self._wrapper = ExponGradWrapper(
            ExponGradConfig(
                epsilon=self._epsilon,
                max_iter=self._max_iter,
                nu=self._nu,
            ),
            base_estimator=base_estimator,
        )
        self._wrapper.fit(X_tr, y_tr, A_tr, intersection_level=intersection_level)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        y_prob = self._wrapper.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        return y_prob, y_pred
