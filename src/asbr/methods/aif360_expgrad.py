"""AIF360 ExponentiatedGradientReduction method variants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asbr.baselines.aif360_expgrad import AIF360ExponGradConfig, AIF360ExponGradWrapper
from asbr.baselines.pytorch_mlp_wrapper import PyTorchMLPWrapper
from asbr.experiments.config import TrainingConfig
from asbr.methods.base import TrainedMethod


class AIF360ExponGradMethod(TrainedMethod):
    """AIF360 ExponentiatedGradientReduction with LogisticRegression base."""

    def __init__(
        self,
        name: str = "AIF360-ExponGrad",
        eps: float = 0.01,
        max_iter: int = 50,
        nu: float | None = None,
        eta0: float = 2.0,
        drop_prot_attr: bool = True,
    ):
        self.name = name
        self._eps = eps
        self._max_iter = max_iter
        self._nu = nu
        self._eta0 = eta0
        self._drop_prot_attr = drop_prot_attr

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
        self._wrapper = AIF360ExponGradWrapper(
            AIF360ExponGradConfig(
                eps=self._eps,
                max_iter=self._max_iter,
                nu=self._nu,
                eta0=self._eta0,
                drop_prot_attr=self._drop_prot_attr,
            )
        )
        self._wrapper.fit(X_tr, y_tr, A_tr, intersection_level=intersection_level)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        y_prob = self._wrapper.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        return y_prob, y_pred


class AIF360ExponGradMLPMethod(TrainedMethod):
    """AIF360 ExponentiatedGradientReduction with PyTorch MLP base estimator."""

    def __init__(
        self,
        name: str = "AIF360-ExponGrad-MLP",
        mlp_max_epochs: int = 20,
        eps: float = 0.01,
        max_iter: int = 50,
        nu: float | None = None,
        eta0: float = 2.0,
        drop_prot_attr: bool = True,
    ):
        self.name = name
        self._mlp_max_epochs = mlp_max_epochs
        self._eps = eps
        self._max_iter = max_iter
        self._nu = nu
        self._eta0 = eta0
        self._drop_prot_attr = drop_prot_attr

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
        self._wrapper = AIF360ExponGradWrapper(
            AIF360ExponGradConfig(
                eps=self._eps,
                max_iter=self._max_iter,
                nu=self._nu,
                eta0=self._eta0,
                drop_prot_attr=self._drop_prot_attr,
            ),
            base_estimator=base_estimator,
        )
        self._wrapper.fit(X_tr, y_tr, A_tr, intersection_level=intersection_level)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        y_prob = self._wrapper.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        return y_prob, y_pred
