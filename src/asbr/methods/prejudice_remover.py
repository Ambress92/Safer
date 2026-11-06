"""Prejudice Remover Regularizer method (Kamishima et al. 2012)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asbr.baselines.prejudice_remover import PrejudiceRemoverConfig, PrejudiceRemoverWrapper
from asbr.experiments.config import TrainingConfig
from asbr.methods.base import TrainedMethod


class PrejudiceRemoverMethod(TrainedMethod):
    """Prejudice Remover via AIF360's classic API."""

    def __init__(
        self,
        name: str = "PrejudiceRemover",
        eta: float = 1.0,
        lambd: float | None = None,  # alias for eta so sweeps use standard lambd dirs
    ):
        self.name = name
        self._eta = eta if lambd is None else lambd

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
        self._wrapper = PrejudiceRemoverWrapper(
            PrejudiceRemoverConfig(eta=self._eta),
        )
        self._wrapper.fit(X_tr, y_tr, A_tr, intersection_level=intersection_level)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        y_prob = self._wrapper.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        return y_prob, y_pred
