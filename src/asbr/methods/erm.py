"""ERM baseline method."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asbr.evaluation.evaluator import compute_predictions
from asbr.experiments.config import TrainingConfig
from asbr.methods.base import TrainedMethod
from asbr.models.base_mlp import FairnessMLP
from asbr.training.trainer import BaselineTrainer


class ERMMethod(TrainedMethod):
    """Empirical Risk Minimization baseline (no fairness regularization)."""

    def __init__(self, name: str = "ERM", lambd: float = 0):
        self.name = name

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
        self._model = FairnessMLP(
            input_dim=X_tr.shape[1],
            hidden_dims=training_config.hidden_dims,
            dropout=training_config.dropout,
        )
        trainer = BaselineTrainer(model=self._model, training_config=training_config)
        trainer.train(X_tr, y_tr, X_vl, y_vl)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        return compute_predictions(self._model, X, threshold=threshold)
