"""Abstract base class for all training methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from asbr.experiments.config import TrainingConfig


class TrainedMethod(ABC):
    """Uniform interface for all fairness methods.

    Hydra instantiates concrete subclasses via ``_target_`` in method YAML
    configs. After instantiation, call :meth:`train` then :meth:`predict`.
    """

    name: str

    @abstractmethod
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
        """Train the method on the given data."""
        ...

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_prob, y_pred) arrays."""
        ...
