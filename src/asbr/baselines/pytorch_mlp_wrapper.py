"""Sklearn-compatible wrapper for FairnessMLP.

Enables using the same MLP architecture as ASBR methods within Fairlearn's
ExponentiatedGradient reduction. Supports sample_weight in fit() as required
by the reduction framework.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

from asbr.models.base_mlp import FairnessMLP


class PyTorchMLPWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around FairnessMLP.

    Designed for use as the base estimator in Fairlearn's
    ExponentiatedGradient, which calls fit() repeatedly with different
    sample_weight vectors.

    Args:
        input_dim: Number of input features.
        hidden_dims: Tuple of hidden layer sizes.
        dropout: Dropout rate.
        lr: Learning rate for Adam optimizer.
        max_epochs: Maximum training epochs per fit() call.
        batch_size: Mini-batch size.
        device: Torch device string.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
        max_epochs: int = 20,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> PyTorchMLPWrapper:
        """Fit the MLP with optional sample weights.

        Reinitializes the model each call (standard sklearn semantics).
        """
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        dev = torch.device(self.device)
        self.model_ = FairnessMLP(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(dev)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        X_t = torch.tensor(X_arr, dtype=torch.float32, device=dev)
        y_t = torch.tensor(y_arr, dtype=torch.float32, device=dev)

        if sample_weight is not None:
            w_arr = np.asarray(sample_weight, dtype=np.float32)
            w_t = torch.tensor(w_arr, dtype=torch.float32, device=dev)
            dataset = TensorDataset(X_t, y_t, w_t)
        else:
            dataset = TensorDataset(X_t, y_t)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.max_epochs):
            for batch in loader:
                if sample_weight is not None:
                    X_b, y_b, w_b = batch
                else:
                    X_b, y_b = batch
                    w_b = None

                optimizer.zero_grad()
                y_pred = self.model_(X_b)
                loss = self._weighted_bce(y_pred, y_b, w_b)
                loss.backward()
                optimizer.step()

        return self

    @staticmethod
    def _weighted_bce(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Binary cross-entropy with optional per-sample weights."""
        eps = 1e-7
        bce = -(
            y_true * torch.log(y_pred + eps)
            + (1 - y_true) * torch.log(1 - y_pred + eps)
        )
        if weights is not None:
            bce = bce * weights
        return bce.mean()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 2) array of class probabilities."""
        self.model_.eval()
        dev = torch.device(self.device)
        with torch.no_grad():
            X_arr = np.asarray(X, dtype=np.float32)
            X_t = torch.tensor(X_arr, dtype=torch.float32, device=dev)
            prob_1 = self.model_(X_t).cpu().numpy()
        return np.column_stack([1 - prob_1, prob_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
