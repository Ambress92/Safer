"""Two-layer feedforward neural network for binary classification."""

import torch
import torch.nn as nn


class FairnessMLP(nn.Module):
    """Base classifier used by all methods.

    Architecture: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> Sigmoid
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probabilities of shape (batch_size,)."""
        return torch.sigmoid(self.net(x)).squeeze(-1)
