"""Full ASBR loss function composing all fairness components.

L_ASBR(theta) = L_pred + lambda * sum_S omega(n_S) * Psi(Z_S; alpha)

Supports ablation variants by toggling use_hinge, use_hybrid, and
use_size_weight flags in ASBRConfig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from scipy.stats import norm

from asbr.experiments.config import ASBRConfig
from asbr.fairness.soft_counts import compute_soft_counts, compute_hard_counts
from asbr.fairness.hybrid_gradient import compute_hybrid_estimate
from asbr.fairness.wald import (
    compute_asymptotic_variance,
    compute_statistical_parity,
    compute_wald_statistic,
)


@dataclass
class SubgroupDiagnostics:
    """Per-subgroup diagnostics recorded during a forward pass."""

    n_S: int
    sp: float
    sigma2: float
    z_score: float
    active: bool
    hinge_value: float
    beta: float  # Transition weight (0 = Bayesian, 1 = frequentist)


class ASBRLoss(nn.Module):
    """Adaptive Significance-Based Regularization loss.

    Composes:
    - Binary cross-entropy prediction loss
    - Significance hinge fairness penalty with size-weight and hybrid gradient

    Args:
        config: ASBRConfig with all hyperparameters and ablation flags.
    """

    def __init__(self, config: ASBRConfig | None = None):
        super().__init__()
        self.config = config or ASBRConfig()
        self.z_alpha = norm.ppf(1.0 - self.config.alpha / 2.0)
        self.ce_loss = nn.BCELoss()

    def compute_fairness_loss(
        self,
        y_pred: torch.Tensor,
        subgroup_masks: dict[str, torch.Tensor],
        y_true: torch.Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, SubgroupDiagnostics]]:
        """Compute the fairness regularization term.

        This is separated from the prediction loss so callers can control
        how the two losses are combined (e.g., different batch strategies).

        Args:
            y_pred: Model output probabilities, shape (n,).
            subgroup_masks: Dict mapping subgroup names to boolean masks.
            y_true: Ground-truth labels (unused by this Wald-based loss;
                accepted for API compatibility with ASBRLRTLoss).
            collect_diagnostics: If False (default), skip all .item() calls
                that force GPU→CPU synchronization. Pass True only when
                per-subgroup diagnostics are actually needed (e.g., final epoch).

        Returns:
            fair_loss: Scalar fairness loss (sum of hinge penalties).
            diagnostics: Per-subgroup diagnostics dict (empty when collect_diagnostics=False).
        """
        cfg = self.config
        n = y_pred.shape[0]
        diagnostics: dict[str, SubgroupDiagnostics] = {}

        # Accumulate fairness loss; start with a zero that shares the compute graph
        fair_loss = torch.zeros(1, device=y_pred.device, dtype=y_pred.dtype)
        n_contributing = 0

        for name, mask_S in subgroup_masks.items():
            n_S = int(mask_S.sum().item())
            n_Sbar = n - n_S

            # Skip degenerate subgroups
            if n_S < 2 or n_Sbar < 2:
                continue

            n_contributing += 1
            p_hat = compute_soft_counts(y_pred, mask_S)

            # --- Estimate SP and variance ---
            if cfg.use_hybrid:
                counts = compute_hard_counts(y_pred.detach(), mask_S)
                sp_est, sigma2_est, z_s = compute_hybrid_estimate(
                    p_hat=p_hat,
                    counts=counts,
                    n_S=n_S,
                    n_total=n,
                    n_tilde=cfg.n_tilde,
                    kappa=cfg.kappa,
                    dirichlet_prior=cfg.dirichlet_prior,
                    epsilon=cfg.epsilon,
                )
            else:
                # Frequentist only (ablation: -Hybrid Gradient)
                sp_est = compute_statistical_parity(p_hat, epsilon=cfg.epsilon)
                sigma2_est = compute_asymptotic_variance(p_hat, epsilon=cfg.epsilon)
                z_s = compute_wald_statistic(sp_est, sigma2_est, n, cfg.epsilon)

            # --- Significance hinge or raw penalty ---
            if cfg.use_hinge:
                hinge = torch.nn.functional.softplus(torch.abs(z_s) - self.z_alpha) ** 2
            else:
                # Ablation: replace hinge with raw SP^2
                hinge = sp_est ** 2

            # --- Size weight ---
            if cfg.use_size_weight:
                omega = math.log(1.0 + n_S)
            else:
                omega = 1.0

            fair_loss = fair_loss + omega * hinge

            if collect_diagnostics:
                if cfg.use_hybrid:
                    beta_val = torch.sigmoid(
                        torch.tensor(cfg.kappa * (n_S - cfg.n_tilde))
                    ).item()
                else:
                    beta_val = 1.0
                is_active = torch.abs(z_s).item() > self.z_alpha if cfg.use_hinge else True
                diagnostics[name] = SubgroupDiagnostics(
                    n_S=n_S,
                    sp=sp_est.item(),
                    sigma2=sigma2_est.item(),
                    z_score=z_s.item(),
                    active=is_active,
                    hinge_value=hinge.item(),
                    beta=beta_val,
                )

        # Normalize by number of contributing subgroups (mean instead of sum)
        if n_contributing > 0:
            fair_loss = fair_loss / n_contributing

        return fair_loss.squeeze(), diagnostics

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        subgroup_masks: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, SubgroupDiagnostics]]:
        """Compute the full ASBR loss.

        L = L_pred + lambda * L_fair

        Args:
            y_pred: Model output probabilities, shape (n,).
            y_true: Binary labels, shape (n,).
            subgroup_masks: Dict mapping subgroup names to boolean masks.

        Returns:
            total_loss: Scalar total loss.
            diagnostics: Per-subgroup diagnostics dict.
        """
        pred_loss = self.ce_loss(y_pred, y_true)
        fair_loss, diagnostics = self.compute_fairness_loss(y_pred, subgroup_masks)
        total_loss = pred_loss + self.config.lambd * fair_loss
        return total_loss, diagnostics
