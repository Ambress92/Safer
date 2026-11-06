"""Hybrid meta-learning gradient combining Bayesian and frequentist regimes.

The core of ASBR's size-adaptive estimation: for subgroups with n_S < n_tilde,
the Bayesian Dirichlet posterior dominates; for n_S >= n_tilde, the analytic
Wald-based estimator takes over. The transition is smoothed by a sigmoid
function to ensure training stability.
"""

import torch

from asbr.fairness.wald import (
    compute_asymptotic_variance,
    compute_statistical_parity,
    compute_wald_statistic,
)
from asbr.fairness.bayesian import bayesian_sp_estimate


def compute_transition_weight(
    n_S: int,
    n_tilde: int = 30,
    kappa: float = 0.5,
) -> torch.Tensor:
    """Compute the sigmoid transition weight beta(n_S).

    beta(n_S) = sigmoid(kappa * (n_S - n_tilde))

    Returns ~0 for n_S << n_tilde (Bayesian regime)
    and ~1 for n_S >> n_tilde (frequentist regime).

    Args:
        n_S: Subgroup sample size.
        n_tilde: Switching threshold.
        kappa: Steepness of the sigmoid transition.

    Returns:
        beta: Transition weight in [0, 1] (scalar tensor).
    """
    return torch.sigmoid(torch.tensor(kappa * (n_S - n_tilde), dtype=torch.float))


def compute_hybrid_estimate(
    p_hat: torch.Tensor,
    counts: torch.Tensor,
    n_S: int,
    n_total: int,
    n_tilde: int = 30,
    kappa: float = 0.5,
    dirichlet_prior: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    epsilon: float = 1e-8,
    force_frequentist: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the hybrid SP estimate, variance, and Wald statistic.

    Blends Bayesian and frequentist estimates:
        SP = (1 - beta) * SP_bayes + beta * SP_freq
        sigma2 = (1 - beta) * sigma2_bayes + beta * sigma2_freq

    Args:
        p_hat: Soft count probability vector, shape (4,). Differentiable.
        counts: Hard count vector, shape (4,). Non-differentiable.
        n_S: Subgroup sample size.
        n_total: Total sample size.
        n_tilde: Bayesian switching threshold.
        kappa: Sigmoid steepness.
        dirichlet_prior: Dirichlet prior hyperparameters.
        epsilon: Numerical stability constant.
        force_frequentist: If True, use only the frequentist path (beta=1).

    Returns:
        sp_est: Blended statistical parity estimate (scalar).
        sigma2_est: Blended variance estimate (scalar).
        z_s: Wald statistic computed from the blended estimates (scalar).
    """
    # Frequentist path (differentiable through soft counts)
    sp_freq = compute_statistical_parity(p_hat, epsilon=epsilon)
    sigma2_freq = compute_asymptotic_variance(p_hat, epsilon=epsilon)

    if force_frequentist:
        z_s = compute_wald_statistic(sp_freq, sigma2_freq, n_total, epsilon)
        return sp_freq, sigma2_freq, z_s

    # Bayesian path (uses hard counts; gradient flows through posterior mean)
    sp_bayes, sigma2_bayes = bayesian_sp_estimate(
        counts, alpha_0=dirichlet_prior, epsilon=epsilon
    )

    # Smooth blending
    beta = compute_transition_weight(n_S, n_tilde, kappa)

    sp_est = (1.0 - beta) * sp_bayes + beta * sp_freq
    sigma2_est = (1.0 - beta) * sigma2_bayes + beta * sigma2_freq
    sigma2_est = torch.clamp(sigma2_est, min=epsilon)

    z_s = compute_wald_statistic(sp_est, sigma2_est, n_total, epsilon)

    return sp_est, sigma2_est, z_s
