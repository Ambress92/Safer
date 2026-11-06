"""Wald test components for the frequentist regime of ASBR.

Implements the gradient vector V, covariance matrix Sigma_4, asymptotic
variance sigma^2, statistical parity estimator, and Wald statistic Z_S,
all as differentiable PyTorch operations.
"""

import torch


def compute_gradient_vector(
    p_hat: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute the gradient vector V of the parity function phi.

    V = nabla_p phi, where phi(p) = p_{1,S}/p_S - p_{1,Sbar}/p_Sbar.

    Args:
        p_hat: Probability vector [p_{1,S}, p_{0,S}, p_{1,Sbar}, p_{0,Sbar}],
               shape (4,).
        epsilon: Minimum clamp for group prevalences to avoid division by zero.

    Returns:
        V: Gradient vector, shape (4,).
    """
    p1S, p0S, p1Sbar, p0Sbar = p_hat[0], p_hat[1], p_hat[2], p_hat[3]
    pS = torch.clamp(p1S + p0S, min=epsilon)
    pSbar = torch.clamp(p1Sbar + p0Sbar, min=epsilon)

    V = torch.stack([
        p0S / (pS ** 2),
        -p1S / (pS ** 2),
        -p0Sbar / (pSbar ** 2),
        p1Sbar / (pSbar ** 2),
    ])
    return V


def compute_covariance_matrix(p_hat: torch.Tensor) -> torch.Tensor:
    """Compute the multinomial covariance matrix Sigma_4.

    Sigma_4 = diag(p) - p p^T

    Args:
        p_hat: Probability vector, shape (4,).

    Returns:
        Sigma_4: Covariance matrix, shape (4, 4).
    """
    return torch.diag(p_hat) - torch.outer(p_hat, p_hat)


def compute_asymptotic_variance(
    p_hat: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute sigma^2(S) = V^T Sigma_4 V.

    This is the asymptotic variance of the statistical parity estimator,
    used as the denominator of the Wald test statistic.

    Args:
        p_hat: Probability vector, shape (4,).
        epsilon: Clamp for prevalences passed to compute_gradient_vector.

    Returns:
        sigma2: Scalar asymptotic variance (clamped to be >= epsilon).
    """
    V = compute_gradient_vector(p_hat, epsilon=epsilon)
    Sigma4 = compute_covariance_matrix(p_hat)
    sigma2 = V @ Sigma4 @ V
    return torch.clamp(sigma2, min=epsilon)


def compute_statistical_parity(
    p_hat: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute the differentiable statistical parity estimate.

    SP(S) = p_{1,S}/p_S - p_{1,Sbar}/p_Sbar

    Args:
        p_hat: Probability vector, shape (4,).
        epsilon: Minimum clamp for group prevalences.

    Returns:
        sp: Scalar statistical parity difference.
    """
    p1S, p0S, p1Sbar, p0Sbar = p_hat[0], p_hat[1], p_hat[2], p_hat[3]
    pS = torch.clamp(p1S + p0S, min=epsilon)
    pSbar = torch.clamp(p1Sbar + p0Sbar, min=epsilon)
    return p1S / pS - p1Sbar / pSbar


def compute_wald_statistic(
    sp_n: torch.Tensor,
    sigma2: torch.Tensor,
    n: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute the Wald test statistic Z_S.

    Z_S = sqrt(n) * SP_n(S) / sqrt(sigma^2(S) + epsilon)

    Args:
        sp_n: Differentiable statistical parity estimate (scalar).
        sigma2: Asymptotic variance estimate (scalar).
        n: Total sample size.
        epsilon: Numerical stability constant added to variance.

    Returns:
        z_s: Wald statistic (scalar).
    """
    return (n ** 0.5) * sp_n / torch.sqrt(sigma2 + epsilon)
