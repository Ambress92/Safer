"""Bayesian Dirichlet-multinomial estimator for the small-sample regime.

For subgroups with n_S < n_tilde, the Wald test's asymptotic guarantees
break down. This module provides a Bayesian alternative that uses a
Dirichlet prior to stabilize the statistical parity estimate and its
variance, preventing gradient explosion in sparse intersectional slices.
"""

import torch

from asbr.fairness.wald import compute_gradient_vector


def bayesian_sp_estimate(
    counts: torch.Tensor,
    alpha_0: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SP and variance from the Dirichlet posterior (analytic form).

    Uses the posterior mean of the Dirichlet-multinomial model rather than
    Monte Carlo sampling, providing a deterministic and more stable gradient.

    The posterior is Dir(alpha_0 + counts). The posterior mean of p is:
        E[p | data] = (alpha_0 + counts) / sum(alpha_0 + counts)

    The variance is computed using the population-scale covariance
    (diag(m) - m m^T) — NOT the Dirichlet posterior covariance which
    includes a 1/(alpha_sum+1) factor. This ensures sigma2_bayes is on
    the same scale as sigma2_freq, so the Wald formula Z = sqrt(n)*SP/sqrt(sigma2)
    works correctly in both regimes without double-counting the sample size.

    Args:
        counts: Integer count vector [c_{1,S}, c_{0,S}, c_{1,Sbar}, c_{0,Sbar}],
                shape (4,).
        alpha_0: Dirichlet prior hyperparameters. Default (1,1,1,1) is uniform.
        epsilon: Clamp for prevalences in gradient vector computation.

    Returns:
        sp_posterior: Posterior mean of statistical parity (scalar).
        sigma2_posterior: Posterior variance of statistical parity via
                         Delta Method (scalar).
    """
    alpha_prior = torch.tensor(alpha_0, dtype=counts.dtype, device=counts.device)
    alpha_posterior = alpha_prior + counts.float()
    alpha_sum = alpha_posterior.sum()

    # Posterior mean of p
    p_posterior = alpha_posterior / alpha_sum

    # Statistical parity from posterior mean
    p1S, p0S, p1Sbar, p0Sbar = (
        p_posterior[0],
        p_posterior[1],
        p_posterior[2],
        p_posterior[3],
    )
    pS = torch.clamp(p1S + p0S, min=epsilon)
    pSbar = torch.clamp(p1Sbar + p0Sbar, min=epsilon)
    sp_posterior = p1S / pS - p1Sbar / pSbar

    # Population-scale covariance: Sigma_4 = diag(m) - m m^T
    # (NOT divided by alpha_sum, so sigma2 matches the frequentist scale
    # and can be used directly in the Wald formula Z = sqrt(n)*SP/sqrt(sigma2))
    pop_cov = torch.diag(p_posterior) - torch.outer(p_posterior, p_posterior)

    # Delta Method: sigma2 = V^T Sigma_4 V (population-scale asymptotic variance)
    V = compute_gradient_vector(p_posterior, epsilon=epsilon)
    sigma2_posterior = V @ pop_cov @ V
    sigma2_posterior = torch.clamp(sigma2_posterior, min=epsilon)

    return sp_posterior, sigma2_posterior
