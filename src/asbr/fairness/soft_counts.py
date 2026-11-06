"""Differentiable soft count computation for fairness metrics.

Converts model output probabilities and subgroup masks into a 4-element
probability vector [p_{1,S}, p_{0,S}, p_{1,Sbar}, p_{0,Sbar}] that is
fully differentiable with respect to model parameters.
"""

import torch


def compute_soft_counts(
    y_pred: torch.Tensor,
    mask_S: torch.Tensor,
) -> torch.Tensor:
    """Compute differentiable probability vector from predictions and subgroup mask.

    Args:
        y_pred: Model output probabilities, shape (n,).
        mask_S: Boolean mask indicating membership in subgroup S, shape (n,).

    Returns:
        p_hat: Probability vector [p_{1,S}, p_{0,S}, p_{1,Sbar}, p_{0,Sbar}],
               shape (4,). All elements sum to 1.
    """
    mask_S_f = mask_S.float()
    mask_Sbar_f = 1.0 - mask_S_f
    n = y_pred.shape[0]

    n1_S = (y_pred * mask_S_f).sum()
    n0_S = ((1.0 - y_pred) * mask_S_f).sum()
    n1_Sbar = (y_pred * mask_Sbar_f).sum()
    n0_Sbar = ((1.0 - y_pred) * mask_Sbar_f).sum()

    p_hat = torch.stack([n1_S, n0_S, n1_Sbar, n0_Sbar]) / n
    return p_hat


def compute_hard_counts(
    y_pred: torch.Tensor,
    mask_S: torch.Tensor,
) -> torch.Tensor:
    """Compute integer counts for the Bayesian estimator.

    Uses thresholded predictions (>= 0.5). Returns a non-differentiable
    count tensor used only by the Bayesian regime.

    Args:
        y_pred: Model output probabilities, shape (n,).
        mask_S: Boolean mask indicating membership in subgroup S, shape (n,).

    Returns:
        counts: Integer count vector [c_{1,S}, c_{0,S}, c_{1,Sbar}, c_{0,Sbar}],
                shape (4,).
    """
    y_binary = (y_pred >= 0.5).float()
    mask_S_f = mask_S.float()
    mask_Sbar_f = 1.0 - mask_S_f

    c1_S = (y_binary * mask_S_f).sum()
    c0_S = ((1.0 - y_binary) * mask_S_f).sum()
    c1_Sbar = (y_binary * mask_Sbar_f).sum()
    c0_Sbar = ((1.0 - y_binary) * mask_Sbar_f).sum()

    return torch.stack([c1_S, c0_S, c1_Sbar, c0_Sbar])
