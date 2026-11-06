"""Likelihood Ratio Test statistics for the ASBR-LRT variant.

Two fairness metrics are supported, each implemented as a differentiable
LRT statistic over soft counts:

Statistical Parity (SP, default)
---------------------------------
Tests H_0: P(y_hat=1 | S) = P(y_hat=1 | Sbar) — equal positive-prediction
rates across groups.  Use ``compute_lrt_statistic``.

Equal Opportunity (EO)
-----------------------
Tests H_0: P(y_hat=1 | y=1, S) = P(y_hat=1 | y=1, Sbar) — equal true
positive rates (TPR) across groups, conditioning on positive ground-truth
labels.  Use ``compute_eo_lrt_statistic`` and ``compute_eo_soft_rates``.

To train with EO instead of SP, set ``fairness_metric="eo"`` in
``ASBRConfig``::

    from asbr.experiments.config import ASBRConfig
    cfg = ASBRConfig(fairness_metric="eo", lambd=0.3)

or via the Hydra CLI::

    python -m asbr.experiments.run_hydra \\
        method=asbr_lrt_joint method.asbr.fairness_metric=eo

Both statistics follow Lambda -> chi^2_1 under H_0 (Wilks' theorem) and
use the same significance hinge ``max(0, sqrt(Lambda) - z_alpha)^2``, so
the same lambda and alpha hyperparameters transfer across metrics.

Note: EO skips subgroups that have no positive ground-truth labels in
either S or Sbar (``compute_eo_soft_rates`` returns None in that case).
"""

from __future__ import annotations

import torch


def compute_soft_rates(
    y_pred: torch.Tensor,
    mask_S: torch.Tensor,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute differentiable soft positive rates for S, Sbar, and pooled.

    Args:
        y_pred: Model output probabilities, shape (n,).
        mask_S: Boolean mask for subgroup S, shape (n,).
        epsilon: Clamping bound to keep rates in (epsilon, 1-epsilon).

    Returns:
        p_S: Soft positive rate in S.
        p_Sbar: Soft positive rate in Sbar.
        p_pooled: Pooled soft positive rate.
    """
    mask_S_f = mask_S.float()
    mask_Sbar_f = 1.0 - mask_S_f

    n_S = mask_S_f.sum()
    n_Sbar = mask_Sbar_f.sum()

    # Soft positive counts
    n1_S = (y_pred * mask_S_f).sum()
    n1_Sbar = (y_pred * mask_Sbar_f).sum()

    # Float32-safe epsilon: values < ~1.19e-7 cannot be represented distinctly
    # from 0 or 1 in float32, causing 0*inf=NaN in the LRT log terms.
    eps_f32 = max(float(epsilon), 1e-6)

    p_S = torch.clamp(n1_S / n_S, min=eps_f32, max=1.0 - eps_f32)
    p_Sbar = torch.clamp(n1_Sbar / n_Sbar, min=eps_f32, max=1.0 - eps_f32)
    p_pooled = torch.clamp(
        (n1_S + n1_Sbar) / (n_S + n_Sbar), min=eps_f32, max=1.0 - eps_f32
    )

    return p_S, p_Sbar, p_pooled


def compute_lrt_statistic(
    y_pred: torch.Tensor,
    mask_S: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute the Likelihood Ratio Test statistic for equal positive rates.

    Tests H_0: P(y_hat=1 | S) = P(y_hat=1 | Sbar) using the generalized
    LRT with soft counts for differentiability.

    Lambda = -2 [ n1_S log(p/p_S) + n0_S log((1-p)/(1-p_S))
                 + n1_Sb log(p/p_Sb) + n0_Sb log((1-p)/(1-p_Sb)) ]

    Under H_0: Lambda -> chi^2_1 (Wilks' theorem).

    Args:
        y_pred: Model output probabilities, shape (n,).
        mask_S: Boolean mask for subgroup S, shape (n,).
        epsilon: Clamping bound for numerical stability.

    Returns:
        lambda_stat: LRT statistic (scalar, non-negative).
    """
    mask_S_f = mask_S.float()
    mask_Sbar_f = 1.0 - mask_S_f

    n_S = mask_S_f.sum()
    n_Sbar = mask_Sbar_f.sum()

    # Soft counts (differentiable through y_pred)
    n1_S = (y_pred * mask_S_f).sum()
    n0_S = ((1.0 - y_pred) * mask_S_f).sum()
    n1_Sbar = (y_pred * mask_Sbar_f).sum()
    n0_Sbar = ((1.0 - y_pred) * mask_Sbar_f).sum()

    # Float32-safe epsilon: 1e-8 < float32 machine epsilon near 1.0 (~1.19e-7),
    # so clamp(x, max=1-1e-8) has no effect in float32 and (1-p) can be exactly
    # 0.0.  Use at least 1e-6 so the log never sees 0/0 arguments.
    eps_f32 = max(float(epsilon), 1e-6)

    # Soft positive rates (clamped for numerical safety)
    p_S = torch.clamp(n1_S / n_S, min=eps_f32, max=1.0 - eps_f32)
    p_Sbar = torch.clamp(n1_Sbar / n_Sbar, min=eps_f32, max=1.0 - eps_f32)
    p_pooled = torch.clamp(
        (n1_S + n1_Sbar) / (n_S + n_Sbar), min=eps_f32, max=1.0 - eps_f32
    )

    # Log-likelihood ratio terms
    lambda_stat = -2.0 * (
        n1_S * torch.log(p_pooled / p_S)
        + n0_S * torch.log((1.0 - p_pooled) / (1.0 - p_S))
        + n1_Sbar * torch.log(p_pooled / p_Sbar)
        + n0_Sbar * torch.log((1.0 - p_pooled) / (1.0 - p_Sbar))
    )

    # Lambda >= 0 by construction; clamp for numerical safety
    return torch.clamp(lambda_stat, min=0.0)


def compute_eo_soft_rates(
    y_pred: torch.Tensor,
    mask_S: torch.Tensor,
    y_true: torch.Tensor,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Compute differentiable soft TPR rates for S, Sbar, and pooled.

    Restricts to ground-truth positive samples (y_true == 1) before
    computing rates, implementing Equal Opportunity instead of Statistical
    Parity.  Gradient flows through y_pred; y_true is treated as a fixed
    mask (no gradient).

    Args:
        y_pred: Model output probabilities, shape (n,).
        mask_S: Boolean mask for subgroup S, shape (n,).
        y_true: Ground-truth binary labels, shape (n,).  Must be float or
            int; values are compared to 1.0.
        epsilon: Clamping bound to keep rates in (epsilon, 1-epsilon).

    Returns:
        (tpr_S, tpr_Sbar, tpr_pooled) if both sides have at least one
        positive label; None otherwise (caller must skip this subgroup).
    """
    eps_f32 = max(float(epsilon), 1e-6)

    # Positive-sample masks — detach so no gradient flows through y_true.
    pos_mask = (y_true == 1).float().detach()
    pos_mask_S = mask_S.float() * pos_mask
    pos_mask_Sbar = (1.0 - mask_S.float()) * pos_mask

    n_pos_S = pos_mask_S.sum()
    n_pos_Sbar = pos_mask_Sbar.sum()

    # Skip subgroups where either side has no positive labels.
    if n_pos_S.item() < 1 or n_pos_Sbar.item() < 1:
        return None

    # Soft TP counts (differentiable through y_pred).
    tp_S = (y_pred * pos_mask_S).sum()
    tp_Sbar = (y_pred * pos_mask_Sbar).sum()

    tpr_S = torch.clamp(tp_S / n_pos_S, min=eps_f32, max=1.0 - eps_f32)
    tpr_Sbar = torch.clamp(tp_Sbar / n_pos_Sbar, min=eps_f32, max=1.0 - eps_f32)
    tpr_pooled = torch.clamp(
        (tp_S + tp_Sbar) / (n_pos_S + n_pos_Sbar), min=eps_f32, max=1.0 - eps_f32
    )

    return tpr_S, tpr_Sbar, tpr_pooled


def compute_eo_lrt_statistic(
    y_pred: torch.Tensor,
    mask_S: torch.Tensor,
    y_true: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor | None:
    """Compute the LRT statistic for Equal Opportunity.

    Tests H_0: P(y_hat=1 | y=1, S) = P(y_hat=1 | y=1, Sbar) using soft
    counts restricted to ground-truth positive samples.  The 4-cell table
    is [TP_S, FN_S, TP_Sbar, FN_Sbar] (true positives and false negatives),
    analogous to the SP table [pos_S, neg_S, pos_Sbar, neg_Sbar].

    The same LRT formula applies:

        Lambda = -2 [ TP_S log(tpr/tpr_S) + FN_S log((1-tpr)/(1-tpr_S))
                    + TP_Sb log(tpr/tpr_Sb) + FN_Sb log((1-tpr)/(1-tpr_Sb)) ]

    Under H_0: Lambda -> chi^2_1 (Wilks' theorem), so the same significance
    hinge and z_alpha threshold as the SP variant apply.

    Args:
        y_pred: Model output probabilities, shape (n,).
        mask_S: Boolean mask for subgroup S, shape (n,).
        y_true: Ground-truth binary labels, shape (n,).
        epsilon: Clamping bound for numerical stability.

    Returns:
        lambda_stat: LRT statistic (scalar, non-negative), or None if either
            side of the subgroup has no positive ground-truth labels (caller
            must skip this subgroup).
    """
    eps_f32 = max(float(epsilon), 1e-6)

    pos_mask = (y_true == 1).float().detach()
    pos_mask_S = mask_S.float() * pos_mask
    pos_mask_Sbar = (1.0 - mask_S.float()) * pos_mask

    n_pos_S = pos_mask_S.sum()
    n_pos_Sbar = pos_mask_Sbar.sum()

    if n_pos_S.item() < 1 or n_pos_Sbar.item() < 1:
        return None

    # Soft TP / FN counts (differentiable through y_pred).
    tp_S = (y_pred * pos_mask_S).sum()
    fn_S = ((1.0 - y_pred) * pos_mask_S).sum()
    tp_Sbar = (y_pred * pos_mask_Sbar).sum()
    fn_Sbar = ((1.0 - y_pred) * pos_mask_Sbar).sum()

    # Soft TPR rates (clamped for numerical safety).
    tpr_S = torch.clamp(tp_S / n_pos_S, min=eps_f32, max=1.0 - eps_f32)
    tpr_Sbar = torch.clamp(tp_Sbar / n_pos_Sbar, min=eps_f32, max=1.0 - eps_f32)
    tpr_pooled = torch.clamp(
        (tp_S + tp_Sbar) / (n_pos_S + n_pos_Sbar), min=eps_f32, max=1.0 - eps_f32
    )

    lambda_stat = -2.0 * (
        tp_S * torch.log(tpr_pooled / tpr_S)
        + fn_S * torch.log((1.0 - tpr_pooled) / (1.0 - tpr_S))
        + tp_Sbar * torch.log(tpr_pooled / tpr_Sbar)
        + fn_Sbar * torch.log((1.0 - tpr_pooled) / (1.0 - tpr_Sbar))
    )

    return torch.clamp(lambda_stat, min=0.0)
