"""ASBR-LRT loss: unified LRT-based fairness regularization.

L_ASBR-LRT(theta) = L_pred + lambda * sum_S omega(n_S) * Psi(sqrt(Lambda_S); alpha)

Replaces the Wald + Bayesian hybrid gradient with a single variance-free
LRT statistic. The significance hinge uses sqrt(Lambda_S) for scale
matching with the Wald Z-score (since Lambda ~ Z^2 asymptotically).

Supports ablation via use_hinge and use_size_weight flags in ASBRConfig.
The use_hybrid flag is ignored (no hybrid gradient needed).

Fairness metric
---------------
The loss supports two metrics controlled by ``ASBRConfig.fairness_metric``:

* ``"sp"`` (default) — Statistical Parity.  ``compute_fairness_loss`` does
  not require ``y_true``; pass only ``y_pred`` and ``subgroup_masks``.

* ``"eo"`` — Equal Opportunity.  ``compute_fairness_loss`` requires
  ``y_true`` (raises ``RuntimeError`` if omitted).  Subgroups with no
  positive ground-truth labels on either side are silently skipped.

Example — train with Equal Opportunity::

    from asbr.experiments.config import ASBRConfig
    from asbr.fairness.asbr_lrt_loss import ASBRLRTLoss

    cfg = ASBRConfig(fairness_metric="eo", lambd=0.3)
    loss_fn = ASBRLRTLoss(cfg)

    # y_true must be passed when fairness_metric="eo"
    total_loss, diag = loss_fn(y_pred, y_true, subgroup_masks)

Via Hydra CLI::

    python -m asbr.experiments.run_hydra \\
        method=asbr_lrt_joint method.asbr.fairness_metric=eo
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from scipy.stats import norm

from asbr.experiments.config import ASBRConfig
from asbr.fairness.lrt import (
    compute_eo_lrt_statistic,
    compute_eo_soft_rates,
    compute_lrt_statistic,
    compute_soft_rates,
)


@dataclass
class LRTDiagnostics:
    """Per-subgroup diagnostics for the LRT-based loss."""

    n_S: int
    sp: float  # Raw SP point estimate (always recorded for monitoring)
    lambda_stat: float  # LRT statistic for the configured fairness_metric
    sqrt_lambda: float  # sqrt(Lambda), comparable to |Z_S|
    active: bool  # Whether the hinge is nonzero
    hinge_value: float
    gate_value: float = 0.0  # sigmoid gate value (gated-SP mode only)
    eo: float = 0.0  # EO point estimate (non-zero only when fairness_metric="eo")


class ASBRLRTLoss(nn.Module):
    """ASBR with Likelihood Ratio Test — unified variance-free formulation.

    Composes:
    - Binary cross-entropy prediction loss
    - Significance hinge on sqrt(Lambda_S) with size-weight

    No Bayesian regime, no hybrid gradient, no transition function.
    The LRT's better finite-sample calibration makes the Bayesian
    correction unnecessary.

    Args:
        config: ASBRConfig with hyperparameters. The fields n_tilde,
                kappa, and dirichlet_prior are ignored.
    """

    def __init__(self, config: ASBRConfig | None = None):
        super().__init__()
        self.config = config or ASBRConfig()
        self.z_alpha = norm.ppf(1.0 - self.config.alpha / 2.0)
        hinge_alpha = (
            self.config.hinge_alpha
            if self.config.hinge_alpha is not None
            else self.config.alpha
        )
        self.z_hinge = float(norm.ppf(1.0 - hinge_alpha / 2.0))
        # Gate threshold — may differ from z_alpha when gate_alpha is set.
        # gate_alpha controls the conservatism spectrum: lower gate_alpha means
        # higher z_gate (stricter gate, fewer groups penalized). When None,
        # defaults to alpha so the gate matches the LRT significance level.
        gate_alpha = (
            self.config.gate_alpha
            if self.config.gate_alpha is not None
            else self.config.alpha
        )
        self.z_gate = float(norm.ppf(1.0 - gate_alpha / 2.0))
        self.ce_loss = nn.BCELoss()

    def compute_fairness_loss(
        self,
        y_pred: torch.Tensor,
        subgroup_masks: dict[str, torch.Tensor],
        y_true: torch.Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, LRTDiagnostics]]:
        """Compute the LRT-based fairness regularization term.

        Args:
            y_pred: Model output probabilities, shape (n,).
            subgroup_masks: Dict mapping subgroup names to boolean masks.
            y_true: Ground-truth binary labels, shape (n,).  Required when
                ``config.fairness_metric="eo"``; ignored for ``"sp"``.
            collect_diagnostics: If False (default), skip all .item() calls
                that force GPU→CPU synchronization. Pass True only when
                per-subgroup diagnostics are actually needed (e.g., final epoch).

        Returns:
            fair_loss: Scalar fairness loss (sum of hinge penalties).
            diagnostics: Per-subgroup LRT diagnostics dict (empty when collect_diagnostics=False).

        Raises:
            RuntimeError: If ``fairness_metric="eo"`` and ``y_true`` is None.
            ValueError: If ``fairness_metric`` is not ``"sp"`` or ``"eo"``.
        """
        cfg = self.config
        n = y_pred.shape[0]
        diagnostics: dict[str, LRTDiagnostics] = {}

        if cfg.fairness_metric not in ("sp", "eo"):
            raise ValueError(
                f"Unknown fairness_metric {cfg.fairness_metric!r}. "
                "Expected 'sp' or 'eo'."
            )
        if cfg.fairness_metric == "eo" and y_true is None:
            raise RuntimeError(
                "fairness_metric='eo' requires y_true to be passed to "
                "compute_fairness_loss(), but y_true is None."
            )
        if cfg.loss_shape not in ("lrt_hinge", "direct_excess", "direct_lrt_excess"):
            raise ValueError(
                f"Unknown loss_shape {cfg.loss_shape!r}. "
                "Expected 'lrt_hinge', 'direct_excess', or 'direct_lrt_excess'."
            )
        if cfg.direct_pred_temperature is not None and cfg.direct_pred_temperature <= 0:
            raise ValueError("direct_pred_temperature must be positive when set.")
        if cfg.direct_lrt_sqrt_min <= 0:
            raise ValueError("direct_lrt_sqrt_min must be positive.")
        if cfg.direct_lrt_se_min < 0:
            raise ValueError("direct_lrt_se_min must be non-negative.")
        if cfg.direct_lrt_se_max < cfg.direct_lrt_se_min:
            raise ValueError("direct_lrt_se_max must be >= direct_lrt_se_min.")
        if cfg.dense_tail_shape not in ("lrt", "metric", "normalized_metric"):
            raise ValueError(
                f"Unknown dense_tail_shape {cfg.dense_tail_shape!r}. "
                "Expected 'lrt', 'metric', or 'normalized_metric'."
            )

        fair_loss = torch.zeros(1, device=y_pred.device, dtype=y_pred.dtype)
        n_contributing = 0

        # MSEE is evaluated on hard thresholded predictions. For the direct
        # surrogate, optionally use a smooth approximation to that hard decision
        # in the fairness counts, rather than calibrated probabilities.
        y_fair = y_pred
        if cfg.loss_shape in ("direct_excess", "direct_lrt_excess") and cfg.direct_pred_temperature is not None:
            temp = torch.tensor(
                cfg.direct_pred_temperature,
                dtype=y_pred.dtype,
                device=y_pred.device,
            )
            threshold = torch.tensor(
                cfg.direct_threshold,
                dtype=y_pred.dtype,
                device=y_pred.device,
            )
            y_fair = torch.sigmoid((y_pred - threshold) / temp)

        # Pre-compute overall rate for use_overall_reference mode.
        # This is the pooled rate across all examples, used as a shared
        # reference point instead of the per-subgroup complement rate.
        overall_rate: torch.Tensor | None = None
        if cfg.use_overall_reference and cfg.use_sp_magnitude:
            eps_f32 = max(float(cfg.epsilon), 1e-6)
            if cfg.fairness_metric == "eo":
                pos_mask = (y_true == 1).float().detach()  # type: ignore[union-attr]
                n_pos_total = pos_mask.sum()
                if n_pos_total > 0:
                    overall_rate = torch.clamp(
                        (y_fair * pos_mask).sum() / n_pos_total,
                        min=eps_f32, max=1.0 - eps_f32,
                    )
            else:
                overall_rate = torch.clamp(
                    y_fair.mean(), min=eps_f32, max=1.0 - eps_f32,
                )

        for name, mask_S in subgroup_masks.items():
            n_S = int(mask_S.sum().item())
            n_Sbar = n - n_S

            if n_S < 2 or n_Sbar < 2:
                continue

            # --- LRT statistic and point estimate (metric-dependent) ---
            if cfg.fairness_metric == "eo":
                pos_mask = (y_true == 1).float().detach()  # type: ignore[union-attr]
                mask_S_f = mask_S.float()
                mask_Sbar_f = 1.0 - mask_S_f
                n_eff_S = (mask_S_f * pos_mask).sum()
                n_eff_Sbar = (mask_Sbar_f * pos_mask).sum()
                lambda_s = compute_eo_lrt_statistic(
                    y_fair, mask_S, y_true, epsilon=cfg.epsilon  # type: ignore[arg-type]
                )
                if lambda_s is None:
                    # No positive labels on one side — skip this subgroup.
                    continue
                rates = compute_eo_soft_rates(
                    y_fair, mask_S, y_true, epsilon=cfg.epsilon  # type: ignore[arg-type]
                )
                if rates is None:
                    continue
                tpr_S, tpr_Sbar, pooled_rate = rates
                if cfg.use_overall_reference and overall_rate is not None:
                    metric_tensor = tpr_S - overall_rate
                else:
                    metric_tensor = tpr_S - tpr_Sbar
                se_tensor = torch.sqrt(
                    tpr_S * (1.0 - tpr_S) / n_eff_S
                    + tpr_Sbar * (1.0 - tpr_Sbar) / n_eff_Sbar
                    + max(float(cfg.epsilon), 1e-6)
                )
            else:
                # SP (default)
                lambda_s = compute_lrt_statistic(y_fair, mask_S, epsilon=cfg.epsilon)
                p_S, p_Sbar, pooled_rate = compute_soft_rates(y_fair, mask_S, epsilon=cfg.epsilon)
                if cfg.use_overall_reference and overall_rate is not None:
                    metric_tensor = p_S - overall_rate
                else:
                    metric_tensor = p_S - p_Sbar
                n_eff_S = mask_S.float().sum()
                n_eff_Sbar = (1.0 - mask_S.float()).sum()
                se_tensor = torch.sqrt(
                    p_S * (1.0 - p_S) / n_eff_S
                    + p_Sbar * (1.0 - p_Sbar) / n_eff_Sbar
                    + max(float(cfg.epsilon), 1e-6)
                )

            n_contributing += 1

            # --- Significance hinge on sqrt(Lambda) for scale matching ---
            sqrt_lambda = torch.sqrt(lambda_s + cfg.epsilon)

            # Branch 0: direct significant-excess surrogate.
            # This is the manuscript's metric-aligned objective in raw
            # disparity units: softplus_tau(|phi_S| - z_alpha * SE_S).
            # It is intentionally separate from the LRT/Z-scale branches below.
            if cfg.loss_shape == "direct_excess":
                se_for_threshold = se_tensor.detach() if cfg.direct_detach_se else se_tensor
                abs_metric = torch.sqrt(
                    metric_tensor ** 2
                    + torch.tensor(
                        cfg.direct_abs_eps ** 2,
                        dtype=metric_tensor.dtype,
                        device=metric_tensor.device,
                    )
                )
                excess = abs_metric - self.z_hinge * se_for_threshold
                tau = torch.tensor(
                    cfg.direct_tau,
                    dtype=metric_tensor.dtype,
                    device=metric_tensor.device,
                )
                hinge = tau * torch.nn.functional.softplus(excess / tau)

            # Branch 0b: LRT-calibrated direct significant-excess surrogate.
            # Derives SE_eff = |phi_S| / sqrt(Lambda_S), replacing the Wald SE
            # with LRT calibration while keeping raw-disparity-unit alignment.
            # Both numerator and denominator are detached (by default) so the
            # threshold is a moving calibration target, not a gradient path.
            elif cfg.loss_shape == "direct_lrt_excess":
                eps2 = torch.tensor(
                    cfg.direct_abs_eps ** 2,
                    dtype=metric_tensor.dtype,
                    device=metric_tensor.device,
                )
                abs_metric = torch.sqrt(metric_tensor ** 2 + eps2)

                sqrt_lambda_ref = (
                    sqrt_lambda.detach() if cfg.direct_lrt_detach else sqrt_lambda
                )
                abs_metric_ref = (
                    abs_metric.detach() if cfg.direct_lrt_detach else abs_metric
                )

                sqrt_lambda_safe = torch.clamp(
                    sqrt_lambda_ref, min=cfg.direct_lrt_sqrt_min
                )
                se_eff = abs_metric_ref / sqrt_lambda_safe
                se_eff = torch.clamp(
                    se_eff, min=cfg.direct_lrt_se_min, max=cfg.direct_lrt_se_max
                )

                excess = abs_metric - self.z_hinge * se_eff
                tau = torch.tensor(
                    cfg.direct_tau,
                    dtype=metric_tensor.dtype,
                    device=metric_tensor.device,
                )
                hinge = tau * torch.nn.functional.softplus(excess / tau)

            # Four legacy branches depending on (use_sp_magnitude, use_hinge):
            #
            #  use_sp_magnitude=T, use_hinge=T  →  metric² · sigmoid(κ(√Λ − z_gate))
            #      Gated metric²: soft sigmoid gate controls selectivity,
            #      metric² (SP² or EO²) controls magnitude.
            #
            #  use_sp_magnitude=T, use_hinge=F  →  metric²
            #      Plain SP²/EO² penalty — the correct "EO-Penalty" baseline.
            #      Used by EOPenaltyMethod / eo_penalty_preset().
            #
            #  use_sp_magnitude=F, use_hinge=T  →  softplus(√Λ − z_hinge)²
            #      ASBR significance hinge on the LRT statistic.
            #      Used by all Gated-EO / Gated-SP variants.
            #
            #  use_sp_magnitude=F, use_hinge=F  →  Λ_S (raw LRT statistic)
            #      Raw LRT penalty — Λ_S ≈ n_S · metric² / variance, so it is
            #      already size- and variance-scaled.  NOT a plain metric² penalty.
            elif cfg.use_sp_magnitude and cfg.use_hinge:
                kappa_t = torch.tensor(
                    cfg.kappa, dtype=sqrt_lambda.dtype, device=sqrt_lambda.device
                )
                gate = torch.sigmoid(kappa_t * (sqrt_lambda - self.z_gate))
                hinge = metric_tensor ** 2 * gate
            elif cfg.use_sp_magnitude:
                # Plain metric² — genuine SP²/EO² penalty, no significance weighting.
                hinge = metric_tensor ** 2
            elif cfg.use_hinge:
                sp_hinge = torch.nn.functional.softplus(sqrt_lambda - self.z_hinge)
                hinge = sp_hinge ** 2 if cfg.use_squared_hinge else sp_hinge
            else:
                # Raw LRT statistic — NOT metric²; already encodes group size and variance.
                hinge = lambda_s

            if cfg.dense_tail_weight > 0:
                if cfg.loss_shape in ("direct_excess", "direct_lrt_excess") and cfg.dense_tail_shape != "lrt":
                    tail = torch.sqrt(
                        metric_tensor ** 2
                        + torch.tensor(
                            cfg.direct_abs_eps ** 2,
                            dtype=metric_tensor.dtype,
                            device=metric_tensor.device,
                        )
                    )
                    if cfg.dense_tail_shape == "normalized_metric":
                        tail = tail / torch.clamp(
                            pooled_rate,
                            min=max(float(cfg.epsilon), 1e-6),
                        )
                else:
                    tail = sqrt_lambda
                hinge = hinge + cfg.dense_tail_weight * tail

            # --- Size weight ---
            if cfg.use_size_weight:
                omega = math.log(1.0 + n_S)
            else:
                omega = 1.0

            aggregate_term = (
                hinge ** 2
                if cfg.loss_shape in ("direct_excess", "direct_lrt_excess") and cfg.use_rms_aggregation
                else hinge
            )
            fair_loss = fair_loss + omega * aggregate_term

            if collect_diagnostics:
                # SP is always recorded for monitoring regardless of metric.
                p_S_sp, p_Sbar_sp, _ = compute_soft_rates(y_pred, mask_S, epsilon=cfg.epsilon)
                sp_val = (p_S_sp - p_Sbar_sp).item()
                eo_val = metric_tensor.item() if cfg.fairness_metric == "eo" else 0.0
                if cfg.loss_shape == "direct_lrt_excess":
                    gate_val = 0.0
                    is_active = bool(sqrt_lambda.item() > self.z_hinge)
                elif cfg.loss_shape == "direct_excess":
                    gate_val = 0.0
                    is_active = bool((torch.abs(metric_tensor) > self.z_hinge * se_tensor).item())
                elif cfg.use_sp_magnitude and cfg.use_hinge:
                    gate_val = gate.item()  # type: ignore[possibly-undefined]
                    is_active = bool(sqrt_lambda.item() > self.z_gate)
                elif cfg.use_sp_magnitude:
                    # Raw metric² — always active, no gate.
                    gate_val = 1.0
                    is_active = True
                elif cfg.use_hinge:
                    gate_val = 0.0
                    is_active = bool(sqrt_lambda.item() > self.z_hinge)
                else:
                    gate_val = 0.0
                    is_active = True
                diagnostics[name] = LRTDiagnostics(
                    n_S=n_S,
                    sp=sp_val,
                    lambda_stat=lambda_s.item(),
                    sqrt_lambda=sqrt_lambda.item(),
                    active=is_active,
                    hinge_value=hinge.item(),
                    gate_value=gate_val,
                    eo=eo_val,
                )

        # Normalize by number of contributing subgroups (mean instead of sum).
        # With use_rms_aggregation=True, take sqrt of the mean (RMS), which
        # self-normalizes the gradient: ∂L/∂EO_S ∝ EO_S / L_fair (bounded)
        # instead of ∂L/∂EO_S ∝ EO_S (unbounded at high λ).
        if n_contributing > 0:
            fair_loss = fair_loss / n_contributing
            if cfg.use_rms_aggregation:
                fair_loss = torch.sqrt(fair_loss + cfg.epsilon)

        return fair_loss.squeeze(), diagnostics

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        subgroup_masks: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, LRTDiagnostics]]:
        """Compute the full ASBR-LRT loss.

        L = L_pred + lambda * L_fair

        Args:
            y_pred: Model output probabilities, shape (n,).
            y_true: Binary labels, shape (n,).  Always used for the BCE
                prediction loss; also forwarded to ``compute_fairness_loss``
                when ``config.fairness_metric="eo"``.
            subgroup_masks: Dict mapping subgroup names to boolean masks.

        Returns:
            total_loss: Scalar total loss.
            diagnostics: Per-subgroup LRT diagnostics dict.
        """
        pred_loss = self.ce_loss(y_pred, y_true)
        fair_loss, diagnostics = self.compute_fairness_loss(
            y_pred, subgroup_masks, y_true=y_true, collect_diagnostics=True
        )
        total_loss = pred_loss + self.config.lambd * fair_loss
        return total_loss, diagnostics
