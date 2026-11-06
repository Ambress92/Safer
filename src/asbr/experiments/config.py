"""Central configuration for ASBR experiments.

Fairness metric selection
--------------------------
ASBR-LRT supports two fairness metrics via ``ASBRConfig.fairness_metric``:

* ``"sp"`` (default) — Statistical Parity: penalises subgroups where
  ``P(ŷ=1 | S) ≠ P(ŷ=1 | S̄)``.

* ``"eo"`` — Equal Opportunity: penalises subgroups where
  ``P(ŷ=1 | y=1, S) ≠ P(ŷ=1 | y=1, S̄)`` (TPR disparity).
  Only subgroups with at least one positive ground-truth label on each
  side contribute to the loss.

To switch metric from the Hydra CLI::

    python -m asbr.experiments.run_hydra \\
        method=asbr_lrt_joint method.asbr.fairness_metric=eo

Or in Python::

    from asbr.experiments.config import ASBRConfig
    cfg = ASBRConfig(fairness_metric="eo", lambd=0.3)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ASBRConfig:
    """Configuration for the ASBR fairness loss function.

    Controls the significance hinge, hybrid gradient, and size-weight components.
    Ablation variants are created by toggling the use_* flags.
    """

    # Significance level for the Wald test
    alpha: float = 0.05
    # Regularization strength (lambda in the paper)
    lambd: float = 1.0
    # Bayesian switching threshold (n_tilde in the paper)
    n_tilde: int = 30
    # Transition steepness for sigmoid switch (kappa in the paper)
    kappa: float = 0.5
    # Dirichlet prior hyperparameters (uniform by default)
    dirichlet_prior: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    # Numerical stability constant
    epsilon: float = 1e-8

    # --- Fairness metric ---
    # "sp": Statistical Parity (default) — penalises positive-rate disparity.
    # "eo": Equal Opportunity — penalises TPR disparity (requires y_true at
    #        training time; subgroups with no positives on either side are skipped).
    fairness_metric: str = "sp"

    # --- Ablation flags ---
    # When False, replaces significance hinge with raw SP^2 penalty
    use_hinge: bool = True
    # When False, forces frequentist-only estimation (beta=1 for all groups)
    use_hybrid: bool = True
    # When False, sets omega(n_S) = 1 for all groups
    use_size_weight: bool = True
    # When True with use_hinge=True: use SP² as penalty magnitude, gated by
    # sigmoid(kappa * (sqrt(Lambda) - z_alpha)) instead of the raw hinge
    # (sqrt(Lambda) - z_alpha)². Separates the significance gate from the
    # penalty scale so large-group disparities are not over-penalized.
    use_sp_magnitude: bool = False
    # Significance level for the GATE threshold in gated-SP mode.
    # When None, falls back to alpha (same threshold as the LRT evaluation).
    # Controls the conservatism spectrum:
    #   gate_alpha → 0     (z_gate → ∞):  nothing penalized (no-op)
    #   gate_alpha = 0.05  (z_gate=1.96): standard significance gate (default)
    #   gate_alpha = 0.20  (z_gate=1.28): moderate — penalizes borderline groups
    #   gate_alpha → 1.0   (z_gate → 0):  everything penalized ≈ SP-Penalty
    # Only used when use_sp_magnitude=True and use_hinge=True.
    gate_alpha: float | None = None
    # Optional significance level for the hinge threshold itself. When None,
    # falls back to alpha. Setting hinge_alpha=1.0 yields z_hinge=0, which
    # removes the significance gate while preserving the same softplus hinge
    # shape on sqrt(Lambda).
    hinge_alpha: float | None = None
    # When True (default), the hinge is softplus(√Λ - z_α)² (quadratic in
    # excess u). When False, uses softplus(√Λ - z_α) (linear in u), which is
    # perfectly MSPE-aligned since MSPE's target shape is linear in excess.
    use_squared_hinge: bool = True
    # Optional weak dense tail added to the significance hinge. When > 0,
    # the fairness penalty becomes hinge + dense_tail_weight * √Λ, which keeps
    # a nonzero sub-threshold gradient without removing threshold awareness.
    dense_tail_weight: float = 0.0
    # Shape for the optional dense tail. "lrt" keeps the existing √Λ tail.
    # For direct_excess, "metric" uses |phi_S| and "normalized_metric" uses
    # |phi_S| / pooled_rate, which is closer to fairret's EO norm.
    dense_tail_shape: str = "lrt"
    # Fairness penalty shape used by ASBRLRTLoss.
    # "lrt_hinge": existing SAFER/LRT hinge on sqrt(Lambda).
    # "direct_excess": manuscript direct surrogate in raw disparity units:
    #   softplus_tau(|phi_S| - z_alpha * SE_S).  Uses Wald-type SE.
    # "direct_lrt_excess": LRT-calibrated variant of direct_excess.  Derives an
    #   effective SE from the LRT statistic: SE_eff = |phi_S| / sqrt(Lambda_S).
    #   Replaces the fragile Wald SE with LRT calibration while keeping the same
    #   raw-disparity-unit penalty structure.  Theoretically clean only for
    #   complement-reference mode (use_overall_reference=false), where the metric
    #   and the LRT test the same S-vs-Sbar contrast.
    loss_shape: str = "lrt_hinge"
    # Temperature for the direct-excess softplus approximation.
    direct_tau: float = 0.01
    # Smoothing for |phi_S| in the direct-excess surrogate.
    direct_abs_eps: float = 1e-3
    # Detach SE in the direct-excess gradient while keeping the same forward
    # threshold. This prevents the optimizer from reducing loss by inflating
    # uncertainty instead of reducing disparity.
    direct_detach_se: bool = True
    # For direct_excess, optionally compute fairness counts on a differentiable
    # approximation of the hard classifier used by MSEE:
    # sigmoid((p - direct_threshold) / direct_pred_temperature). When null,
    # raw probabilities p are used.
    direct_pred_temperature: float | None = None
    direct_threshold: float = 0.5
    # --- direct_lrt_excess settings ---
    # Whether to detach both |phi_S| and sqrt(Lambda_S) when computing SE_eff.
    # When True (default/recommended), gradient flows only through the
    # abs_metric term of the excess, so the loss always pushes disparity down.
    # Setting to False is for diagnostic/ablation only: it re-enables the
    # sign-reversal risk near threshold where the negative multiplier
    # (1 - z/sqrt_Lambda) can cause gradient descent to increase disparity.
    direct_lrt_detach: bool = True
    # Floor for sqrt(Lambda_S) before computing the ratio |phi_S|/sqrt(Lambda_S).
    # The ratio is sensitive near zero; a larger floor (e.g. 1e-3) limits the
    # maximum SE_eff and reduces dead zones for very weak-statistic subgroups.
    direct_lrt_sqrt_min: float = 1e-4
    # Clamp bounds for the LRT-derived effective SE.  Prevents dead zones
    # (huge se_eff → strongly negative excess → only dense tail acts) and
    # over-eager thresholds (tiny se_eff near saturated rates).
    direct_lrt_se_min: float = 0.0
    direct_lrt_se_max: float = 1.0
    # When True, aggregates subgroup penalties as RMS (sqrt of mean of squares)
    # instead of plain mean. This self-normalizes the fairness gradient:
    #   ∂L_fair/∂EO_S ∝ EO_S / L_fair  (bounded regardless of violation magnitude)
    # vs plain mean:
    #   ∂L_fair/∂EO_S ∝ EO_S            (grows unboundedly at high λ)
    # Recommended for the gated-SP/EO joint variant to prevent BCE loss collapse
    # at high lambda values.
    use_rms_aggregation: bool = False
    # When True, the penalty metric uses (metric_S - metric_overall) instead of
    # (metric_S - metric_Sbar). The complement-based formulation amplifies
    # large-group disparities by n/(n-n_S), which can reach 5-7× on imbalanced
    # datasets (e.g. COMPAS sex=Male is 80% of data). The overall-based
    # formulation eliminates this implicit size amplification, distributing
    # gradient more uniformly across subgroups. Only affects the penalty
    # magnitude when use_sp_magnitude=True; the LRT statistic (used for the
    # gate) still tests S vs Sbar.
    use_overall_reference: bool = False

    @classmethod
    def sp_penalty_preset(cls) -> ASBRConfig:
        """SP² penalty baseline: raw squared parity without significance testing."""
        return cls(use_hinge=False, use_hybrid=False, use_size_weight=False)

    @classmethod
    def eo_penalty_preset(cls) -> ASBRConfig:
        """EO² penalty baseline: raw squared TPR-disparity without significance testing.

        Uses ``use_sp_magnitude=True, use_hinge=False`` to route to the
        ``metric_tensor**2`` branch in ASBRLRTLoss, producing a penalty of
        exactly EO²(S) per subgroup.  Without ``use_sp_magnitude=True`` the
        ``use_hinge=False`` branch falls through to the raw LRT statistic Λ_S,
        which is already size- and variance-scaled — not a plain EO² penalty.
        """
        return cls(
            fairness_metric="eo",
            use_hinge=False,
            use_hybrid=False,
            use_size_weight=False,
            use_sp_magnitude=True,
        )


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    lr: float = 1e-3
    patience: int = 10
    max_epochs: int = 200
    batch_size: int = 256
    max_grad_norm: float = 1.0
    weight_decay: float = 5e-4
    # JointTrainer only: number of BCE-only epochs before applying the
    # combined BCE + fairness objective. The best warmup validation-BCE
    # checkpoint is restored before the joint phase starts.
    bce_warmup_epochs: int = 0
    # JointTrainer checkpointing. "val_bce" preserves the historical selector.
    # "bce_guarded_fairness" selects the lowest λ-weighted objective among
    # epochs whose validation BCE remains within joint_bce_guard_rel_tol of
    # the BCE anchor. This is useful when BCE warmup is used with high lambda:
    # otherwise the best-BCE checkpoint is often the first post-warmup epoch.
    joint_checkpoint_mode: str = "val_bce"
    joint_bce_guard_rel_tol: float = 0.05
    joint_bce_guard_min_lambd: float = 0.0
    # JointTrainer only: ramp the effective fairness weight from 0 to lambda
    # over the first N joint epochs. Unlike BCE-only warmup, this lets BCE and
    # fairness shape the boundary together while avoiding random-weight fairness
    # gradients at the very beginning.
    fairness_ramp_epochs: int = 0
    fairness_ramp_shape: str = "linear"
    fairness_ramp_min_lambd: float = 0.0
    # ReduceLROnPlateau settings (JointTrainer only)
    lr_reduce_factor: float = 0.1
    min_lr: float = 1e-5
    # When True, Phase 2 fairness loss is normalized by bce_phase1/fair_loss_initial
    # so that λ=1 means equal weight to BCE at the start of Phase 2.
    normalize_phase2: bool = False
