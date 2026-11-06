"""Size-Adaptive Fairness Testing (SAFT) — post-hoc assessment module.

SAFT audits a trained model's hard predictions for statistically significant
statistical parity violations. For each subgroup it tests H₀: SP(S) = 0 using:
- Wald test for large subgroups (min cell count ≥ n_tilde)
- Bayesian Dirichlet-MC test for small subgroups (min cell count < n_tilde)

Two rejection methods are supported:
- ``"p_value"``: reject H₀ when p-value < alpha (default).
- ``"ci"``: reject H₀ when 0 lies outside the confidence/credible interval.

For continuous distributions the two methods are theoretically equivalent.

This reuses the same statistical foundations as ASBR (wald.py, bayesian.py)
but operates on hard binary predictions rather than soft counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from typing import Literal

from asbr.fairness.wald import (
    compute_asymptotic_variance,
    compute_statistical_parity,
)

RejectionMethod = Literal["p_value", "ci"]


@dataclass
class SAFTResult:
    """Per-subgroup SAFT test result."""

    subgroup: str
    n_S: int
    min_cell_count: int
    sp_estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    reject_h0: bool
    test_type: str  # "wald" or "bayesian"


@dataclass
class SAFTSummary:
    """Aggregate SAFT audit summary."""

    n_subgroups: int
    n_violations: int
    violation_rate: float
    violations_by_size_bucket: dict[str, dict[str, int | float]]
    mean_sp_among_violations: float
    mspe: float  # Mean Significant Parity Excess (see compute_mspe)


def compute_hard_counts(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask_S: np.ndarray,
) -> np.ndarray:
    """Compute the 4-cell contingency counts from hard binary predictions.

    The cells are: [n_{1,S}, n_{0,S}, n_{1,Sbar}, n_{0,Sbar}]
    where 1/0 refers to the predicted label and S/Sbar to group membership.

    Note: for SP testing we use y_pred (not y_true) as the "outcome" — we are
    testing whether the model's acceptance rate differs across groups.

    Args:
        y_pred: Binary predictions, shape (n,).
        y_true: True labels, shape (n,). Not used in SP computation but
                kept for API consistency.
        mask_S: Boolean mask for subgroup S, shape (n,).

    Returns:
        counts: Integer array [c_{1,S}, c_{0,S}, c_{1,Sbar}, c_{0,Sbar}].
    """
    mask_Sbar = ~mask_S
    c1S = int(np.sum(y_pred[mask_S] == 1))
    c0S = int(np.sum(y_pred[mask_S] == 0))
    c1Sbar = int(np.sum(y_pred[mask_Sbar] == 1))
    c0Sbar = int(np.sum(y_pred[mask_Sbar] == 0))
    return np.array([c1S, c0S, c1Sbar, c0Sbar])


def _wald_test(
    p_hat_np: np.ndarray,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Wald CI and two-sided p-value for SP(S) = 0.

    Args:
        p_hat_np: Probability vector (4,) as numpy.
        n: Total sample size.
        alpha: Significance level.

    Returns:
        (sp, ci_lower, ci_upper, p_value)
    """
    import torch

    p_hat = torch.tensor(p_hat_np, dtype=torch.float64)
    sp = compute_statistical_parity(p_hat).item()
    sigma2 = compute_asymptotic_variance(p_hat).item()
    se = (sigma2 / n) ** 0.5

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = sp - z_crit * se
    ci_upper = sp + z_crit * se

    if se > 0:
        z_stat = abs(sp) / se
        p_value = 2 * (1 - stats.norm.cdf(z_stat))
    else:
        p_value = 1.0 if abs(sp) < 1e-12 else 0.0

    return sp, ci_lower, ci_upper, float(p_value)


def _bayesian_mc_test(
    counts: np.ndarray,
    alpha: float = 0.05,
    K: int = 10000,
    alpha_0: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, float]:
    """Bayesian Dirichlet-MC test for SP(S) = 0.

    Samples from the Dirichlet posterior and computes the empirical
    distribution of SP, yielding a credible interval and posterior
    tail probability.

    Args:
        counts: Integer count vector [c_{1,S}, c_{0,S}, c_{1,Sbar}, c_{0,Sbar}].
        alpha: Significance level for credible interval.
        K: Number of MC samples.
        alpha_0: Dirichlet prior hyperparameters.
        rng: NumPy random generator for reproducibility.

    Returns:
        (sp_mean, ci_lower, ci_upper, p_value)
    """
    if rng is None:
        rng = np.random.default_rng()

    alpha_posterior = np.array(alpha_0) + counts.astype(float)
    samples = rng.dirichlet(alpha_posterior, size=K)  # (K, 4)

    # SP for each sample: p_{1,S}/p_S - p_{1,Sbar}/p_Sbar
    eps = 1e-12
    p_S = samples[:, 0] + samples[:, 1]
    p_Sbar = samples[:, 2] + samples[:, 3]
    p_S = np.maximum(p_S, eps)
    p_Sbar = np.maximum(p_Sbar, eps)

    sp_samples = samples[:, 0] / p_S - samples[:, 2] / p_Sbar

    sp_mean = float(np.mean(sp_samples))
    ci_lower = float(np.percentile(sp_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(sp_samples, 100 * (1 - alpha / 2)))

    # Posterior tail probability: P(SP ≤ 0 | data) or P(SP ≥ 0 | data)
    # whichever is smaller, doubled for two-sided test
    p_positive = np.mean(sp_samples > 0)
    p_negative = np.mean(sp_samples < 0)
    p_value = float(2 * min(p_positive, p_negative))
    p_value = min(p_value, 1.0)

    return sp_mean, ci_lower, ci_upper, p_value


def _reject_h0(
    p_value: float,
    ci_lower: float,
    ci_upper: float,
    alpha: float,
    method: RejectionMethod,
) -> bool:
    """Decide whether to reject H₀: SP(S) = 0.

    Args:
        p_value: Two-sided p-value (or posterior tail probability).
        ci_lower: Lower bound of confidence/credible interval.
        ci_upper: Upper bound of confidence/credible interval.
        alpha: Significance level.
        method: ``"p_value"`` for p-value thresholding,
                ``"ci"`` for CI containment of zero.

    Returns:
        True if H₀ is rejected.
    """
    if method == "p_value":
        return p_value < alpha
    elif method == "ci":
        return ci_lower > 0 or ci_upper < 0
    else:
        raise ValueError(f"Unknown rejection method: {method!r}")


def saft_test_subgroup(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask_S: np.ndarray,
    subgroup_name: str,
    alpha: float = 0.05,
    n_tilde: int = 30,
    mc_samples: int = 10000,
    rng: np.random.Generator | None = None,
    rejection_method: RejectionMethod = "p_value",
) -> SAFTResult:
    """Run SAFT on a single subgroup.

    Uses a hard switch: if min cell count ≥ n_tilde, uses Wald test;
    otherwise uses Bayesian MC test.

    Args:
        y_pred: Binary predictions, shape (n,).
        y_true: True labels, shape (n,).
        mask_S: Boolean subgroup mask, shape (n,).
        subgroup_name: Name of the subgroup.
        alpha: Significance level.
        n_tilde: Switching threshold (on min cell count).
        mc_samples: Number of MC samples for Bayesian test.
        rng: NumPy random generator for reproducibility.
        rejection_method: ``"p_value"`` (default) or ``"ci"``
            (CI containment of zero).

    Returns:
        SAFTResult for this subgroup.
    """
    counts = compute_hard_counts(y_pred, y_true, mask_S)
    n = int(counts.sum())
    n_S = int(counts[0] + counts[1])
    n_Sbar = n - n_S
    min_cell = int(counts.min())

    # Short-circuit: if the observed acceptance rates are identical, the
    # data show zero disparity and we must not reject.  Without this check
    # the Bayesian prior pseudocounts can fabricate phantom SP in small
    # subgroups (e.g. when the model predicts a constant label).
    eps_sp = 1e-12
    if n_S > 0 and n_Sbar > 0:
        rate_S = counts[0] / n_S
        rate_Sbar = counts[2] / n_Sbar
        observed_sp = rate_S - rate_Sbar
    else:
        observed_sp = 0.0

    if abs(observed_sp) < eps_sp:
        return SAFTResult(
            subgroup=subgroup_name,
            n_S=n_S,
            min_cell_count=min_cell,
            sp_estimate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
            reject_h0=False,
            test_type="exact_zero",
        )

    if min_cell >= n_tilde:
        p_hat = counts / n
        sp, ci_lo, ci_hi, p_val = _wald_test(p_hat, n, alpha)
        test_type = "wald"
    else:
        sp, ci_lo, ci_hi, p_val = _bayesian_mc_test(
            counts, alpha, mc_samples, rng=rng,
        )
        test_type = "bayesian"

    return SAFTResult(
        subgroup=subgroup_name,
        n_S=n_S,
        min_cell_count=min_cell,
        sp_estimate=sp,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p_value=p_val,
        reject_h0=_reject_h0(p_val, ci_lo, ci_hi, alpha, rejection_method),
        test_type=test_type,
    )


def saft_audit(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    subgroup_masks: dict[str, np.ndarray],
    alpha: float = 0.05,
    n_tilde: int = 30,
    mc_samples: int = 10000,
    seed: int = 42,
    rejection_method: RejectionMethod = "ci",
) -> list[SAFTResult]:
    """Run SAFT on all subgroups.

    Args:
        y_pred: Binary predictions, shape (n,).
        y_true: True labels, shape (n,).
        subgroup_masks: Dict mapping subgroup name -> boolean mask.
        alpha: Significance level.
        n_tilde: Switching threshold.
        mc_samples: Number of MC samples for Bayesian test.
        seed: Random seed for Bayesian MC sampling.
        rejection_method: ``"p_value"`` (default) or ``"ci"``
            (CI containment of zero).

    Returns:
        List of SAFTResult, one per subgroup.
    """
    rng = np.random.default_rng(seed)
    results = []
    for name, mask in subgroup_masks.items():
        result = saft_test_subgroup(
            y_pred, y_true, mask, name,
            alpha=alpha, n_tilde=n_tilde,
            mc_samples=mc_samples, rng=rng,
            rejection_method=rejection_method,
        )
        results.append(result)
    return results


def apply_multiplicity_correction(
    results: list[SAFTResult],
    method: Literal["bonferroni", "bh"] = "bh",
    alpha: float = 0.05,
) -> list[SAFTResult]:
    """Re-evaluate rejection decisions with a multiple-testing correction.

    When auditing many subgroups simultaneously, uncorrected per-test α
    inflates the family-wise error rate. This function adjusts the rejection
    decisions in-place using either Bonferroni (FWER control) or
    Benjamini-Hochberg (FDR control).

    Note: for results obtained with ``rejection_method="ci"``, the CI bounds
    are *not* widened — only the ``reject_h0`` flag is updated based on the
    p-values.  This is valid because the CI and p-value decisions are
    equivalent for a given α (see the comparison report).

    Args:
        results: List of SAFTResult from ``saft_audit``.
        method: ``"bonferroni"`` for FWER control, ``"bh"`` for
            Benjamini-Hochberg FDR control (default).
        alpha: Family-wise or FDR significance level.

    Returns:
        New list of SAFTResult with corrected ``reject_h0`` flags.
    """
    m = len(results)
    if m == 0:
        return []

    if method == "bonferroni":
        adj_alpha = alpha / m
        corrected = []
        for r in results:
            corrected.append(SAFTResult(
                subgroup=r.subgroup,
                n_S=r.n_S,
                min_cell_count=r.min_cell_count,
                sp_estimate=r.sp_estimate,
                ci_lower=r.ci_lower,
                ci_upper=r.ci_upper,
                p_value=r.p_value,
                reject_h0=(r.p_value < adj_alpha),
                test_type=r.test_type,
            ))
        return corrected

    elif method == "bh":
        # Benjamini-Hochberg procedure
        indexed = sorted(enumerate(results), key=lambda t: t[1].p_value)
        reject_flags = [False] * m

        # Find the largest k such that p_(k) <= k/m * alpha
        max_k = 0
        for rank, (orig_idx, r) in enumerate(indexed, start=1):
            threshold = rank / m * alpha
            if r.p_value <= threshold:
                max_k = rank

        # Reject all hypotheses with rank <= max_k
        for rank, (orig_idx, _) in enumerate(indexed, start=1):
            if rank <= max_k:
                reject_flags[orig_idx] = True

        corrected = []
        for i, r in enumerate(results):
            corrected.append(SAFTResult(
                subgroup=r.subgroup,
                n_S=r.n_S,
                min_cell_count=r.min_cell_count,
                sp_estimate=r.sp_estimate,
                ci_lower=r.ci_lower,
                ci_upper=r.ci_upper,
                p_value=r.p_value,
                reject_h0=reject_flags[i],
                test_type=r.test_type,
            ))
        return corrected

    else:
        raise ValueError(f"Unknown correction method: {method!r}")


def compute_mspe(results: list[SAFTResult]) -> float:
    """Mean Significant Parity Excess (MSPE) — a single continuous fairness metric.

    For each subgroup S, the per-group contribution is the distance between
    the confidence interval for SP(S) and zero:

        excess(S) = max(0, ci_lower(S))      if SP̂(S) ≥ 0
                  = max(0, −ci_upper(S))     if SP̂(S) < 0

    This equals ``max(0, |SP̂(S)| − z_α · SE(S))`` for symmetric Wald CIs —
    i.e. how far the near CI bound is from zero, in SP units.

    MSPE = mean_S excess(S)

    Properties vs SAFT violation rate + mean SP|violations:
    - Single number (no need for two separate metrics).
    - Continuous: barely-significant groups contribute proportionally less
      than highly significant ones.
    - In SP units: directly interpretable as "average residual unfairness
      beyond the statistical noise floor."
    - Aligned with ASBR loss: ASBR minimises Σ_S max(0, |Z_S|−z_α)², which
      is the squared Z-score version; MSPE uses the linear SP-unit version.
    - Non-violations contribute exactly 0 (same selectivity as SAFT).

    Args:
        results: List of SAFTResult from ``saft_audit`` (after any multiplicity
                 correction, which does not affect CI bounds).

    Returns:
        mspe: Mean excess in SP units; 0.0 means no significant violations.
    """
    if not results:
        return 0.0
    excesses = []
    for r in results:
        if r.sp_estimate >= 0:
            excess = max(0.0, r.ci_lower)
        else:
            excess = max(0.0, -r.ci_upper)
        excesses.append(excess)
    return float(np.mean(excesses))


def saft_summary(results: list[SAFTResult]) -> SAFTSummary:
    """Aggregate SAFT results into a summary.

    Stratifies violations by subgroup size buckets:
    [2, 10), [10, 30), [30, 100), [100, ∞)

    Args:
        results: List of per-subgroup SAFTResult.

    Returns:
        SAFTSummary with aggregate statistics.
    """
    n_subgroups = len(results)
    violations = [r for r in results if r.reject_h0]
    n_violations = len(violations)
    violation_rate = n_violations / n_subgroups if n_subgroups > 0 else 0.0

    # Size buckets
    buckets = {
        "[2, 10)": (2, 10),
        "[10, 30)": (10, 30),
        "[30, 100)": (30, 100),
        "[100, inf)": (100, float("inf")),
    }
    violations_by_size: dict[str, dict[str, int | float]] = {}
    for label, (lo, hi) in buckets.items():
        in_bucket = [r for r in results if lo <= r.n_S < hi]
        violated = [r for r in in_bucket if r.reject_h0]
        n_total = len(in_bucket)
        n_viol = len(violated)
        violations_by_size[label] = {
            "n_subgroups": n_total,
            "n_violations": n_viol,
            "violation_rate": n_viol / n_total if n_total > 0 else 0.0,
        }

    mean_sp = (
        float(np.mean([abs(r.sp_estimate) for r in violations]))
        if violations
        else 0.0
    )

    return SAFTSummary(
        n_subgroups=n_subgroups,
        n_violations=n_violations,
        violation_rate=violation_rate,
        violations_by_size_bucket=violations_by_size,
        mean_sp_among_violations=mean_sp,
        mspe=compute_mspe(results),
    )


# ---------------------------------------------------------------------------
# Equal Opportunity SAFT
# ---------------------------------------------------------------------------

@dataclass
class EOSAFTResult:
    """Per-subgroup SAFT test result for Equal Opportunity (TPR disparity)."""

    subgroup: str
    n_S_pos: int          # number of y_true==1 samples in S (denominator for TPR_S)
    min_cell_count: int   # min of [TP_S, FN_S, TP_Sbar, FN_Sbar]
    eo_estimate: float    # TPR_S − TPR_Sbar (point estimate)
    ci_lower: float
    ci_upper: float
    p_value: float
    reject_h0: bool
    test_type: str        # "wald", "bayesian", or "exact_zero"


@dataclass
class EOSAFTSummary:
    """Aggregate EO-SAFT audit summary."""

    n_subgroups: int
    n_violations: int
    violation_rate: float
    violations_by_size_bucket: dict[str, dict[str, int | float]]
    mean_eo_among_violations: float
    msee: float           # Mean Significant EO Excess (see compute_msee)


def compute_eo_counts(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask_S: np.ndarray,
) -> np.ndarray | None:
    """Compute 4-cell EO contingency counts restricted to y_true==1 rows.

    Cells are [TP_S, FN_S, TP_Sbar, FN_Sbar] — true positives and false
    negatives for each side, among ground-truth positive examples only.

    Args:
        y_pred: Binary predictions, shape (n,).
        y_true: Ground-truth binary labels, shape (n,).
        mask_S: Boolean mask for subgroup S, shape (n,).

    Returns:
        counts: Integer array [TP_S, FN_S, TP_Sbar, FN_Sbar], or None if
                either side has no positive ground-truth labels (EO undefined).
    """
    mask_Sbar = ~mask_S
    pos_S    = mask_S    & (y_true == 1)
    pos_Sbar = mask_Sbar & (y_true == 1)

    if pos_S.sum() == 0 or pos_Sbar.sum() == 0:
        return None

    TP_S    = int(np.sum(y_pred[pos_S]    == 1))
    FN_S    = int(np.sum(y_pred[pos_S]    == 0))
    TP_Sbar = int(np.sum(y_pred[pos_Sbar] == 1))
    FN_Sbar = int(np.sum(y_pred[pos_Sbar] == 0))
    return np.array([TP_S, FN_S, TP_Sbar, FN_Sbar])


def saft_eo_test_subgroup(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask_S: np.ndarray,
    subgroup_name: str,
    alpha: float = 0.05,
    n_tilde: int = 30,
    mc_samples: int = 10000,
    rng: np.random.Generator | None = None,
    rejection_method: RejectionMethod = "p_value",
) -> EOSAFTResult | None:
    """Run EO-SAFT on a single subgroup.

    Identical to ``saft_test_subgroup`` but measures TPR disparity
    (Equal Opportunity) instead of acceptance-rate disparity (Statistical
    Parity). Only y_true==1 rows contribute to the contingency table.

    Args:
        y_pred: Binary predictions, shape (n,).
        y_true: Ground-truth binary labels, shape (n,).
        mask_S: Boolean subgroup mask, shape (n,).
        subgroup_name: Name of the subgroup.
        alpha: Significance level.
        n_tilde: Switching threshold (on min cell count).
        mc_samples: Number of MC samples for Bayesian test.
        rng: NumPy random generator for reproducibility.
        rejection_method: ``"p_value"`` (default) or ``"ci"``.

    Returns:
        EOSAFTResult, or None if EO is undefined for this subgroup (no
        positive ground-truth labels on one side).
    """
    counts = compute_eo_counts(y_pred, y_true, mask_S)
    if counts is None:
        return None

    n_pos_S    = int(counts[0] + counts[1])
    n_pos_Sbar = int(counts[2] + counts[3])
    n_pos      = n_pos_S + n_pos_Sbar
    min_cell   = int(counts.min())

    eps_eo = 1e-12
    tpr_S    = counts[0] / n_pos_S    if n_pos_S    > 0 else 0.0
    tpr_Sbar = counts[2] / n_pos_Sbar if n_pos_Sbar > 0 else 0.0
    observed_eo = tpr_S - tpr_Sbar

    if abs(observed_eo) < eps_eo:
        return EOSAFTResult(
            subgroup=subgroup_name,
            n_S_pos=n_pos_S,
            min_cell_count=min_cell,
            eo_estimate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
            reject_h0=False,
            test_type="exact_zero",
        )

    if min_cell >= n_tilde:
        p_hat = counts / n_pos
        eo, ci_lo, ci_hi, p_val = _wald_test(p_hat, n_pos, alpha)
        test_type = "wald"
    else:
        eo, ci_lo, ci_hi, p_val = _bayesian_mc_test(
            counts, alpha, mc_samples, rng=rng,
        )
        test_type = "bayesian"

    return EOSAFTResult(
        subgroup=subgroup_name,
        n_S_pos=n_pos_S,
        min_cell_count=min_cell,
        eo_estimate=eo,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p_value=p_val,
        reject_h0=_reject_h0(p_val, ci_lo, ci_hi, alpha, rejection_method),
        test_type=test_type,
    )


def saft_eo_audit(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    subgroup_masks: dict[str, np.ndarray],
    alpha: float = 0.05,
    n_tilde: int = 30,
    mc_samples: int = 10000,
    seed: int = 42,
    rejection_method: RejectionMethod = "ci",
) -> list[EOSAFTResult]:
    """Run EO-SAFT on all subgroups.

    Subgroups for which EO is undefined (no positive ground-truth labels on
    one side) are silently excluded from the returned list.

    Args:
        y_pred: Binary predictions, shape (n,).
        y_true: Ground-truth binary labels, shape (n,).
        subgroup_masks: Dict mapping subgroup name -> boolean mask.
        alpha: Significance level.
        n_tilde: Switching threshold.
        mc_samples: Number of MC samples for Bayesian test.
        seed: Random seed for Bayesian MC sampling.
        rejection_method: ``"p_value"`` or ``"ci"`` (default).

    Returns:
        List of EOSAFTResult for subgroups where EO is defined.
    """
    rng = np.random.default_rng(seed)
    results = []
    for name, mask in subgroup_masks.items():
        result = saft_eo_test_subgroup(
            y_pred, y_true, mask, name,
            alpha=alpha, n_tilde=n_tilde,
            mc_samples=mc_samples, rng=rng,
            rejection_method=rejection_method,
        )
        if result is not None:
            results.append(result)
    return results


def apply_multiplicity_correction_eo(
    results: list[EOSAFTResult],
    method: Literal["bonferroni", "bh"] = "bh",
    alpha: float = 0.05,
) -> list[EOSAFTResult]:
    """Re-evaluate rejection decisions with a multiple-testing correction.

    Analogous to ``apply_multiplicity_correction`` but for EOSAFTResult.

    Args:
        results: List of EOSAFTResult from ``saft_eo_audit``.
        method: ``"bonferroni"`` for FWER control, ``"bh"`` for
            Benjamini-Hochberg FDR control (default).
        alpha: Family-wise or FDR significance level.

    Returns:
        New list of EOSAFTResult with corrected ``reject_h0`` flags.
    """
    m = len(results)
    if m == 0:
        return []

    if method == "bonferroni":
        adj_alpha = alpha / m
        return [
            EOSAFTResult(
                subgroup=r.subgroup,
                n_S_pos=r.n_S_pos,
                min_cell_count=r.min_cell_count,
                eo_estimate=r.eo_estimate,
                ci_lower=r.ci_lower,
                ci_upper=r.ci_upper,
                p_value=r.p_value,
                reject_h0=(r.p_value < adj_alpha),
                test_type=r.test_type,
            )
            for r in results
        ]

    elif method == "bh":
        indexed = sorted(enumerate(results), key=lambda t: t[1].p_value)
        reject_flags = [False] * m

        max_k = 0
        for rank, (_, r) in enumerate(indexed, start=1):
            if r.p_value <= rank / m * alpha:
                max_k = rank

        for rank, (orig_idx, _) in enumerate(indexed, start=1):
            if rank <= max_k:
                reject_flags[orig_idx] = True

        return [
            EOSAFTResult(
                subgroup=r.subgroup,
                n_S_pos=r.n_S_pos,
                min_cell_count=r.min_cell_count,
                eo_estimate=r.eo_estimate,
                ci_lower=r.ci_lower,
                ci_upper=r.ci_upper,
                p_value=r.p_value,
                reject_h0=reject_flags[i],
                test_type=r.test_type,
            )
            for i, r in enumerate(results)
        ]

    else:
        raise ValueError(f"Unknown correction method: {method!r}")


def compute_msee(results: list[EOSAFTResult]) -> float:
    """Mean Significant EO Excess (MSEE) — EO analogue of MSPE.

    For each subgroup S where EO is defined, the per-group contribution is
    the distance between the CI for EO(S) = TPR_S - TPR_Sbar and zero:

        excess(S) = max(0, ci_lower(S))      if EO(S) ≥ 0
                  = max(0, −ci_upper(S))     if EO(S) < 0

    MSEE = mean_S excess(S)   (averaged over defined subgroups only)

    Non-violations contribute exactly 0. The averaging denominator is
    ``len(results)`` — the number of subgroups where EO is defined.

    Args:
        results: List of EOSAFTResult from ``saft_eo_audit`` (after any
                 multiplicity correction).

    Returns:
        msee: Mean excess in EO units; 0.0 means no significant violations.
    """
    if not results:
        return 0.0
    excesses = []
    for r in results:
        if r.eo_estimate >= 0:
            excess = max(0.0, r.ci_lower)
        else:
            excess = max(0.0, -r.ci_upper)
        excesses.append(excess)
    return float(np.mean(excesses))


def saft_eo_summary(results: list[EOSAFTResult]) -> EOSAFTSummary:
    """Aggregate EO-SAFT results into a summary.

    Uses n_S_pos (positive-label subgroup size) for size bucket assignment.

    Args:
        results: List of per-subgroup EOSAFTResult.

    Returns:
        EOSAFTSummary with aggregate statistics.
    """
    n_subgroups = len(results)
    violations = [r for r in results if r.reject_h0]
    n_violations = len(violations)
    violation_rate = n_violations / n_subgroups if n_subgroups > 0 else 0.0

    buckets = {
        "[2, 10)": (2, 10),
        "[10, 30)": (10, 30),
        "[30, 100)": (30, 100),
        "[100, inf)": (100, float("inf")),
    }
    violations_by_size: dict[str, dict[str, int | float]] = {}
    for label, (lo, hi) in buckets.items():
        in_bucket = [r for r in results if lo <= r.n_S_pos < hi]
        violated = [r for r in in_bucket if r.reject_h0]
        n_total = len(in_bucket)
        n_viol = len(violated)
        violations_by_size[label] = {
            "n_subgroups": n_total,
            "n_violations": n_viol,
            "violation_rate": n_viol / n_total if n_total > 0 else 0.0,
        }

    mean_eo = (
        float(np.mean([abs(r.eo_estimate) for r in violations]))
        if violations
        else 0.0
    )

    return EOSAFTSummary(
        n_subgroups=n_subgroups,
        n_violations=n_violations,
        violation_rate=violation_rate,
        violations_by_size_bucket=violations_by_size,
        mean_eo_among_violations=mean_eo,
        msee=compute_msee(results),
    )


def print_saft_report(
    results: list[SAFTResult],
    summary: SAFTSummary,
    top_n: int = 10,
) -> None:
    """Print a formatted SAFT audit report.

    Args:
        results: List of per-subgroup SAFTResult.
        summary: Aggregate SAFTSummary.
        top_n: Number of top violations to display.
    """
    print("\n" + "=" * 70)
    print("SAFT Audit Report")
    print("=" * 70)

    print(f"\n  Subgroups audited:  {summary.n_subgroups}")
    print(f"  Violations found:  {summary.n_violations}")
    print(f"  Violation rate:    {summary.violation_rate:.3f}")
    if summary.n_violations > 0:
        print(f"  Mean |SP| among violations: {summary.mean_sp_among_violations:.4f}")

    print("\n  Violations by size bucket:")
    print(f"  {'Bucket':<15} {'Subgroups':>10} {'Violations':>11} {'Rate':>8}")
    print("  " + "-" * 46)
    for label, info in summary.violations_by_size_bucket.items():
        print(f"  {label:<15} {info['n_subgroups']:>10} "
              f"{info['n_violations']:>11} {info['violation_rate']:>8.3f}")

    # Top violations sorted by p-value
    violations = sorted(
        [r for r in results if r.reject_h0],
        key=lambda r: r.p_value,
    )
    if violations:
        show = violations[:top_n]
        print(f"\n  Top {len(show)} violations (by p-value):")
        print(f"  {'Subgroup':<30} {'n_S':>5} {'SP':>8} "
              f"{'CI':>18} {'p-value':>10} {'Test':>8}")
        print("  " + "-" * 81)
        for r in show:
            ci_str = f"[{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]"
            print(f"  {r.subgroup:<30} {r.n_S:>5} {r.sp_estimate:>+8.4f} "
                  f"{ci_str:>18} {r.p_value:>10.2e} {r.test_type:>8}")
    print()
