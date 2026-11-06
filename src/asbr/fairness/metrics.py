"""Fairness evaluation metrics computed on hard (thresholded) predictions.

These metrics are used for evaluation only (not during training).
They operate on binary predictions and numpy arrays.

Two fairness notions are supported:

Statistical Parity (SP)
-----------------------
``SP(S) = P(ŷ=1 | S) − P(ŷ=1 | S̄)``

Measures the difference in positive-prediction rates between the subgroup
and its complement.  Use ``statistical_parity``, ``max_sp_violation``,
``mean_sp_violation``.

Equal Opportunity (EO)
-----------------------
``EO(S) = P(ŷ=1 | y=1, S) − P(ŷ=1 | y=1, S̄)``

Measures the difference in true-positive rates (TPR), conditioning on
ground-truth positive labels.  Subgroups with no positive labels on
either side are excluded from aggregates (not counted as zero).
Use ``equal_opportunity``, ``max_eo_violation``, ``mean_eo_violation``.

Example::

    from asbr.fairness.metrics import max_eo_violation, mean_eo_violation

    max_eo  = max_eo_violation(y_pred_binary, y_true, subgroup_masks)
    mean_eo = mean_eo_violation(y_pred_binary, y_true, subgroup_masks)
"""

from __future__ import annotations

import numpy as np


def statistical_parity(
    y_pred_binary: np.ndarray,
    mask_S: np.ndarray,
) -> float:
    """Compute statistical parity for a single subgroup.

    SP(S) = P(y_hat=1 | S) - P(y_hat=1 | not S)

    Args:
        y_pred_binary: Binary predictions, shape (n,).
        mask_S: Boolean mask for subgroup S, shape (n,).

    Returns:
        sp: Statistical parity difference (can be negative).
    """
    mask_Sbar = ~mask_S
    if mask_S.sum() == 0 or mask_Sbar.sum() == 0:
        return 0.0
    rate_S = y_pred_binary[mask_S].mean()
    rate_Sbar = y_pred_binary[mask_Sbar].mean()
    return float(rate_S - rate_Sbar)


def max_sp_violation(
    y_pred_binary: np.ndarray,
    subgroup_masks: dict[str, np.ndarray],
) -> float:
    """Worst-case absolute SP violation across all subgroups."""
    violations = [
        abs(statistical_parity(y_pred_binary, mask))
        for mask in subgroup_masks.values()
    ]
    return max(violations) if violations else 0.0


def mean_sp_violation(
    y_pred_binary: np.ndarray,
    subgroup_masks: dict[str, np.ndarray],
) -> float:
    """Mean absolute SP violation across all subgroups."""
    violations = [
        abs(statistical_parity(y_pred_binary, mask))
        for mask in subgroup_masks.values()
    ]
    return float(np.mean(violations)) if violations else 0.0


def fairness_generalization_gap(
    y_pred_train_binary: np.ndarray,
    subgroup_masks_train: dict[str, np.ndarray],
    y_pred_test_binary: np.ndarray,
    subgroup_masks_test: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute per-subgroup fairness generalization gap.

    Delta_SP(S) = |SP_train(S) - SP_test(S)|

    Only computes for subgroups present in both train and test sets
    (with at least 1 member in each partition of S and Sbar).

    Args:
        y_pred_train_binary: Binary predictions on train set.
        subgroup_masks_train: Subgroup masks for train set.
        y_pred_test_binary: Binary predictions on test set.
        subgroup_masks_test: Subgroup masks for test set.

    Returns:
        gaps: Dict mapping subgroup name -> generalization gap.
    """
    gaps: dict[str, float] = {}
    shared_groups = set(subgroup_masks_train) & set(subgroup_masks_test)

    for name in shared_groups:
        mask_train = subgroup_masks_train[name]
        mask_test = subgroup_masks_test[name]

        # Need at least 1 in each partition
        if mask_train.sum() < 1 or (~mask_train).sum() < 1:
            continue
        if mask_test.sum() < 1 or (~mask_test).sum() < 1:
            continue

        sp_train = statistical_parity(y_pred_train_binary, mask_train)
        sp_test = statistical_parity(y_pred_test_binary, mask_test)
        gaps[name] = abs(sp_train - sp_test)

    return gaps


def stratified_gap_summary(
    gaps: dict[str, float],
    subgroup_sizes: dict[str, int],
) -> dict[str, float]:
    """Summarize generalization gap stratified by subgroup size.

    Reports overall, bottom-25% (tail), and top-25% (bulk) means.

    Args:
        gaps: Per-subgroup generalization gaps.
        subgroup_sizes: Per-subgroup sample sizes (from training set).

    Returns:
        summary: Dict with keys "gap_overall", "gap_tail", "gap_bulk".
    """
    if not gaps:
        return {"gap_overall": float("nan"), "gap_tail": float("nan"), "gap_bulk": float("nan")}

    # Sort by size
    shared = sorted(
        [name for name in gaps if name in subgroup_sizes],
        key=lambda g: subgroup_sizes[g],
    )
    n_groups = len(shared)
    q25_cutoff = max(1, n_groups // 4)

    tail_groups = shared[:q25_cutoff]
    bulk_groups = shared[-q25_cutoff:]

    gap_values = [gaps[g] for g in shared]
    tail_values = [gaps[g] for g in tail_groups]
    bulk_values = [gaps[g] for g in bulk_groups]

    return {
        "gap_overall": float(np.mean(gap_values)),
        "gap_tail": float(np.mean(tail_values)) if tail_values else float("nan"),
        "gap_bulk": float(np.mean(bulk_values)) if bulk_values else float("nan"),
    }


def equal_opportunity(
    y_pred_binary: np.ndarray,
    y_true: np.ndarray,
    mask_S: np.ndarray,
) -> float | None:
    """Compute Equal Opportunity for a single subgroup.

    EO(S) = P(y_hat=1 | y=1, S) - P(y_hat=1 | y=1, not S)

    Conditions on ground-truth positive labels (y=1) before computing rates,
    so this measures TPR disparity rather than positive-prediction-rate disparity.

    Args:
        y_pred_binary: Binary predictions, shape (n,).
        y_true: Ground-truth binary labels, shape (n,).
        mask_S: Boolean mask for subgroup S, shape (n,).

    Returns:
        eo: Equal opportunity difference (can be negative), or None if either
            side has no positive ground-truth labels (metric undefined).
    """
    mask_Sbar = ~mask_S
    pos_S = mask_S & (y_true == 1)
    pos_Sbar = mask_Sbar & (y_true == 1)

    if pos_S.sum() == 0 or pos_Sbar.sum() == 0:
        return None

    tpr_S = y_pred_binary[pos_S].mean()
    tpr_Sbar = y_pred_binary[pos_Sbar].mean()
    return float(tpr_S - tpr_Sbar)


def max_eo_violation(
    y_pred_binary: np.ndarray,
    y_true: np.ndarray,
    subgroup_masks: dict[str, np.ndarray],
) -> float:
    """Worst-case absolute EO violation across all subgroups.

    Subgroups for which Equal Opportunity is undefined (no positive labels
    on one side) are excluded from the maximum.

    Args:
        y_pred_binary: Binary predictions, shape (n,).
        y_true: Ground-truth binary labels, shape (n,).
        subgroup_masks: Dict mapping subgroup name -> boolean mask.

    Returns:
        Worst-case |EO| across eligible subgroups; 0.0 if none qualify.
    """
    violations = [
        abs(eo)
        for mask in subgroup_masks.values()
        if (eo := equal_opportunity(y_pred_binary, y_true, mask)) is not None
    ]
    return max(violations) if violations else 0.0


def mean_eo_violation(
    y_pred_binary: np.ndarray,
    y_true: np.ndarray,
    subgroup_masks: dict[str, np.ndarray],
) -> float:
    """Mean absolute EO violation across all subgroups.

    Subgroups for which Equal Opportunity is undefined (no positive labels
    on one side) are excluded from the mean.

    Args:
        y_pred_binary: Binary predictions, shape (n,).
        y_true: Ground-truth binary labels, shape (n,).
        subgroup_masks: Dict mapping subgroup name -> boolean mask.

    Returns:
        Mean |EO| across eligible subgroups; 0.0 if none qualify.
    """
    violations = [
        abs(eo)
        for mask in subgroup_masks.values()
        if (eo := equal_opportunity(y_pred_binary, y_true, mask)) is not None
    ]
    return float(np.mean(violations)) if violations else 0.0
