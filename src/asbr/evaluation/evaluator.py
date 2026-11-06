"""Evaluation pipeline: compute all metrics on a trained model.

Produces a flat dict of metrics suitable for DataFrame aggregation
across cross-validation folds and seeds.

Both Statistical Parity (SP) and Equal Opportunity (EO) metrics are
always computed at evaluation time, regardless of which metric was used
during training.  This lets you compare methods trained with different
objectives on a common footing.

Output keys
-----------
Accuracy / AUC / Log Loss:
    accuracy_train, accuracy_test, auc_train, auc_test, log_loss_train, log_loss_test

Statistical Parity:
    max_sp_train, max_sp_test, mean_sp_train, mean_sp_test

Equal Opportunity:
    max_eo_train, max_eo_test, mean_eo_train, mean_eo_test

Generalization gap (SP-based):
    gap_overall, gap_tail, gap_bulk

Other:
    n_subgroups
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from asbr.fairness.metrics import (
    statistical_parity,
    max_sp_violation,
    mean_sp_violation,
    max_eo_violation,
    mean_eo_violation,
    fairness_generalization_gap,
    stratified_gap_summary,
)


def compute_predictions(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device | str | None = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Get model predictions as numpy arrays.

    Args:
        model: Trained model.
        X: Features, shape (n, d).
        device: Torch device. Defaults to the device the model is already on.
        threshold: Binary classification threshold.

    Returns:
        y_prob: Predicted probabilities, shape (n,).
        y_pred: Binary predictions, shape (n,).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_prob = model(X_t).cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred


def compute_all_metrics_from_predictions(
    y_prob_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_train: np.ndarray,
    subgroup_masks_train: dict[str, np.ndarray],
    y_prob_test: np.ndarray,
    y_pred_test: np.ndarray,
    y_test: np.ndarray,
    subgroup_masks_test: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute all evaluation metrics from pre-computed predictions.

    Model-agnostic: works with any method that produces numpy predictions.

    Returns a flat dict with keys:
    - accuracy_train, accuracy_test
    - auc_train, auc_test
    - log_loss_train, log_loss_test
    - max_sp_train, max_sp_test, mean_sp_train, mean_sp_test
    - max_eo_train, max_eo_test, mean_eo_train, mean_eo_test
    - gap_overall, gap_tail, gap_bulk
    - n_subgroups

    SP and EO are always computed regardless of which metric was used during
    training, allowing cross-method comparison on a common footing.

    Args:
        y_prob_train: Predicted probabilities on train set, shape (n_train,).
        y_pred_train: Binary predictions on train set, shape (n_train,).
        y_train: True labels for train set, shape (n_train,).
        subgroup_masks_train: Subgroup boolean masks for training set.
        y_prob_test: Predicted probabilities on test set, shape (n_test,).
        y_pred_test: Binary predictions on test set, shape (n_test,).
        y_test: True labels for test set, shape (n_test,).
        subgroup_masks_test: Subgroup boolean masks for test set.

    Returns:
        metrics: Flat dict of metric name -> value.
    """
    # Accuracy
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    # AUC-ROC (handle degenerate case)
    try:
        auc_train = roc_auc_score(y_train, y_prob_train)
    except ValueError:
        auc_train = float("nan")
    try:
        auc_test = roc_auc_score(y_test, y_prob_test)
    except ValueError:
        auc_test = float("nan")

    try:
        logloss_train = log_loss(y_train, y_prob_train, labels=[0, 1])
    except ValueError:
        logloss_train = float("nan")
    try:
        logloss_test = log_loss(y_test, y_prob_test, labels=[0, 1])
    except ValueError:
        logloss_test = float("nan")

    # Statistical Parity
    max_sp_train = max_sp_violation(y_pred_train, subgroup_masks_train)
    max_sp_test = max_sp_violation(y_pred_test, subgroup_masks_test)
    mean_sp_train = mean_sp_violation(y_pred_train, subgroup_masks_train)
    mean_sp_test = mean_sp_violation(y_pred_test, subgroup_masks_test)

    # Equal Opportunity
    max_eo_train = max_eo_violation(y_pred_train, y_train, subgroup_masks_train)
    max_eo_test = max_eo_violation(y_pred_test, y_test, subgroup_masks_test)
    mean_eo_train = mean_eo_violation(y_pred_train, y_train, subgroup_masks_train)
    mean_eo_test = mean_eo_violation(y_pred_test, y_test, subgroup_masks_test)

    # Generalization gap (SP-based)
    gaps = fairness_generalization_gap(
        y_pred_train, subgroup_masks_train,
        y_pred_test, subgroup_masks_test,
    )
    subgroup_sizes = {
        name: int(mask.sum()) for name, mask in subgroup_masks_train.items()
    }
    gap_summary = stratified_gap_summary(gaps, subgroup_sizes)

    return {
        "accuracy_train": acc_train,
        "accuracy_test": acc_test,
        "auc_train": auc_train,
        "auc_test": auc_test,
        "log_loss_train": logloss_train,
        "log_loss_test": logloss_test,
        "max_sp_train": max_sp_train,
        "max_sp_test": max_sp_test,
        "mean_sp_train": mean_sp_train,
        "mean_sp_test": mean_sp_test,
        "max_eo_train": max_eo_train,
        "max_eo_test": max_eo_test,
        "mean_eo_train": mean_eo_train,
        "mean_eo_test": mean_eo_test,
        "gap_overall": gap_summary["gap_overall"],
        "gap_tail": gap_summary["gap_tail"],
        "gap_bulk": gap_summary["gap_bulk"],
        "n_subgroups": len(subgroup_masks_train),
    }


def compute_all_metrics(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    subgroup_masks_train: dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    subgroup_masks_test: dict[str, np.ndarray],
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Compute all evaluation metrics for a trained model.

    Returns a flat dict with keys:
    - accuracy_train, accuracy_test
    - auc_train, auc_test
    - log_loss_train, log_loss_test
    - max_sp_train, max_sp_test, mean_sp_train, mean_sp_test
    - max_eo_train, max_eo_test, mean_eo_train, mean_eo_test
    - gap_overall, gap_tail, gap_bulk
    - n_subgroups

    Args:
        model: Trained model.
        X_train, y_train: Training data.
        subgroup_masks_train: Subgroup boolean masks for training set.
        X_test, y_test: Test data.
        subgroup_masks_test: Subgroup boolean masks for test set.
        device: Torch device.

    Returns:
        metrics: Flat dict of metric name -> value.
    """
    y_prob_train, y_pred_train = compute_predictions(model, X_train, device)
    y_prob_test, y_pred_test = compute_predictions(model, X_test, device)

    return compute_all_metrics_from_predictions(
        y_prob_train, y_pred_train, y_train, subgroup_masks_train,
        y_prob_test, y_pred_test, y_test, subgroup_masks_test,
    )
