"""Stratified K-fold cross-validation with seed control.

Runs ASBR training across multiple folds and seeds, collecting
all evaluation metrics for statistical analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from asbr.data.intersections import build_subgroup_masks, masks_to_torch
from asbr.data.loader import FairnessDataset, prepare_split
from asbr.evaluation.evaluator import compute_all_metrics
from asbr.experiments.config import ASBRConfig, TrainingConfig
from asbr.models.base_mlp import FairnessMLP
from asbr.training.trainer import Trainer, BaselineTrainer


def run_cv(
    dataset: FairnessDataset,
    asbr_config: ASBRConfig | None = None,
    training_config: TrainingConfig | None = None,
    n_folds: int = 5,
    n_seeds: int = 10,
    intersection_level: int = 3,
    device: torch.device | str = "cpu",
    use_fairness: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run stratified K-fold CV across multiple seeds.

    Args:
        dataset: FairnessDataset to train on.
        asbr_config: ASBR loss configuration.
        training_config: Training loop configuration.
        n_folds: Number of CV folds.
        n_seeds: Number of random seeds.
        intersection_level: Max intersection level for subgroups.
        device: Torch device.
        use_fairness: If True, use ASBR trainer; otherwise use ERM baseline.
        verbose: Print progress.

    Returns:
        results: DataFrame with one row per (seed, fold) containing all metrics.
    """
    asbr_config = asbr_config or ASBRConfig()
    training_config = training_config or TrainingConfig()

    all_results: list[dict] = []

    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold, (train_idx, test_idx) in enumerate(skf.split(dataset.X, dataset.y)):
            if verbose:
                print(f"  Seed {seed}/{n_seeds-1}, Fold {fold}/{n_folds-1}")

            # Split data
            X_train_raw, X_test_raw = dataset.X[train_idx], dataset.X[test_idx]
            y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]
            A_train = dataset.A.iloc[train_idx].reset_index(drop=True)
            A_test = dataset.A.iloc[test_idx].reset_index(drop=True)

            # Standardize features
            X_train, X_test, _ = prepare_split(X_train_raw, X_test_raw)

            # Build subgroup masks
            masks_train_np = build_subgroup_masks(A_train, level=intersection_level)
            masks_test_np = build_subgroup_masks(A_test, level=intersection_level)

            # Train/val split from training fold (80/20)
            n_train = len(y_train)
            rng = np.random.RandomState(seed * 1000 + fold)
            perm = rng.permutation(n_train)
            n_val = max(1, n_train // 5)
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]

            X_tr, X_vl = X_train[tr_idx], X_train[val_idx]
            y_tr, y_vl = y_train[tr_idx], y_train[val_idx]

            # Subgroup masks for the actual training subset (for fairness loss)
            A_tr = A_train.iloc[tr_idx].reset_index(drop=True)
            masks_tr_np = build_subgroup_masks(A_tr, level=intersection_level)
            masks_tr_torch = masks_to_torch(masks_tr_np, device=device)

            # Set seed for reproducibility
            torch.manual_seed(seed * 1000 + fold)
            np.random.seed(seed * 1000 + fold)

            # Create model
            input_dim = X_train.shape[1]
            model = FairnessMLP(
                input_dim=input_dim,
                hidden_dims=training_config.hidden_dims,
                dropout=training_config.dropout,
            )

            # Train
            if use_fairness:
                trainer = Trainer(model, asbr_config, training_config, device)
                history = trainer.train(X_tr, y_tr, masks_tr_torch, X_vl, y_vl)
            else:
                trainer = BaselineTrainer(model, training_config, device)
                history = trainer.train(X_tr, y_tr, X_vl, y_vl)

            # Evaluate on full train/test split (not tr/vl subsplit)
            metrics = compute_all_metrics(
                model=model,
                X_train=X_train,
                y_train=y_train,
                subgroup_masks_train=masks_train_np,
                X_test=X_test,
                y_test=y_test,
                subgroup_masks_test=masks_test_np,
                device=device,
            )
            metrics["seed"] = seed
            metrics["fold"] = fold
            metrics["n_epochs"] = len(history)

            all_results.append(metrics)

    return pd.DataFrame(all_results)
