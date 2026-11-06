"""Hydra-based experiment runner for ASBR fairness comparison.

This is the primary entry point for running experiments. Each method has its
own config schema (see ``conf/method/``), and Hydra handles parameter sweeps
via ``--multirun``.

Usage::

    # Single method
    python -m asbr.experiments.run_hydra method=asbr_lrt_joint dataset=adult

    # Quick debug
    python -m asbr.experiments.run_hydra method=erm training=fast n_splits=2 n_seeds=1

    # Lambda sweep
    python -m asbr.experiments.run_hydra --multirun \\
        method=asbr_wald,asbr_lrt,asbr_wald_joint,asbr_lrt_joint \\
        method.asbr.lambd=0.1,0.5,1.0,5.0,10.0

    # Full comparison
    python -m asbr.experiments.run_hydra --multirun \\
        method=erm,expgrad,sp_penalty,asbr_wald,asbr_lrt,asbr_wald_joint,asbr_lrt_joint
"""

from __future__ import annotations

import os
import time

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from asbr.data.loader import prepare_split
from asbr.data.intersections import build_subgroup_masks
from asbr.evaluation.evaluator import compute_all_metrics_from_predictions
from asbr.experiments.config import TrainingConfig
from asbr.fairness.saft import (
    apply_multiplicity_correction,
    apply_multiplicity_correction_eo,
    saft_audit,
    saft_eo_audit,
    saft_summary,
    saft_eo_summary,
)


# ---------------------------------------------------------------------------
# Bridge functions: DictConfig -> dataclasses
# ---------------------------------------------------------------------------

def _build_training_config(cfg: DictConfig) -> TrainingConfig:
    """Convert the Hydra training config to the existing TrainingConfig dataclass."""
    return TrainingConfig(
        hidden_dims=tuple(cfg.training.hidden_dims),
        dropout=cfg.training.dropout,
        lr=cfg.training.lr,
        patience=cfg.training.patience,
        max_epochs=cfg.training.max_epochs,
        batch_size=cfg.training.batch_size,
        max_grad_norm=cfg.training.get("max_grad_norm", 1.0),
        weight_decay=cfg.training.get("weight_decay", 5e-4),
        bce_warmup_epochs=cfg.training.get("bce_warmup_epochs", 0),
        joint_checkpoint_mode=cfg.training.get("joint_checkpoint_mode", "val_bce"),
        joint_bce_guard_rel_tol=cfg.training.get("joint_bce_guard_rel_tol", 0.05),
        joint_bce_guard_min_lambd=cfg.training.get("joint_bce_guard_min_lambd", 0.0),
        fairness_ramp_epochs=cfg.training.get("fairness_ramp_epochs", 0),
        fairness_ramp_shape=cfg.training.get("fairness_ramp_shape", "linear"),
        fairness_ramp_min_lambd=cfg.training.get("fairness_ramp_min_lambd", 0.0),
        lr_reduce_factor=cfg.training.get("lr_reduce_factor", 0.1),
        min_lr=cfg.training.get("min_lr", 1e-5),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run a single method on a single dataset with repeated stratified splits."""
    method_name = cfg.method.name

    # Ensure deterministic behaviour for reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(OmegaConf.to_yaml(cfg))

    dataset = hydra.utils.instantiate(cfg.dataset)
    dataset_name = dataset.name
    intersection_level = dataset.intersection_level

    method = hydra.utils.instantiate(cfg.method, _recursive_=False)
    training_config = _build_training_config(cfg)

    # Two-phase training requires no gradient clipping (SGD in Phase 2)
    trainer = cfg.method.get("trainer", "joint")
    if trainer == "two_phase" and training_config.max_grad_norm != 0:
        print(f"  [auto] Overriding max_grad_norm={training_config.max_grad_norm} -> 0 for two_phase trainer")
        training_config = TrainingConfig(
            hidden_dims=training_config.hidden_dims,
            dropout=training_config.dropout,
            lr=training_config.lr,
            patience=training_config.patience,
            max_epochs=training_config.max_epochs,
            batch_size=training_config.batch_size,
            max_grad_norm=0,
            weight_decay=training_config.weight_decay,
            bce_warmup_epochs=training_config.bce_warmup_epochs,
            joint_checkpoint_mode=training_config.joint_checkpoint_mode,
            joint_bce_guard_rel_tol=training_config.joint_bce_guard_rel_tol,
            joint_bce_guard_min_lambd=training_config.joint_bce_guard_min_lambd,
            fairness_ramp_epochs=training_config.fairness_ramp_epochs,
            fairness_ramp_shape=training_config.fairness_ramp_shape,
            fairness_ramp_min_lambd=training_config.fairness_ramp_min_lambd,
            lr_reduce_factor=training_config.lr_reduce_factor,
            min_lr=training_config.min_lr,
        )

    print(f"Method: {method_name}, Dataset: {dataset_name}")
    print(f"Loaded {dataset_name}: n={len(dataset.y)}, d={dataset.X.shape[1]}")

    test_size = cfg.get("test_size", 0.4)
    n_splits = cfg.get("n_splits", 5)
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=42,
    )
    print(f"Split strategy: {n_splits} splits, test_size={test_size}")

    out_path = os.path.join(os.getcwd(), "results.csv")
    force_rerun = cfg.get("force_rerun", False)
    save_predictions = cfg.get("save_predictions", False)
    pred_path = os.path.join(os.getcwd(), "predictions.csv") if save_predictions else None
    if force_rerun:
        _remove_existing_results_csv(out_path)
        if pred_path is not None:
            _remove_existing_results_csv(pred_path)
        all_results: list[dict] = []
        completed_keys: set[tuple[int, int, float | None]] = set()
        print(f"Force rerun enabled: ignoring previous results in {out_path}")
    else:
        all_results = _load_existing_results_csv(out_path)
        completed_keys = {_result_key(row) for row in all_results}
        if all_results:
            print(f"Resuming from {out_path}: loaded {len(all_results)} completed rows")
    all_predictions: list[pd.DataFrame] = []

    existing_schedule = getattr(method, "lambda_schedule", None)
    configured_lambd = _normalize_lambd(_configured_lambd(cfg))

    def _seed_keys(split_idx: int, seed: int) -> set[tuple[int, int, float | None]]:
        if existing_schedule:
            return {
                (split_idx, seed, _normalize_lambd(lambd))
                for lambd in existing_schedule
            }
        return {(split_idx, seed, configured_lambd)}

    for split_idx, (train_idx, test_idx) in enumerate(
        sss.split(dataset.X, dataset.y)
    ):
        split_keys = set().union(
            *(_seed_keys(split_idx, seed_idx + cfg.seed_offset)
              for seed_idx in range(cfg.n_seeds))
        )
        if split_keys.issubset(completed_keys):
            print(f"  Skipping split {split_idx}: all seeds already in results.csv")
            continue

        X_fold_train = dataset.X[train_idx]
        y_fold_train = dataset.y[train_idx]
        A_fold_train = dataset.A.iloc[train_idx].reset_index(drop=True)
        X_fold_test = dataset.X[test_idx]
        y_fold_test = dataset.y[test_idx]
        A_fold_test = dataset.A.iloc[test_idx].reset_index(drop=True)

        masks_train_np = build_subgroup_masks(
            A_fold_train, level=intersection_level, min_size=2
        )
        masks_test_np = build_subgroup_masks(
            A_fold_test, level=intersection_level, min_size=2
        )

        for seed_idx in range(cfg.n_seeds):
            seed = seed_idx + cfg.seed_offset
            seed_keys = _seed_keys(split_idx, seed)

            if seed_keys.issubset(completed_keys):
                print(f"  Skipping split {split_idx}, seed {seed}: already in results.csv")
                continue

            idx_tr, idx_vl = train_test_split(
                np.arange(len(X_fold_train)),
                test_size=cfg.get("val_size", 0.2),
                stratify=y_fold_train,
                random_state=seed,
            )
            X_tr, X_vl = X_fold_train[idx_tr], X_fold_train[idx_vl]
            y_tr, y_vl = y_fold_train[idx_tr], y_fold_train[idx_vl]
            A_tr = A_fold_train.iloc[idx_tr].reset_index(drop=True)

            X_tr_s, X_vl_s, scaler = prepare_split(X_tr, X_vl)
            X_train_scaled = np.nan_to_num(scaler.transform(X_fold_train).astype(np.float32))
            X_test_scaled = np.nan_to_num(scaler.transform(X_fold_test).astype(np.float32))

            t0 = time.time()
            np.random.seed(seed)
            torch.manual_seed(seed)

            method.train(
                X_tr_s, y_tr, X_vl_s, y_vl, A_tr,
                intersection_level, training_config,
            )

            saft_correction = cfg.get("saft_correction", "none")

            # Progressive methods produce one snapshot per λ; standard methods produce one.
            if hasattr(method, "lambda_schedule") and method.lambda_schedule:
                predict_items = [
                    (lambd, lambda X, l=lambd: method.predict_at(l, X))
                    for lambd in method.lambda_schedule
                ]
            else:
                predict_items = [(_configured_lambd(cfg), method.predict)]

            for lambd, predict_fn in predict_items:
                row_key = (split_idx, seed, _normalize_lambd(lambd))
                if row_key in completed_keys:
                    lambd_str = _format_lambd_label(cfg, lambd)
                    print(f"  Skipping split {split_idx}, seed {seed}, {lambd_str}: already in results.csv")
                    continue

                y_prob_train, y_pred_train = predict_fn(X_train_scaled)
                y_prob_test, y_pred_test = predict_fn(X_test_scaled)

                if save_predictions:
                    pred_df = A_fold_test.copy()
                    pred_df["sample_idx"] = test_idx
                    pred_df["y_true"] = y_fold_test
                    pred_df["y_prob"] = y_prob_test
                    pred_df["y_pred"] = y_pred_test
                    pred_df["split"] = split_idx
                    pred_df["seed"] = seed
                    pred_df["lambd"] = lambd
                    all_predictions.append(pred_df)

                metrics = compute_all_metrics_from_predictions(
                    y_prob_train, y_pred_train, y_fold_train, masks_train_np,
                    y_prob_test, y_pred_test, y_fold_test, masks_test_np,
                )

                ckpt_meta: dict = {}
                if hasattr(method, "get_checkpoint_meta") and lambd is not None:
                    ckpt_meta = method.get_checkpoint_meta(lambd)

                saft_results = saft_audit(
                    y_pred_test, y_fold_test, masks_test_np,
                    alpha=cfg.saft_alpha, n_tilde=cfg.saft_n_tilde,
                    seed=seed,
                )
                if saft_correction != "none":
                    saft_results = apply_multiplicity_correction(
                        saft_results, method=saft_correction, alpha=cfg.saft_alpha,
                    )
                saft_summ = saft_summary(saft_results)

                saft_eo_results = saft_eo_audit(
                    y_pred_test, y_fold_test, masks_test_np,
                    alpha=cfg.saft_alpha, n_tilde=cfg.saft_n_tilde,
                    seed=seed,
                )
                if saft_correction != "none":
                    saft_eo_results = apply_multiplicity_correction_eo(
                        saft_eo_results, method=saft_correction, alpha=cfg.saft_alpha,
                    )
                saft_eo_summ = saft_eo_summary(saft_eo_results)

                train_saft_eo_results = saft_eo_audit(
                    y_pred_train, y_fold_train, masks_train_np,
                    alpha=cfg.saft_alpha, n_tilde=cfg.saft_n_tilde,
                    seed=seed,
                )
                if saft_correction != "none":
                    train_saft_eo_results = apply_multiplicity_correction_eo(
                        train_saft_eo_results, method=saft_correction, alpha=cfg.saft_alpha,
                    )
                train_saft_eo_summ = saft_eo_summary(train_saft_eo_results)

                row = {
                    "method": method_name,
                    "dataset": dataset_name,
                    "split": split_idx,
                    "seed": seed,
                    "lambd": lambd,
                    "n_train": len(X_fold_train),
                    "n_test": len(X_fold_test),
                    **metrics,
                    "saft_n_violations": saft_summ.n_violations,
                    "saft_violation_rate": saft_summ.violation_rate,
                    "saft_mean_sp_violations": saft_summ.mean_sp_among_violations,
                    "saft_mspe": saft_summ.mspe,
                    "saft_eo_n_violations": saft_eo_summ.n_violations,
                    "saft_eo_violation_rate": saft_eo_summ.violation_rate,
                    "saft_eo_mean_eo_violations": saft_eo_summ.mean_eo_among_violations,
                    "saft_msee": saft_eo_summ.msee,
                    "train_saft_eo_n_violations": train_saft_eo_summ.n_violations,
                    "train_saft_eo_violation_rate": train_saft_eo_summ.violation_rate,
                    "train_saft_eo_mean_eo_violations": train_saft_eo_summ.mean_eo_among_violations,
                    "train_saft_msee": train_saft_eo_summ.msee,
                    "train_msee_step": ckpt_meta.get("train_msee"),
                    "stop_reason": ckpt_meta.get("stop_reason"),
                    "selected": ckpt_meta.get("selected"),
                    "elapsed_s": time.time() - t0,
                }
                all_results.append(row)
                completed_keys.add(row_key)
                _write_results_csv(all_results, out_path)

                lambd_str = _format_lambd_label(cfg, lambd)
                print(
                    f"  Split {split_idx}, seed {seed}, {method_name} {lambd_str}: "
                    f"acc={metrics['accuracy_test']:.4f}, "
                    f"auc={metrics['auc_test']:.4f}, "
                    f"logloss={metrics['log_loss_test']:.4f}, "
                    f"max_sp={metrics['max_sp_test']:.4f}, "
                    f"max_eo={metrics['max_eo_test']:.4f}, "
                    f"saft_viol={saft_summ.violation_rate:.3f} "
                    f"({time.time() - t0:.1f}s)"
                )

    df = pd.DataFrame(all_results)

    # Save final copy to Hydra output directory. The same file is also updated
    # incrementally after each completed split/seed row.
    _write_results_csv(all_results, out_path)
    print(f"\nResults saved to {out_path}")

    if pred_path is not None and all_predictions:
        prediction_frames = []
        if os.path.exists(pred_path):
            prediction_frames.append(pd.read_csv(pred_path))
        prediction_frames.extend(all_predictions)
        pd.concat(prediction_frames, ignore_index=True).to_csv(pred_path, index=False)
        print(f"Predictions saved to {pred_path}")
    elif pred_path is not None and os.path.exists(pred_path):
        print(f"No new predictions; kept existing {pred_path}")

    # Print summary
    _print_summary(df)


def _load_existing_results_csv(out_path: str) -> list[dict]:
    """Load completed rows from a previous run in this Hydra output directory."""
    if not os.path.exists(out_path):
        return []

    df = pd.read_csv(out_path)
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _remove_existing_results_csv(out_path: str) -> None:
    """Remove previous results files before a forced rerun."""
    for path in (out_path, f"{out_path}.tmp"):
        if os.path.exists(path):
            os.remove(path)


def _result_key(row: dict) -> tuple[int, int, float | None]:
    """Return the resume key for one result row."""
    return (
        int(row["split"]),
        int(row["seed"]),
        _normalize_lambd(row.get("lambd")),
    )


def _normalize_lambd(lambd: object) -> float | None:
    """Normalize empty/NaN lambda values from standard-method result rows."""
    if lambd is None or pd.isna(lambd):
        return None
    return float(lambd)


def _format_lambd_label(cfg: DictConfig, lambd: object) -> str:
    """Return the lambda label used in per-split progress messages."""
    display_lambd = _configured_lambd(cfg) if lambd is None else lambd
    if display_lambd is None or pd.isna(display_lambd):
        return "λ=n/a"
    return f"λ={display_lambd}"


def _configured_lambd(cfg: DictConfig) -> float | None:
    """Read the configured lambda for standard, non-progressive methods."""
    for key in ("method.asbr.lambd", "method.lambd"):
        value = OmegaConf.select(cfg, key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _write_results_csv(rows: list[dict], out_path: str) -> None:
    """Atomically write the completed result rows to ``out_path``."""
    df = pd.DataFrame(rows)
    tmp_path = f"{out_path}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def _print_summary(df: pd.DataFrame) -> None:
    """Print aggregated summary for the run."""
    key_metrics = [
        "accuracy_test", "auc_test", "log_loss_test",
        "max_sp_test", "mean_sp_test",
        "max_eo_test", "mean_eo_test",
        "saft_violation_rate", "saft_mean_sp_violations", "saft_mspe",
        "saft_eo_violation_rate", "saft_eo_mean_eo_violations", "saft_msee",
    ]

    print("\n" + "=" * 60)
    print(f"Summary: {df['method'].iloc[0]} on {df['dataset'].iloc[0]}")
    print(f"  n_train={df['n_train'].iloc[0]}, n_test={df['n_test'].iloc[0]}")
    print("=" * 60)
    for metric in key_metrics:
        if metric in df.columns:
            vals = df[metric]
            print(f"  {metric:<30} {vals.mean():>8.4f} +/- {vals.std():>6.4f}")


if __name__ == "__main__":
    main()
