"""Report dataset stress-variant statistics for the paper.

This script mirrors the experiment runner's split/training/evaluation path:

* outer splits use StratifiedShuffleSplit(random_state=42)
* validation splits use train_test_split(random_state=seed)
* ERM is trained with the default TrainingConfig
* SAFT summaries use the same alpha, n_tilde, and optional multiplicity
  correction as the Hydra experiments

It prints table-ready split statistics and ERM audit summaries, and can also
write CSV/Markdown files for direct use while drafting the paper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from asbr.data.adult import AdultDataset
from asbr.data.compas import CompasDataset
from asbr.data.german import GermanDataset
from asbr.data.intersections import build_subgroup_masks
from asbr.data.loader import FairnessDataset, prepare_split
from asbr.evaluation.evaluator import compute_all_metrics_from_predictions
from asbr.experiments.config import TrainingConfig
from asbr.fairness.saft import (
    apply_multiplicity_correction,
    apply_multiplicity_correction_eo,
    saft_audit,
    saft_eo_audit,
    saft_eo_summary,
    saft_summary,
)
from asbr.methods.erm import ERMMethod


DATASETS: dict[str, Callable[[int], FairnessDataset]] = {
    "adult": lambda level: AdultDataset(intersection_level=level),
    "compas": lambda level: CompasDataset(intersection_level=level),
    "german": lambda level: GermanDataset(intersection_level=level),
}


@dataclass(frozen=True)
class StressSpec:
    dataset: str
    level: int
    test_size: float
    label: str
    regime: str

    @property
    def split_label(self) -> str:
        train_pct = round((1.0 - self.test_size) * 100)
        test_pct = round(self.test_size * 100)
        return f"{train_pct}/{test_pct}"


DEFAULT_SPECS = (
    StressSpec("adult", 4, 0.9, "Adult Income", "high intersectionality, low budget"),
    StressSpec("compas", 3, 0.8, "COMPAS", "sparse intersectional regime"),
    StressSpec("german", 3, 0.5, "German Credit", "safety regime"),
)


def _parse_spec(raw: str) -> StressSpec:
    """Parse dataset:level:test_size:label[:regime]."""
    parts = raw.split(":")
    if len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError(
            "spec must be dataset:level:test_size:label[:regime]"
        )
    dataset, level, test_size, label = parts[:4]
    regime = parts[4] if len(parts) == 5 else ""
    if dataset not in DATASETS:
        raise argparse.ArgumentTypeError(
            f"unknown dataset {dataset!r}; valid choices are {sorted(DATASETS)}"
        )
    return StressSpec(dataset, int(level), float(test_size), label, regime)


def _std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _mean_std(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        rows.append({
            "metric": col,
            "mean": float(df[col].mean()),
            "std": _std(df[col]),
        })
    return pd.DataFrame(rows)


def _positive_counts(
    y: np.ndarray,
    masks: dict[str, np.ndarray],
) -> tuple[list[int], list[int]]:
    all_pos: list[int] = []
    eo_defined_pos: list[int] = []
    positives = y == 1
    for mask in masks.values():
        mask_bool = np.asarray(mask, dtype=bool)
        n_pos_s = int(np.sum(positives & mask_bool))
        n_pos_sbar = int(np.sum(positives & ~mask_bool))
        all_pos.append(n_pos_s)
        if n_pos_s >= 2 and n_pos_sbar >= 2:
            eo_defined_pos.append(n_pos_s)
    return all_pos, eo_defined_pos


def _mask_size_summary(
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, float]:
    sizes = [int(np.sum(mask)) for mask in masks.values()]
    pos_all, pos_eo = _positive_counts(y, masks)
    if not sizes:
        return {
            f"{prefix}_n_subgroups": 0,
            f"{prefix}_n_mean": float("nan"),
            f"{prefix}_n_median": float("nan"),
            f"{prefix}_n_min": float("nan"),
            f"{prefix}_n_p10": float("nan"),
            f"{prefix}_pos_mean_all": float("nan"),
            f"{prefix}_pos_mean_eo_defined": float("nan"),
            f"{prefix}_eo_defined_subgroups": 0,
        }
    return {
        f"{prefix}_n_subgroups": len(sizes),
        f"{prefix}_n_mean": float(np.mean(sizes)),
        f"{prefix}_n_median": float(np.median(sizes)),
        f"{prefix}_n_min": float(np.min(sizes)),
        f"{prefix}_n_p10": float(np.percentile(sizes, 10)),
        f"{prefix}_pos_mean_all": float(np.mean(pos_all)) if pos_all else float("nan"),
        f"{prefix}_pos_mean_eo_defined": float(np.mean(pos_eo)) if pos_eo else float("nan"),
        f"{prefix}_eo_defined_subgroups": len(pos_eo),
    }


def _dataset_static_row(spec: StressSpec, dataset: FairnessDataset) -> dict[str, object]:
    levels = {
        attr: int(dataset.A[attr].nunique())
        for attr in dataset.protected_attrs
    }
    level_values = {
        attr: ", ".join(map(str, sorted(dataset.A[attr].unique())))
        for attr in dataset.protected_attrs
    }
    return {
        "dataset": spec.dataset,
        "label": spec.label,
        "level": spec.level,
        "split": spec.split_label,
        "n_total": len(dataset.y),
        "n_features": dataset.X.shape[1],
        "positive_rate": float(np.mean(dataset.y)),
        "protected_attrs": ", ".join(dataset.protected_attrs),
        "protected_levels": "; ".join(f"{k}={v}" for k, v in levels.items()),
        "protected_values": "; ".join(f"{k}: {v}" for k, v in level_values.items()),
    }


def _apply_saft_correction_sp(results, method: str, alpha: float):
    if method == "none":
        return results
    return apply_multiplicity_correction(results, method=method, alpha=alpha)


def _apply_saft_correction_eo(results, method: str, alpha: float):
    if method == "none":
        return results
    return apply_multiplicity_correction_eo(results, method=method, alpha=alpha)


def _run_spec(
    spec: StressSpec,
    *,
    n_splits: int,
    seed: int,
    val_size: float,
    saft_alpha: float,
    saft_n_tilde: int,
    saft_correction: str,
    skip_erm: bool,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    dataset = DATASETS[spec.dataset](spec.level)
    static_row = _dataset_static_row(spec, dataset)

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=spec.test_size,
        random_state=42,
    )
    split_rows: list[dict[str, object]] = []
    erm_rows: list[dict[str, object]] = []
    training_config = TrainingConfig()

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(dataset.X, dataset.y)):
        X_fold_train = dataset.X[train_idx]
        y_fold_train = dataset.y[train_idx]
        A_fold_train = dataset.A.iloc[train_idx].reset_index(drop=True)
        X_fold_test = dataset.X[test_idx]
        y_fold_test = dataset.y[test_idx]
        A_fold_test = dataset.A.iloc[test_idx].reset_index(drop=True)

        masks_train = build_subgroup_masks(A_fold_train, level=spec.level, min_size=2)
        masks_test = build_subgroup_masks(A_fold_test, level=spec.level, min_size=2)

        split_row: dict[str, object] = {
            "dataset": spec.dataset,
            "label": spec.label,
            "level": spec.level,
            "split": spec.split_label,
            "split_idx": split_idx,
            "n_train": len(y_fold_train),
            "n_test": len(y_fold_test),
            "train_positive_rate": float(np.mean(y_fold_train)),
            "test_positive_rate": float(np.mean(y_fold_test)),
        }
        split_row.update(_mask_size_summary(y_fold_train, masks_train, "train"))
        split_row.update(_mask_size_summary(y_fold_test, masks_test, "test"))
        split_row["train_examples_per_subgroup_budget"] = (
            len(y_fold_train) / split_row["train_n_subgroups"]
            if split_row["train_n_subgroups"] else float("nan")
        )
        split_row["train_positive_examples_per_subgroup_budget"] = (
            int(np.sum(y_fold_train == 1)) / split_row["train_n_subgroups"]
            if split_row["train_n_subgroups"] else float("nan")
        )
        split_row["test_examples_per_subgroup_budget"] = (
            len(y_fold_test) / split_row["test_n_subgroups"]
            if split_row["test_n_subgroups"] else float("nan")
        )
        split_row["test_positive_examples_per_subgroup_budget"] = (
            int(np.sum(y_fold_test == 1)) / split_row["test_n_subgroups"]
            if split_row["test_n_subgroups"] else float("nan")
        )
        split_rows.append(split_row)

        if skip_erm:
            continue

        idx_tr, idx_vl = train_test_split(
            np.arange(len(X_fold_train)),
            test_size=val_size,
            stratify=y_fold_train,
            random_state=seed,
        )
        X_tr, X_vl = X_fold_train[idx_tr], X_fold_train[idx_vl]
        y_tr, y_vl = y_fold_train[idx_tr], y_fold_train[idx_vl]
        A_tr = A_fold_train.iloc[idx_tr].reset_index(drop=True)

        X_tr_s, X_vl_s, scaler = prepare_split(X_tr, X_vl)
        X_train_scaled = np.nan_to_num(scaler.transform(X_fold_train).astype(np.float32))
        X_test_scaled = np.nan_to_num(scaler.transform(X_fold_test).astype(np.float32))

        np.random.seed(seed)
        torch.manual_seed(seed)
        method = ERMMethod()
        method.train(X_tr_s, y_tr, X_vl_s, y_vl, A_tr, spec.level, training_config)
        y_prob_train, y_pred_train = method.predict(X_train_scaled)
        y_prob_test, y_pred_test = method.predict(X_test_scaled)

        metrics = compute_all_metrics_from_predictions(
            y_prob_train,
            y_pred_train,
            y_fold_train,
            masks_train,
            y_prob_test,
            y_pred_test,
            y_fold_test,
            masks_test,
        )

        sp_results = saft_audit(
            y_pred_test,
            y_fold_test,
            masks_test,
            alpha=saft_alpha,
            n_tilde=saft_n_tilde,
            seed=seed,
        )
        sp_results = _apply_saft_correction_sp(sp_results, saft_correction, saft_alpha)
        sp_summary = saft_summary(sp_results)

        eo_results = saft_eo_audit(
            y_pred_test,
            y_fold_test,
            masks_test,
            alpha=saft_alpha,
            n_tilde=saft_n_tilde,
            seed=seed,
        )
        eo_results = _apply_saft_correction_eo(eo_results, saft_correction, saft_alpha)
        eo_summary = saft_eo_summary(eo_results)

        erm_rows.append({
            "dataset": spec.dataset,
            "label": spec.label,
            "level": spec.level,
            "split": spec.split_label,
            "split_idx": split_idx,
            "seed": seed,
            **metrics,
            "saft_n_subgroups": sp_summary.n_subgroups,
            "saft_n_violations": sp_summary.n_violations,
            "saft_violation_rate": sp_summary.violation_rate,
            "saft_mspe": sp_summary.mspe,
            "saft_eo_n_subgroups": eo_summary.n_subgroups,
            "saft_eo_n_violations": eo_summary.n_violations,
            "saft_eo_violation_rate": eo_summary.violation_rate,
            "saft_msee": eo_summary.msee,
        })

    return static_row, pd.DataFrame(split_rows), pd.DataFrame(erm_rows)


def _summarize_for_table(split_df: pd.DataFrame, spec: StressSpec) -> dict[str, object]:
    return {
        "Dataset": spec.label,
        "Level": spec.level,
        "Split": spec.split_label,
        "n_train_mean": split_df["n_train"].mean(),
        "n_train_std": _std(split_df["n_train"]),
        "S_train_mean": split_df["train_n_subgroups"].mean(),
        "S_train_std": _std(split_df["train_n_subgroups"]),
        "S_test_mean": split_df["test_n_subgroups"].mean(),
        "S_test_std": _std(split_df["test_n_subgroups"]),
        "nS_train_mean": split_df["train_examples_per_subgroup_budget"].mean(),
        "nS_train_std": _std(split_df["train_examples_per_subgroup_budget"]),
        "mS_train_mean": split_df["train_positive_examples_per_subgroup_budget"].mean(),
        "mS_train_std": _std(split_df["train_positive_examples_per_subgroup_budget"]),
        "raw_mask_nS_train_mean": split_df["train_n_mean"].mean(),
        "raw_mask_mS_train_mean": split_df["train_pos_mean_eo_defined"].mean(),
        "eo_defined_train_mean": split_df["train_eo_defined_subgroups"].mean(),
        "Regime": spec.regime,
    }


def _print_summary(
    static_df: pd.DataFrame,
    split_df: pd.DataFrame,
    erm_df: pd.DataFrame,
    table_df: pd.DataFrame,
) -> None:
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 180)

    print("\nStatic dataset statistics")
    print(static_df.to_string(index=False))

    print("\nTable-ready stress variant summary")
    table_cols = [
        "Dataset", "Level", "Split",
        "n_train_mean", "S_train_mean", "S_test_mean",
        "nS_train_mean", "mS_train_mean", "eo_defined_train_mean", "Regime",
    ]
    print(table_df[table_cols].to_string(index=False, float_format="{:.2f}".format))

    split_metrics = [
        "n_train", "n_test",
        "train_n_subgroups", "test_n_subgroups",
        "train_examples_per_subgroup_budget",
        "train_positive_examples_per_subgroup_budget",
        "train_n_mean", "train_n_median", "train_n_min", "train_n_p10",
        "train_pos_mean_all", "train_pos_mean_eo_defined",
        "train_eo_defined_subgroups",
    ]
    print("\nSplit statistics, mean +/- std")
    for dataset, group in split_df.groupby("dataset", sort=False):
        print(f"\n[{dataset}]")
        print(_mean_std(group, split_metrics).to_string(index=False, float_format="{:.3f}".format))

    if erm_df.empty:
        return

    erm_metrics = [
        "accuracy_test", "auc_test", "log_loss_test",
        "saft_n_violations", "saft_violation_rate", "saft_mspe",
        "saft_eo_n_subgroups", "saft_eo_n_violations",
        "saft_eo_violation_rate", "saft_msee",
        "max_sp_test", "mean_sp_test", "max_eo_test", "mean_eo_test",
    ]
    print("\nERM audit statistics, mean +/- std")
    for dataset, group in erm_df.groupby("dataset", sort=False):
        print(f"\n[{dataset}]")
        print(_mean_std(group, erm_metrics).to_string(index=False, float_format="{:.4f}".format))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paper dataset stress statistics and ERM SAFT audits.",
    )
    parser.add_argument(
        "--spec",
        action="append",
        type=_parse_spec,
        help="dataset:level:test_size:label[:regime]. Defaults to Adult 10/90, COMPAS 20/80, German 50/50.",
    )
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Single ERM seed / validation split seed.")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--saft-alpha", type=float, default=0.05)
    parser.add_argument("--saft-n-tilde", type=int, default=30)
    parser.add_argument(
        "--saft-correction",
        choices=("none", "bonferroni", "bh"),
        default="bh",
    )
    parser.add_argument("--skip-erm", action="store_true", help="Only compute dataset/split statistics.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/tables/dataset_stress_stats"),
        help="Directory for CSV/Markdown outputs.",
    )
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)

    specs = tuple(args.spec) if args.spec else DEFAULT_SPECS
    static_rows: list[dict[str, object]] = []
    split_frames: list[pd.DataFrame] = []
    erm_frames: list[pd.DataFrame] = []
    table_rows: list[dict[str, object]] = []

    for spec in specs:
        print(f"\n==> {spec.label}: level={spec.level}, split={spec.split_label}")
        static_row, split_df, erm_df = _run_spec(
            spec,
            n_splits=args.n_splits,
            seed=args.seed,
            val_size=args.val_size,
            saft_alpha=args.saft_alpha,
            saft_n_tilde=args.saft_n_tilde,
            saft_correction=args.saft_correction,
            skip_erm=args.skip_erm,
        )
        static_rows.append(static_row)
        split_frames.append(split_df)
        if not erm_df.empty:
            erm_frames.append(erm_df)
        table_rows.append(_summarize_for_table(split_df, spec))

    static_df = pd.DataFrame(static_rows)
    split_df_all = pd.concat(split_frames, ignore_index=True)
    erm_df_all = pd.concat(erm_frames, ignore_index=True) if erm_frames else pd.DataFrame()
    table_df = pd.DataFrame(table_rows)

    _print_summary(static_df, split_df_all, erm_df_all, table_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    static_df.to_csv(args.out_dir / "static.csv", index=False)
    split_df_all.to_csv(args.out_dir / "splits.csv", index=False)
    table_df.to_csv(args.out_dir / "table_summary.csv", index=False)
    try:
        table_df.to_markdown(args.out_dir / "table_summary.md", index=False)
    except ImportError:
        table_df.to_csv(args.out_dir / "table_summary.md", index=False)
    if not erm_df_all.empty:
        erm_df_all.to_csv(args.out_dir / "erm_audits.csv", index=False)
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
