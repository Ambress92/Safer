"""Comparison plots for fixed-lambda baseline experiments.

Generates two figures per dataset:
  1. Scatter: AUC vs mean_SP (Pareto trade-off)
  2. Bar chart: key metrics side-by-side

Usage::

    python -m asbr.experiments.make_comparison_plots \
        --acs-dir  multirun/2026-04-02_08-52-48/acs_income \
        --diab-dir multirun/<timestamp>/diabetes \
        --out-dir  docs/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

METHOD_ORDER = [
    "ERM", "SP-Penalty", "ExponGrad",
    "ASBR-LRT-Joint", "ASBR-Wald-Joint", "ASBR-LRT",
]
METHOD_COLOR = {
    "ERM":            "#333333",
    "SP-Penalty":     "#e15759",
    "ExponGrad":      "#f28e2b",
    "ASBR-LRT-Joint": "#4e79a7",
    "ASBR-Wald-Joint":"#76b7b2",
    "ASBR-LRT":       "#59a14f",
}
METHOD_MARKER = {
    "ERM":            "*",
    "SP-Penalty":     "s",
    "ExponGrad":      "^",
    "ASBR-LRT-Joint": "o",
    "ASBR-Wald-Joint":"D",
    "ASBR-LRT":       "v",
}
METHOD_SHORT = {
    "ERM":            "ERM",
    "SP-Penalty":     "SP-Pen.",
    "ExponGrad":      "ExpGrad",
    "ASBR-LRT-Joint": "ASBR-LRT\nJoint",
    "ASBR-Wald-Joint":"ASBR-Wald\nJoint",
    "ASBR-LRT":       "ASBR-LRT\n2-phase",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset_results(results_dirs: str | Path | list[str | Path]) -> pd.DataFrame:
    """Load all results.csv from one or more method subdirs into one DataFrame."""
    if isinstance(results_dirs, (str, Path)):
        results_dirs = [results_dirs]
    frames = []
    for d in results_dirs:
        root = Path(d)
        for csv in root.glob("*/results.csv"):
            frames.append(pd.read_csv(csv))
        for csv in root.glob("*/*/results.csv"):
            frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError(f"No results.csv found under {results_dirs}")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["method", "dataset", "split", "seed"])


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-method mean ± std for key metrics."""
    cols = [
        "accuracy_test", "auc_test",
        "mean_sp_test", "max_sp_test",
        "saft_violation_rate", "saft_mspe",
    ]
    agg = df.groupby("method")[cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["method"] + [f"{c}_{s}" for c, s in agg.columns[1:]]
    return agg


# ---------------------------------------------------------------------------
# Figure 1: AUC vs mean_SP scatter
# ---------------------------------------------------------------------------

def plot_pareto_scatter(summary: pd.DataFrame, dataset_label: str, ax: plt.Axes) -> None:
    methods = [m for m in METHOD_ORDER if m in summary["method"].values]
    for method in methods:
        row = summary[summary["method"] == method].iloc[0]
        c  = METHOD_COLOR.get(method, "gray")
        mk = METHOD_MARKER.get(method, "o")
        ax.errorbar(
            row["mean_sp_test_mean"], row["auc_test_mean"],
            xerr=row["mean_sp_test_std"], yerr=row["auc_test_std"],
            fmt=mk, color=c, capsize=3, markersize=9,
            linewidth=1.5, label=method, zorder=5,
        )
        ax.annotate(
            method, (row["mean_sp_test_mean"], row["auc_test_mean"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7.5, color=c,
        )
    ax.set_xlabel("Mean SP (test, mean ± std)", fontsize=10)
    ax.set_ylabel("AUC (test, mean ± std)", fontsize=10)
    ax.set_title(f"{dataset_label} — AUC vs Mean SP", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")
    # Arrow hint: better = top-left
    ax.annotate(
        "← fairer", xy=(0.02, 0.04), xycoords="axes fraction",
        fontsize=8, color="gray", style="italic",
    )
    ax.annotate(
        "↑ more accurate", xy=(0.02, 0.09), xycoords="axes fraction",
        fontsize=8, color="gray", style="italic",
    )


# ---------------------------------------------------------------------------
# Figure 2: Bar chart of key metrics
# ---------------------------------------------------------------------------

def plot_bar_metrics(summary: pd.DataFrame, dataset_label: str, ax_row: list[plt.Axes]) -> None:
    metrics = [
        ("accuracy_test", "Accuracy"),
        ("auc_test",       "AUC"),
        ("mean_sp_test",   "Mean SP ↓"),
        ("saft_violation_rate", "SAFT viol. rate ↓"),
    ]
    methods = [m for m in METHOD_ORDER if m in summary["method"].values]
    x = np.arange(len(methods))
    colors = [METHOD_COLOR.get(m, "gray") for m in methods]
    short_labels = [METHOD_SHORT.get(m, m) for m in methods]

    for ax, (col, label) in zip(ax_row, metrics):
        means = [summary[summary["method"] == m][f"{col}_mean"].values[0] for m in methods]
        stds  = [summary[summary["method"] == m][f"{col}_std"].values[0]  for m in methods]
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, alpha=0.85, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=7.5)
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_ylim(bottom=0)
        # Value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=6.5,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_figures(results_dir: str | Path | list, dataset_label: str, out_dir: Path) -> None:
    df = load_dataset_results(results_dir)
    summary = summarise(df)

    slug = dataset_label.lower().replace(" ", "_")

    # --- Figure 1: Pareto scatter ---
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_pareto_scatter(summary, dataset_label, ax)
    plt.tight_layout()
    out = out_dir / f"pareto_{slug}.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()

    # --- Figure 2: Bar metrics ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"{dataset_label} — Method Comparison (10 splits × 3 seeds)",
        fontsize=12, y=1.02,
    )
    plot_bar_metrics(summary, dataset_label, list(axes))
    plt.tight_layout()
    out = out_dir / f"bars_{slug}.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()

    # --- Print summary table ---
    print(f"\n=== {dataset_label} ===")
    display_cols = [
        "method",
        "accuracy_test_mean", "accuracy_test_std",
        "auc_test_mean", "auc_test_std",
        "mean_sp_test_mean", "mean_sp_test_std",
        "saft_violation_rate_mean", "saft_violation_rate_std",
    ]
    print(summary[display_cols].to_string(index=False, float_format="{:.4f}".format))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult-dir",  default=None, nargs="+")
    parser.add_argument("--compas-dir", default=None, nargs="+")
    parser.add_argument("--german-dir", default=None, nargs="+")
    parser.add_argument("--diab-dir",   default=None, nargs="+")
    parser.add_argument("--celeba-dir", default=None, nargs="+")
    parser.add_argument("--acs-dir",    default=None, nargs="+")
    parser.add_argument("--out-dir",    default="docs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.adult_dir:
        make_figures(args.adult_dir,  "Adult",      out_dir)
    if args.compas_dir:
        make_figures(args.compas_dir, "COMPAS",     out_dir)
    if args.german_dir:
        make_figures(args.german_dir, "German",     out_dir)
    if args.diab_dir:
        make_figures(args.diab_dir,   "Diabetes",   out_dir)
    if args.celeba_dir:
        make_figures(args.celeba_dir, "CelebA",     out_dir)
    if args.acs_dir:
        make_figures(args.acs_dir,    "ACS Income", out_dir)


if __name__ == "__main__":
    main()
