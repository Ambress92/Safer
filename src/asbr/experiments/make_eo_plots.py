"""EO comparison plots for Equal Opportunity experiments.

Usage::

    python -m asbr.experiments.make_eo_plots \
        --adult-dirs  multirun/<ts1>/adult multirun/<ts2>/adult ... \
        --compas-dirs multirun/<ts1>/compas ... \
        --out-dir     docs/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["ERM", "EO-Penalty", "ExponGrad", "ASBR-LRT-Joint"]
METHOD_COLOR = {
    "ERM":            "#333333",
    "EO-Penalty":     "#e15759",
    "ExponGrad":      "#f28e2b",
    "ASBR-LRT-Joint": "#4e79a7",
}
METHOD_MARKER = {
    "ERM":            "*",
    "EO-Penalty":     "s",
    "ExponGrad":      "^",
    "ASBR-LRT-Joint": "o",
}
METHOD_SHORT = {
    "ERM":            "ERM",
    "EO-Penalty":     "EO-Pen.",
    "ExponGrad":      "ExpGrad",
    "ASBR-LRT-Joint": "ASBR-LRT\nJoint",
}


def load_results(dirs: list[str | Path]) -> pd.DataFrame:
    frames = []
    for d in dirs:
        for csv in Path(d).glob("*/results.csv"):
            frames.append(pd.read_csv(csv))
        for csv in Path(d).glob("*/**/results.csv"):
            frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError(f"No results.csv found under {dirs}")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["method", "dataset", "split", "seed"])


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["accuracy_test", "auc_test", "mean_eo_test", "max_eo_test", "saft_eo_violation_rate"]
    agg = df.groupby("method")[cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["method"] + [f"{c}_{s}" for c, s in agg.columns[1:]]
    return agg


def plot_pareto(summary: pd.DataFrame, label: str, ax: plt.Axes) -> None:
    methods = [m for m in METHOD_ORDER if m in summary["method"].values]
    for m in methods:
        row = summary[summary["method"] == m].iloc[0]
        ax.errorbar(
            row["mean_eo_test_mean"], row["auc_test_mean"],
            xerr=row["mean_eo_test_std"], yerr=row["auc_test_std"],
            fmt=METHOD_MARKER.get(m, "o"), color=METHOD_COLOR.get(m, "gray"),
            capsize=3, markersize=9, linewidth=1.5, label=m, zorder=5,
        )
        ax.annotate(m, (row["mean_eo_test_mean"], row["auc_test_mean"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color=METHOD_COLOR.get(m, "gray"))
    ax.set_xlabel("Mean EO gap (test, mean ± std)", fontsize=10)
    ax.set_ylabel("AUC (test, mean ± std)", fontsize=10)
    ax.set_title(f"{label} — AUC vs Mean EO", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.annotate("← fairer", xy=(0.02, 0.04), xycoords="axes fraction",
                fontsize=8, color="gray", style="italic")
    ax.annotate("↑ more accurate", xy=(0.02, 0.09), xycoords="axes fraction",
                fontsize=8, color="gray", style="italic")


def plot_bars(summary: pd.DataFrame, label: str, axes: list[plt.Axes]) -> None:
    metrics = [
        ("accuracy_test",       "Accuracy"),
        ("auc_test",            "AUC"),
        ("mean_eo_test",        "Mean EO ↓"),
        ("saft_eo_violation_rate", "SAFT EO viol. rate ↓"),
    ]
    methods = [m for m in METHOD_ORDER if m in summary["method"].values]
    x = np.arange(len(methods))
    colors = [METHOD_COLOR.get(m, "gray") for m in methods]
    short = [METHOD_SHORT.get(m, m) for m in methods]

    for ax, (col, title) in zip(axes, metrics):
        means = [summary[summary["method"] == m][f"{col}_mean"].values[0] for m in methods]
        stds  = [summary[summary["method"] == m][f"{col}_std"].values[0]  for m in methods]
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, alpha=0.85, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_ylim(bottom=0)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=6.5)


def make_figures(dirs: list[str | Path], label: str, out_dir: Path) -> None:
    df = load_results(dirs)
    summary = summarise(df)
    slug = label.lower().replace(" ", "_")

    # Pareto
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_pareto(summary, label, ax)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"pareto_eo_{slug}.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close()

    # Bars
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"{label} EO — Method Comparison (10 splits × 3 seeds)", fontsize=12, y=1.02)
    plot_bars(summary, label, list(axes))
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"bars_eo_{slug}.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close()

    # Summary table
    print(f"\n=== {label} EO ===")
    cols = ["method", "accuracy_test_mean", "accuracy_test_std",
            "auc_test_mean", "auc_test_std",
            "mean_eo_test_mean", "mean_eo_test_std",
            "saft_eo_violation_rate_mean", "saft_eo_violation_rate_std"]
    print(summary[cols].to_string(index=False, float_format="{:.4f}".format))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult-dirs",  nargs="+", default=None)
    parser.add_argument("--compas-dirs", nargs="+", default=None)
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.adult_dirs:
        make_figures(args.adult_dirs, "Adult", out_dir)
    if args.compas_dirs:
        make_figures(args.compas_dirs, "COMPAS", out_dir)
    if not args.adult_dirs and not args.compas_dirs:
        parser.error("Provide --adult-dirs and/or --compas-dirs")


if __name__ == "__main__":
    main()
