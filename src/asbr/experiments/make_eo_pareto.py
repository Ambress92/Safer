"""Pareto frontier plots for EO lambda sweep.

Each point on the curve is one lambda value (averaged over splits x seeds).
Overlays ERM and EO-Penalty baselines for reference.

Usage::

    python -m asbr.experiments.make_eo_pareto \
        --adult-sweep   multirun/2026-04-17_12-16-30/adult \
        --adult-base    multirun/2026-04-16_10-45-10/adult \
        --compas-sweep  multirun/2026-04-17_12-34-02/compas \
        --compas-base   multirun/2026-04-16_11-12-09/compas \
        --out-dir       docs/figures
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_COLOR = {
    "ERM":       "#333333",
    "EO-Penalty": "#e15759",
    "ExponGrad": "#f28e2b",
}
BASELINE_MARKER = {
    "ERM":       "*",
    "EO-Penalty": "s",
    "ExponGrad": "^",
}


def load_sweep(sweep_dir: str | Path) -> pd.DataFrame:
    """Load ASBR sweep results, adding lambd from directory name."""
    frames = []
    for csv in Path(sweep_dir).glob("ASBR-LRT-Joint/lambd=*/results.csv"):
        m = re.search(r"lambd=([\d.]+)", str(csv))
        if not m:
            continue
        df = pd.read_csv(csv)
        df["lambd"] = float(m.group(1))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No ASBR sweep results in {sweep_dir}")
    return pd.concat(frames, ignore_index=True)


def load_baselines(base_dirs: list[str | Path]) -> pd.DataFrame:
    frames = []
    for d in base_dirs:
        for csv in Path(d).glob("*/results.csv"):
            frames.append(pd.read_csv(csv))
        for csv in Path(d).glob("*/**/results.csv"):
            frames.append(pd.read_csv(csv))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["method", "dataset", "split", "seed"])


def make_pareto(
    sweep_df: pd.DataFrame,
    base_df: pd.DataFrame,
    label: str,
    out_dir: Path,
) -> None:
    slug = label.lower()

    fig, ax = plt.subplots(figsize=(7, 5))

    # --- ASBR sweep curve ---
    asbr_agg = (
        sweep_df.groupby("lambd")[["auc_test", "mean_eo_test"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    asbr_agg.columns = ["lambd", "auc_mean", "auc_std", "eo_mean", "eo_std"]
    asbr_agg = asbr_agg.sort_values("lambd")

    ax.plot(
        asbr_agg["eo_mean"], asbr_agg["auc_mean"],
        color="#4e79a7", linewidth=2, zorder=4, label="ASBR-LRT-Joint (λ sweep)",
    )
    ax.fill_between(
        asbr_agg["eo_mean"],
        asbr_agg["auc_mean"] - asbr_agg["auc_std"],
        asbr_agg["auc_mean"] + asbr_agg["auc_std"],
        color="#4e79a7", alpha=0.15,
    )
    for _, row in asbr_agg.iterrows():
        ax.scatter(row["eo_mean"], row["auc_mean"],
                   color="#4e79a7", s=60, zorder=5)
        ax.annotate(
            f"λ={row['lambd']:.0f}" if row["lambd"] >= 1 else f"λ={row['lambd']}",
            (row["eo_mean"], row["auc_mean"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7, color="#4e79a7",
        )

    # --- Baselines ---
    for method in ["ERM", "EO-Penalty", "ExponGrad"]:
        sub = base_df[base_df["method"] == method]
        if sub.empty:
            continue
        eo_mean = sub["mean_eo_test"].mean()
        eo_std  = sub["mean_eo_test"].std()
        auc_mean = sub["auc_test"].mean()
        auc_std  = sub["auc_test"].std()
        ax.errorbar(
            eo_mean, auc_mean,
            xerr=eo_std, yerr=auc_std,
            fmt=BASELINE_MARKER.get(method, "o"),
            color=BASELINE_COLOR.get(method, "gray"),
            capsize=3, markersize=9, linewidth=1.5,
            label=method, zorder=6,
        )

    ax.set_xlabel("Mean EO gap (test)", fontsize=10)
    ax.set_ylabel("AUC (test)", fontsize=10)
    ax.set_title(f"{label} — Pareto frontier: AUC vs Mean EO", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.annotate("← fairer", xy=(0.02, 0.04), xycoords="axes fraction",
                fontsize=8, color="gray", style="italic")
    ax.annotate("↑ more accurate", xy=(0.02, 0.09), xycoords="axes fraction",
                fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"pareto_eo_frontier_{slug}.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult-sweep",    default=None)
    parser.add_argument("--adult-base",     nargs="+", default=None)
    parser.add_argument("--compas-sweep",   default=None)
    parser.add_argument("--compas-base",    nargs="+", default=None)
    parser.add_argument("--diabetes-sweep", default=None)
    parser.add_argument("--diabetes-base",  nargs="+", default=None)
    parser.add_argument("--celeba-sweep",   default=None)
    parser.add_argument("--celeba-base",    nargs="+", default=None)
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.adult_sweep and args.adult_base:
        make_pareto(load_sweep(args.adult_sweep), load_baselines(args.adult_base), "Adult", out_dir)
    if args.compas_sweep and args.compas_base:
        make_pareto(load_sweep(args.compas_sweep), load_baselines(args.compas_base), "COMPAS", out_dir)
    if args.diabetes_sweep and args.diabetes_base:
        make_pareto(load_sweep(args.diabetes_sweep), load_baselines(args.diabetes_base), "Diabetes", out_dir)
    if args.celeba_sweep and args.celeba_base:
        make_pareto(load_sweep(args.celeba_sweep), load_baselines(args.celeba_base), "CelebA", out_dir)


if __name__ == "__main__":
    main()
