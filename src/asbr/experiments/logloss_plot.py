"""Single-panel Pareto plot: log loss vs fairness metric (SP- or EO-based).

This is the utility plot recommended for threshold-dependent SAFT metrics:
- SP mode: log loss vs MSPE
- EO mode: log loss vs MSEE

It reuses sweep discovery and aggregation from ``pareto_common.py`` so it can
be driven by the same multirun directories and ``results.csv`` files as the
legacy AUC Pareto plot.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from asbr.experiments.pareto_common import (
    _annotate_lambd,
    add_common_plot_args,
    build_data,
    get_method_color,
    get_method_label,
    get_method_marker,
    parse_lambd_bounds,
    sweeps_from_args,
)


def plot_logloss_pareto(
    data,
    erm_row=None,
    dataset_label: str = "Adult",
    mode: str = "eo",
    erm_halfline: bool = True,
    highlight_best: bool = True,
) -> matplotlib.figure.Figure:
    """Render the single-panel log loss Pareto plot."""
    fairness_methods = (
        [method for method in data["method"].unique() if method != "ERM"]
        if "method" in data.columns else []
    )

    if mode == "eo":
        x_col = "saft_msee"
        x_std_col = "saft_msee_std"
        x_label = "MSEE (Mean Significant EO Excess)"
        title_metric = "MSEE"
        table_cols = [
            "method", "lambd", "n",
            "log_loss", "log_loss_std",
            "auc", "auc_std",
            "saft_msee", "saft_msee_std",
            "saft_eo_violation_rate", "saft_eo_violation_rate_std",
            "mean_eo", "mean_eo_std",
        ]
    else:
        x_col = "saft_mspe"
        x_std_col = "saft_mspe_std"
        x_label = "MSPE (Mean Significant Parity Excess)"
        title_metric = "MSPE"
        table_cols = [
            "method", "lambd", "n",
            "log_loss", "log_loss_std",
            "auc", "auc_std",
            "saft_mspe", "saft_mspe_std",
            "saft_violation_rate", "saft_violation_rate_std",
            "mean_sp", "mean_sp_std",
        ]

    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    # fig.suptitle(
    #     f"Log Loss-{title_metric} Pareto Frontier - {dataset_label} dataset"
    #     " (MLP, error bars = SE over splits)",
    #     fontsize=11,
    #     y=0.98,
    # )

    erm_half = None
    if erm_row is not None:
        erm_half = erm_row[x_col] / 2.0
        if erm_halfline:
            ax.axvline(erm_half, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, zorder=1)

        x_err = erm_row.get(x_std_col, 0.0)
        y_err = erm_row.get("log_loss_std", 0.0)
        ax.errorbar(
            erm_row[x_col],
            erm_row["log_loss"],
            xerr=x_err if not np.isnan(x_err) else None,
            yerr=y_err if not np.isnan(y_err) else None,
            fmt="*",
            markersize=14,
            color=get_method_color("ERM"),
            capsize=4,
            linewidth=1.5,
            zorder=6,
            label="ERM (unconstrained)",
        )
        ax.annotate(
            "ERM",
            (erm_row[x_col], erm_row["log_loss"]),
            textcoords="offset points",
            xytext=(-5, 10),
            fontsize=8,
            color=get_method_color("ERM"),
            fontweight="bold",
        )

    for method in fairness_methods:
        method_df = data[data["method"] == method].sort_values("lambd")
        if method_df.empty:
            continue
        color = get_method_color(method)
        marker = get_method_marker(method)
        label = get_method_label(method)

        ax.errorbar(
            method_df[x_col],
            method_df["log_loss"],
            xerr=method_df[x_std_col],
            yerr=method_df["log_loss_std"],
            fmt=f"{marker}-",
            color=color,
            capsize=3,
            linewidth=1.8,
            markersize=7,
            label=label,
            alpha=0.9,
            zorder=5,
        )
        for _, row in method_df.iterrows():
            _annotate_lambd(ax, row[x_col], row["log_loss"], row["lambd"], color, method=method)

        # Highlight the point closest to (but below) the ERM half-line
        if highlight_best and erm_half is not None:
            below = method_df[method_df[x_col] <= erm_half]
            if not below.empty:
                best = below.loc[below[x_col].idxmax()]
                ax.plot(
                    best[x_col], best["log_loss"],
                    marker=marker, color=color, markersize=8,
                    markeredgecolor="black", markeredgewidth=0.8,
                    zorder=7, linestyle="none",
                )

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Log loss (test, mean +/- SE)", fontsize=10)
    # ax.set_title(f"Log loss vs {x_label}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()

    print("\nData used for plot:")
    print(
        data[[col for col in table_cols if col in data.columns]]
        .sort_values(["method", "lambd"])
        .to_string(index=False, float_format="{:.4f}".format)
    )

    return fig


def plot(
    mode: str = "eo",
    sweeps: Mapping[str, str | list[str]] | None = None,
    extras: list[str] | None = None,
    dataset_label: str = "Adult",
    lambd_min: Mapping[str, float] | None = None,
    lambd_max: Mapping[str, float] | None = None,
    include_methods: list[str] | None = None,
    exclude_methods: list[str] | None = None,
    erm_halfline: bool = True,
    highlight_best: bool = True,
) -> matplotlib.figure.Figure:
    """Build data and render the log loss Pareto plot."""
    curve_data, erm_row = build_data(
        sweeps=sweeps or {},
        extras=extras,
        lambd_min=lambd_min,
        lambd_max=lambd_max,
        include_methods=include_methods,
        exclude_methods=exclude_methods,
        erm_filter_metric="saft_msee" if mode == "eo" else "saft_mspe",
    )
    return plot_logloss_pareto(
        curve_data,
        erm_row=erm_row,
        dataset_label=dataset_label,
        mode=mode,
        erm_halfline=erm_halfline,
        highlight_best=highlight_best,
    )


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Plot log loss vs fairness Pareto frontier for lambda sweeps.",
    )
    add_common_plot_args(parser, include_mode=True)
    parser.set_defaults(mode="eo", output="docs/figures/logloss_pareto_adult.pdf")
    args = parser.parse_args()

    sweeps = sweeps_from_args(args)
    lambd_min = parse_lambd_bounds(args.lambd_min) or None
    lambd_max = parse_lambd_bounds(args.lambd_max) or None

    curve_data, erm_row = build_data(
        sweeps=sweeps,
        extras=args.extra,
        lambd_min=lambd_min,
        lambd_max=lambd_max,
        include_methods=args.include_method,
        exclude_methods=args.exclude_method,
        erm_filter_metric="saft_msee" if args.mode == "eo" else "saft_mspe",
    )
    if curve_data.empty and erm_row is None:
        parser.error("No data found. Provide sweep flags or --extra entries.")

    fig = plot_logloss_pareto(
        curve_data,
        erm_row=erm_row,
        dataset_label=args.dataset_label,
        mode=args.mode,
        erm_halfline=not args.no_erm_halfline,
        highlight_best=not args.no_highlight_best,
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")


if __name__ == "__main__":
    main()
