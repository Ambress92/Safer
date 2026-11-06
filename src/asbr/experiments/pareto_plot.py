"""Pareto frontier plot: AUC vs fairness metric (SP- or EO-based).

This is the legacy two-panel plot:
- primary panel: AUC vs MSPE/MSEE
- supplementary panel: AUC vs SAFT violation rate

Shared sweep discovery and aggregation live in ``pareto_common.py`` so other
plot scripts can reuse the same data-loading path.
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


def plot_pareto(
    data,
    erm_row=None,
    dataset_label: str = "Adult",
    mode: str = "sp",
) -> matplotlib.figure.Figure:
    """Render the two-panel AUC Pareto frontier plot."""
    fairness_methods = (
        [method for method in data["method"].unique() if method != "ERM"]
        if "method" in data.columns else []
    )

    if mode == "eo":
        x_configs = [
            ("saft_msee", "saft_msee_std", "MSEE (Mean Significant EO Excess)"),
            (
                "saft_eo_violation_rate",
                "saft_eo_violation_rate_std",
                "SAFT EO violation rate (fraction of subgroups with significant TPR disparity)",
            ),
        ]
        title_metric = "MSEE"
    else:
        x_configs = [
            ("saft_mspe", "saft_mspe_std", "MSPE (Mean Significant Parity Excess)"),
            (
                "saft_violation_rate",
                "saft_violation_rate_std",
                "SAFT violation rate (fraction of subgroups with significant SP)",
            ),
        ]
        title_metric = "MSPE"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"AUC-{title_metric} Pareto Frontier - {dataset_label} dataset"
        " (MLP, error bars = SE over splits)",
        fontsize=11,
        y=1.01,
    )

    for ax, (x_col, x_std_col, x_label) in zip(axes, x_configs):
        if erm_row is not None:
            x_err = erm_row.get(x_std_col, 0.0)
            y_err = erm_row.get("auc_std", 0.0)
            ax.errorbar(
                erm_row[x_col],
                erm_row["auc"],
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
                (erm_row[x_col], erm_row["auc"]),
                textcoords="offset points",
                xytext=(6, 4),
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
                method_df["auc"],
                xerr=method_df[x_std_col],
                yerr=method_df["auc_std"],
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
                _annotate_lambd(ax, row[x_col], row["auc"], row["lambd"], color, method=method)

        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("AUC (test, mean +/- SE)", fontsize=10)
        is_primary = x_col in ("saft_mspe", "saft_msee")
        title_suffix = " [Primary]" if is_primary else " [Supplementary]"
        ax.set_title(f"AUC vs {x_label}{title_suffix}", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_ylim(bottom=ax.get_ylim()[0] - 0.02)

    plt.tight_layout()

    if mode == "eo":
        table_cols = [
            "method", "lambd", "n",
            "auc", "auc_std",
            "log_loss", "log_loss_std",
            "saft_msee", "saft_msee_std",
            "saft_eo_violation_rate", "saft_eo_violation_rate_std",
            "mean_eo", "mean_eo_std",
        ]
    else:
        table_cols = [
            "method", "lambd", "n",
            "auc", "auc_std",
            "log_loss", "log_loss_std",
            "saft_mspe", "saft_mspe_std",
            "saft_violation_rate", "saft_violation_rate_std",
            "saft_mean_sp_violations", "saft_mean_sp_violations_std",
            "mean_sp", "mean_sp_std",
        ]
    print("\nData used for plot:")
    print(
        data[[col for col in table_cols if col in data.columns]]
        .sort_values(["method", "lambd"])
        .to_string(index=False, float_format="{:.4f}".format)
    )

    return fig


def plot(
    mode: str = "sp",
    sweeps: Mapping[str, str | list[str]] | None = None,
    extras: list[str] | None = None,
    dataset_label: str = "Adult",
    lambd_min: Mapping[str, float] | None = None,
    lambd_max: Mapping[str, float] | None = None,
    include_methods: list[str] | None = None,
    exclude_methods: list[str] | None = None,
) -> matplotlib.figure.Figure:
    """Build data and render the AUC Pareto frontier plot."""
    curve_data, erm_row = build_data(
        sweeps=sweeps or {},
        extras=extras,
        lambd_min=lambd_min,
        lambd_max=lambd_max,
        include_methods=include_methods,
        exclude_methods=exclude_methods,
        erm_filter_metric="saft_msee" if mode == "eo" else "saft_mspe",
    )
    return plot_pareto(
        curve_data,
        erm_row=erm_row,
        dataset_label=dataset_label,
        mode=mode,
    )


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Plot AUC vs fairness Pareto frontier for lambda sweeps.",
    )
    add_common_plot_args(parser, include_mode=True)
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

    fig = plot_pareto(
        curve_data,
        erm_row=erm_row,
        dataset_label=args.dataset_label,
        mode=args.mode,
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")


if __name__ == "__main__":
    main()
