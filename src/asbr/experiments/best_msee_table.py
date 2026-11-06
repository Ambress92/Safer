"""Select best-MSEE operating points under an ERM log-loss budget.

This script is intended for paper tables that complement the Pareto plot:
for each method sweep, choose the hyperparameter with minimum test MSEE among
points whose mean test log loss is within X% of ERM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from asbr.experiments.pareto_common import (
    ALPHA_METHODS,
    TARGET_REDUCTION_METHODS,
    _discover_sweep_root,
    _load_one,
)


def _parameter_name(method: str) -> str:
    if method in ALPHA_METHODS:
        return "alpha"
    if method in TARGET_REDUCTION_METHODS:
        return "target"
    if "ExponGrad" in method:
        return "epsilon"
    return "lambda"


def _fmt_pm(row: pd.Series, mean_col: str, se_col: str, digits: int = 4) -> str:
    mean = row.get(mean_col)
    se = row.get(se_col)
    if pd.isna(mean):
        return ""
    if pd.isna(se):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {se:.{digits}f}"


def build_best_msee_table(
    sweep_root: str | Path,
    erm_results: str | Path,
    tolerance_pct: float,
    include_methods: list[str] | None = None,
    exclude_methods: list[str] | None = None,
) -> pd.DataFrame:
    """Return one selected row per method.

    Selection rule:
      1. Compute ERM mean test log loss.
      2. Keep method points with log_loss <= ERM * (1 + tolerance_pct / 100).
      3. Select the point with minimum saft_msee; break ties by lower log_loss.
      4. If no point qualifies, report the method's minimum-log-loss point and
         mark ``eligible=False`` so it is visible but not treated as selected.
    """
    erm_stats = _load_one(erm_results)
    erm_log_loss = erm_stats["log_loss"]
    threshold = erm_log_loss * (1.0 + tolerance_pct / 100.0)

    entries = _discover_sweep_root(
        str(sweep_root),
        include_methods=include_methods,
        exclude_methods=exclude_methods,
    )
    data = pd.DataFrame(entries)
    if data.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for method, method_df in data.groupby("method", sort=True):
        method_df = method_df.copy()
        method_df = method_df[method_df["log_loss"].notna() & method_df["saft_msee"].notna()]
        if method_df.empty:
            continue

        eligible_df = method_df[method_df["log_loss"] <= threshold]
        eligible = not eligible_df.empty
        pool = eligible_df if eligible else method_df
        sort_cols = ["saft_msee", "log_loss"] if eligible else ["log_loss", "saft_msee"]
        selected = pool.sort_values(sort_cols, ascending=True).iloc[0]

        msee_min = float(method_df["saft_msee"].min())
        msee_max = float(method_df["saft_msee"].max())
        log_loss_min = float(method_df["log_loss"].min())
        log_loss_max = float(method_df["log_loss"].max())

        rows.append({
            "method": method,
            "param": _parameter_name(method),
            "value": selected["lambd"],
            "eligible": eligible,
            "n": int(selected["n"]),
            "n_splits": int(selected.get("n_splits", 0)),
            "erm_log_loss": erm_log_loss,
            "log_loss_threshold": threshold,
            "log_loss": selected["log_loss"],
            "log_loss_se": selected["log_loss_std"],
            "auc": selected["auc"],
            "auc_se": selected["auc_std"],
            "saft_msee": selected["saft_msee"],
            "saft_msee_se": selected["saft_msee_std"],
            "saft_eo_violation_rate": selected["saft_eo_violation_rate"],
            "saft_eo_violation_rate_se": selected["saft_eo_violation_rate_std"],
            "mean_eo": selected["mean_eo"],
            "mean_eo_se": selected["mean_eo_std"],
            "msee_sweep_min": msee_min,
            "msee_sweep_max": msee_max,
            "log_loss_sweep_min": log_loss_min,
            "log_loss_sweep_max": log_loss_max,
            "path": selected["path"],
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["eligible", "saft_msee"], ascending=[False, True])


def to_markdown_table(table: pd.DataFrame, tolerance_pct: float) -> str:
    """Format a compact Markdown table for paper drafting."""
    if table.empty:
        return ""

    rows = []
    for _, row in table.iterrows():
        value = row["value"]
        value_s = f"{value:g}" if pd.notna(value) else ""
        rows.append({
            "Method": row["method"],
            "Param": f"{row['param']}={value_s}",
            "Eligible": "yes" if row["eligible"] else "no",
            "Log loss": _fmt_pm(row, "log_loss", "log_loss_se"),
            "AUC": _fmt_pm(row, "auc", "auc_se"),
            "MSEE": _fmt_pm(row, "saft_msee", "saft_msee_se"),
            "EO viol.": _fmt_pm(row, "saft_eo_violation_rate", "saft_eo_violation_rate_se"),
            "Mean EO": _fmt_pm(row, "mean_eo", "mean_eo_se"),
            "MSEE range": f"{row['msee_sweep_min']:.4f}-{row['msee_sweep_max']:.4f}",
        })
    out = pd.DataFrame(rows)
    header = (
        f"Best MSEE with mean test log loss within {tolerance_pct:g}% of ERM "
        f"(ineligible rows show the minimum-log-loss point)."
    )
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for record in out.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(record[col]) for col in cols) + " |")
    return header + "\n\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a best-MSEE table under an ERM log-loss budget.",
    )
    parser.add_argument("--sweep-root", required=True, help="Dataset sweep root, e.g. multirun/<tag>/adult")
    parser.add_argument("--erm-results", required=True, help="Path to ERM results.csv")
    parser.add_argument("--tolerance-pct", type=float, default=1.0)
    parser.add_argument("--include-method", action="append", default=None)
    parser.add_argument("--exclude-method", action="append", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    table = build_best_msee_table(
        sweep_root=args.sweep_root,
        erm_results=args.erm_results,
        tolerance_pct=args.tolerance_pct,
        include_methods=args.include_method,
        exclude_methods=args.exclude_method,
    )

    markdown = to_markdown_table(table, args.tolerance_pct)
    print(markdown if markdown else "No data found.")

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_csv, index=False)
        print(f"\nSaved CSV to {output_csv}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown + "\n", encoding="utf-8")
        print(f"Saved Markdown to {output_md}")


if __name__ == "__main__":
    main()
