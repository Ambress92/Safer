"""Select best-log-loss operating points under an MSEE budget.

This is the dual table to ``best_msee_table.py``: for each method sweep,
choose the hyperparameter with minimum mean test log loss among points whose
test MSEE is below a chosen threshold.

Example usage for RUN_TAG="compas_l3_5050":

.venv/bin/python -m asbr.experiments.best_logloss_table \
  --sweep-root multirun/compas_l3_5050/compas \
  --msee-rel-pct 50 \
  --erm-results outputs/compas/ERM/compas_l3_5050/results.csv \
  --output-csv docs/tables/compas_eo_best_logloss_msee_rel50pct.csv \
  --output-md docs/tables/compas_eo_best_logloss_msee_rel50pct.md


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


def build_best_logloss_table(
    sweep_root: str | Path,
    msee_threshold: float,
    *,
    erm_msee: float | None = None,
    msee_rel_pct: float | None = None,
    msee_rel_factor: float | None = None,
    include_methods: list[str] | None = None,
    exclude_methods: list[str] | None = None,
) -> pd.DataFrame:
    """Return one selected row per method.

    Selection rule:
      1. Keep method points with ``saft_msee <= msee_threshold``.
      2. Select the point with minimum log_loss; break ties by lower MSEE.
      3. If no point qualifies, report the method's minimum-MSEE point and
         mark ``eligible=False`` so it is visible but not treated as selected.
    """
    if msee_rel_pct is not None and msee_rel_factor is not None:
        raise ValueError("Use only one of msee_rel_pct or msee_rel_factor.")
    if msee_rel_pct is not None or msee_rel_factor is not None:
        if erm_msee is None or pd.isna(erm_msee):
            raise ValueError("Relative MSEE thresholds require erm_msee.")
        factor = msee_rel_factor if msee_rel_factor is not None else (1.0 - msee_rel_pct / 100.0)
        msee_threshold = erm_msee * factor

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

        eligible_df = method_df[method_df["saft_msee"] <= msee_threshold]
        eligible = not eligible_df.empty
        pool = eligible_df if eligible else method_df
        sort_cols = ["log_loss", "saft_msee"] if eligible else ["saft_msee", "log_loss"]
        selected = pool.sort_values(sort_cols, ascending=True).iloc[0]

        rows.append({
            "method": method,
            "param": _parameter_name(method),
            "value": selected["lambd"],
            "eligible": eligible,
            "n": int(selected["n"]),
            "n_splits": int(selected.get("n_splits", 0)),
            "erm_msee": erm_msee,
            "msee_threshold": msee_threshold,
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
            "msee_sweep_min": float(method_df["saft_msee"].min()),
            "msee_sweep_max": float(method_df["saft_msee"].max()),
            "log_loss_sweep_min": float(method_df["log_loss"].min()),
            "log_loss_sweep_max": float(method_df["log_loss"].max()),
            "path": selected["path"],
        })

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["eligible", "log_loss"], ascending=[False, True])


def to_markdown_table(
    table: pd.DataFrame,
    msee_threshold: float,
    *,
    erm_msee: float | None = None,
    msee_rel_pct: float | None = None,
    msee_rel_factor: float | None = None,
) -> str:
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
            "Log-loss range": f"{row['log_loss_sweep_min']:.4f}-{row['log_loss_sweep_max']:.4f}",
        })
    out = pd.DataFrame(rows)
    threshold = float(table["msee_threshold"].iloc[0]) if "msee_threshold" in table else msee_threshold
    if msee_rel_pct is not None:
        header = (
            f"Best mean test log loss with MSEE at least {msee_rel_pct:g}% below ERM "
            f"(ERM MSEE={erm_msee:.4f}, threshold={threshold:.4f}; "
            "ineligible rows show the minimum-MSEE point)."
        )
    elif msee_rel_factor is not None:
        header = (
            f"Best mean test log loss with MSEE <= {msee_rel_factor:g} x ERM "
            f"(ERM MSEE={erm_msee:.4f}, threshold={threshold:.4f}; "
            "ineligible rows show the minimum-MSEE point)."
        )
    else:
        header = (
            f"Best mean test log loss with MSEE <= {threshold:g} "
            f"(ineligible rows show the minimum-MSEE point)."
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
        description="Build a best-log-loss table under an MSEE budget.",
    )
    parser.add_argument("--sweep-root", required=True, help="Dataset sweep root, e.g. multirun/<tag>/adult")
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument("--msee-threshold", type=float, help="Absolute MSEE threshold.")
    threshold_group.add_argument(
        "--msee-rel-pct",
        type=float,
        help="Require MSEE to be this percent below ERM, e.g. 50 means <= 50% of ERM MSEE.",
    )
    threshold_group.add_argument(
        "--msee-rel-factor",
        type=float,
        help="Require MSEE <= this factor times ERM MSEE, e.g. 0.5.",
    )
    parser.add_argument("--erm-results", default=None, help="Required for relative MSEE thresholds.")
    parser.add_argument("--include-method", action="append", default=None)
    parser.add_argument("--exclude-method", action="append", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    erm_msee = None
    if args.msee_rel_pct is not None or args.msee_rel_factor is not None:
        if not args.erm_results:
            parser.error("--erm-results is required with relative MSEE thresholds.")
        erm_msee = _load_one(args.erm_results)["saft_msee"]
    msee_threshold = args.msee_threshold if args.msee_threshold is not None else float("nan")

    table = build_best_logloss_table(
        sweep_root=args.sweep_root,
        msee_threshold=msee_threshold,
        erm_msee=erm_msee,
        msee_rel_pct=args.msee_rel_pct,
        msee_rel_factor=args.msee_rel_factor,
        include_methods=args.include_method,
        exclude_methods=args.exclude_method,
    )

    markdown = to_markdown_table(
        table,
        msee_threshold,
        erm_msee=erm_msee,
        msee_rel_pct=args.msee_rel_pct,
        msee_rel_factor=args.msee_rel_factor,
    )
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
