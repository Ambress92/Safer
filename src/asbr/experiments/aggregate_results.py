"""Aggregate results from Hydra multirun directories.

Usage::

    from asbr.experiments.aggregate_results import aggregate_multirun
    df = aggregate_multirun("multirun/2026-02-16_12-00-00")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def aggregate_multirun(multirun_dir: str | Path) -> pd.DataFrame:
    """Collect all ``results.csv`` files from a Hydra multirun directory.

    Args:
        multirun_dir: Path to the Hydra multirun root (contains subdirectories
            for each sweep combination, each with a ``results.csv``).

    Returns:
        Concatenated DataFrame of all results.

    Raises:
        FileNotFoundError: If no ``results.csv`` files are found.
    """
    multirun_path = Path(multirun_dir)
    csv_files = sorted(multirun_path.rglob("results.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No results.csv files found under {multirun_path}"
        )

    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    print(f"Aggregated {len(csv_files)} result files, {len(df)} total rows")
    return df
