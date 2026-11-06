"""Intersectional subgroup construction from protected attributes.

Builds subgroup boolean masks at multiple intersection levels:
  Level 1: single attributes  (e.g., race=Black)
  Level 2: pairwise            (e.g., race=Black & sex=Female)
  Level 3: three-way           (e.g., race=Black & sex=Female & age=<25)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import torch


def build_subgroup_masks(
    A: pd.DataFrame,
    level: int = 3,
    min_size: int = 1,
) -> dict[str, np.ndarray]:
    """Build boolean subgroup masks from protected attributes.

    Creates all intersectional subgroups up to the specified level.

    Args:
        A: DataFrame of categorical protected attributes, shape (n, k).
        level: Maximum intersection level (1, 2, or 3).
        min_size: Minimum subgroup size to include.

    Returns:
        masks: Dict mapping subgroup name -> boolean array of shape (n,).
    """
    n = len(A)
    cols = list(A.columns)
    masks: dict[str, np.ndarray] = {}

    for lv in range(1, min(level, len(cols)) + 1):
        for attr_combo in combinations(cols, lv):
            # Get unique value combinations for this set of attributes
            grouped = A.groupby(list(attr_combo), observed=True)
            for values, idx in grouped.groups.items():
                if not isinstance(values, tuple):
                    values = (values,)
                mask = np.zeros(n, dtype=bool)
                mask[idx.values] = True

                if mask.sum() < min_size:
                    continue

                # Name: "race=Black&sex=Female"
                name = "&".join(
                    f"{attr}={val}" for attr, val in zip(attr_combo, values)
                )
                masks[name] = mask

    return masks


def masks_to_torch(
    masks: dict[str, np.ndarray],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert numpy boolean masks to torch tensors."""
    return {
        name: torch.tensor(mask, dtype=torch.bool, device=device)
        for name, mask in masks.items()
    }


def subgroup_summary(
    masks: dict[str, np.ndarray],
    y: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summary statistics for all subgroups.

    Args:
        masks: Subgroup masks dict.
        y: Optional binary labels for computing positive rate.

    Returns:
        DataFrame with columns: subgroup, size, fraction, [pos_rate].
    """
    n = next(iter(masks.values())).shape[0] if masks else 0
    rows = []
    for name, mask in sorted(masks.items(), key=lambda x: x[1].sum()):
        row = {
            "subgroup": name,
            "size": int(mask.sum()),
            "fraction": mask.sum() / n if n > 0 else 0.0,
        }
        if y is not None:
            row["pos_rate"] = float(y[mask].mean()) if mask.sum() > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
