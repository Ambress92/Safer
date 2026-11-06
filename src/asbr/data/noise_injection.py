"""Synthetic noise group injection for Experiment 2.

Creates random subgroup assignments with known zero true disparity
to test false discovery rate control.
"""

from __future__ import annotations

import numpy as np


def inject_noise_groups(
    n: int,
    n_groups: int = 20,
    sizes: tuple[int, ...] = (5, 10, 20, 50),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create synthetic subgroup masks with no true disparity.

    Each synthetic group assigns members uniformly at random (independent
    of y), so any detected disparity is a false positive.

    Args:
        n: Total number of samples.
        n_groups: Number of synthetic groups per size bucket.
        sizes: Subgroup sizes to test.
        seed: Random seed for reproducibility.

    Returns:
        masks: Dict mapping "noise_{size}_{i}" -> boolean array of shape (n,).
    """
    rng = np.random.RandomState(seed)
    masks: dict[str, np.ndarray] = {}

    for size in sizes:
        if size >= n:
            continue
        for i in range(n_groups):
            mask = np.zeros(n, dtype=bool)
            indices = rng.choice(n, size=size, replace=False)
            mask[indices] = True
            masks[f"noise_{size}_{i}"] = mask

    return masks
