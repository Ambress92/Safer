"""CelebA dataset loader (pre-extracted ResNet-18 features).

Expects features extracted by scripts/extract_celeba_features.py:
    features.npy   (N, 512)  float32
    labels.npy     (N,)      int8    — Attractive
    attrs.csv      (N, 3)    str     — Male, Young, Pale_Skin
    partition.npy  (N,)      int8    — 0=train, 1=val, 2=test
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from asbr.data.loader import FairnessDataset


class CelebADataset(FairnessDataset):
    """CelebA pre-extracted features for intersectional fairness.

    Label: Attractive
    Protected: Male x Young x Pale_Skin

    Args:
        features_dir: Directory containing features.npy, labels.npy,
            attrs.csv, partition.npy produced by extract_celeba_features.py.
        intersection_level: Maximum intersection depth (default 3 = all three
            protected attributes combined).
        name: Dataset name override.
    """

    def __init__(
        self,
        features_dir: str,
        intersection_level: int = 3,
        name: str | None = None,
    ) -> None:
        root = Path(features_dir)

        X = np.load(root / "features.npy").astype(np.float32)
        y = np.load(root / "labels.npy").astype(int)
        A = pd.read_csv(root / "attrs.csv")

        feature_names = [f"resnet18_{i}" for i in range(X.shape[1])]

        super().__init__(
            X=X,
            y=y,
            A=A,
            feature_names=feature_names,
            protected_attrs=list(A.columns),
            name=name or "celeba",
            intersection_level=intersection_level,
        )
