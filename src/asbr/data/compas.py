"""COMPAS Recidivism dataset loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from asbr.data.loader import FairnessDataset


class CompasDataset(FairnessDataset):
    """Load and preprocess the COMPAS recidivism dataset.

    Label: twoyearrecid
    Protected: race, sex, age_cat

    The OpenML version (data_id=45039) is pre-processed with one-hot encoded
    race and age_cat columns.  We reconstruct the original categoricals for
    the protected-attribute DataFrame.
    """

    def __init__(self, intersection_level: int = 3, name: str | None = None) -> None:
        data = fetch_openml(data_id=45039, as_frame=True, parser="auto")
        df = data.frame.copy()
        df = df.dropna().reset_index(drop=True)

        # --- Target ---
        y = df["twoyearrecid"].astype(int).values

        # --- Reconstruct race from one-hot ---
        race_aa = df["race_African-American"].astype(int)
        race_series = race_aa.map({1: "African-American", 0: "Caucasian"})

        # --- Reconstruct age_cat from one-hot ---
        age_cat_cols = {
            "age_cat_Lessthan25": "<25",
            "age_cat_25-45": "25-45",
            "age_cat_Greaterthan45": ">45",
        }
        age_cat = pd.Series("<25", index=df.index)  # default
        for col, label in age_cat_cols.items():
            if col in df.columns:
                age_cat = age_cat.where(df[col].astype(int) != 1, label)

        # --- Sex ---
        sex_series = df["sex"].astype(int).map({0: "Female", 1: "Male"})

        A = pd.DataFrame({
            "race": race_series.values,
            "sex": sex_series.values,
            "age_cat": age_cat.values,
        })

        # --- Features: everything except target and protected-attribute source columns ---
        exclude = {
            "twoyearrecid",
            "race_African-American", "race_Caucasian",
            "age_cat_25-45", "age_cat_Greaterthan45", "age_cat_Lessthan25",
            "sex",
        }
        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols].astype(np.float32).values
        feature_names = feature_cols

        super().__init__(
            X=X, y=y, A=A,
            feature_names=feature_names,
            protected_attrs=list(A.columns),
            name=name or "compas",
            intersection_level=intersection_level,
        )
