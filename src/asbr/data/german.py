"""German Credit dataset loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

from asbr.data.loader import FairnessDataset, _bin_age, _find_col


class GermanDataset(FairnessDataset):
    """Load and preprocess the German Credit dataset.

    Label: credit risk (good -> 1, bad -> 0)
    Protected: sex (from personal_status), age_bin, foreign_worker
    """

    def __init__(self, intersection_level: int = 3, name: str | None = None) -> None:
        data = fetch_openml(data_id=31, as_frame=True, parser="auto")
        df = data.frame.copy()

        # Label: 'good' -> 1, 'bad' -> 0
        y = (df["class"].astype(str).str.strip() == "good").astype(int).values

        # Sex: derived from personal_status
        personal = df["personal_status"].astype(str).str.strip()
        sex = np.where(personal.str.contains("female", case=False), "Female", "Male")

        # Age bins
        age = df["age"].astype(float) if "age" in df.columns else pd.Series(np.zeros(len(df)))
        age_bin = _bin_age(age, [0, 30, 50, 200], ["<30", "30-50", ">50"])

        # Foreign worker
        fw_col = _find_col(df, ["foreign_worker", "ForeignWorker"])
        if fw_col:
            foreign = df[fw_col].astype(str).str.strip()
        else:
            foreign = pd.Series(["unknown"] * len(df))

        A = pd.DataFrame({
            "sex": sex,
            "age_bin": age_bin.values,
            "foreign_worker": foreign.values,
        })

        # Features
        protected_raw = {"personal_status", "age", "foreign_worker", "class"}
        feature_cols = [c for c in df.columns if c not in protected_raw]

        df_feats = df[feature_cols].copy()
        num_cols = df_feats.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_feats.select_dtypes(exclude=[np.number]).columns.tolist()

        X_num = df_feats[num_cols].values.astype(np.float32) if num_cols else np.empty((len(df), 0), dtype=np.float32)

        if cat_cols:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary")
            X_cat = enc.fit_transform(df_feats[cat_cols]).astype(np.float32)
            cat_feature_names = list(enc.get_feature_names_out(cat_cols))
        else:
            X_cat = np.empty((len(df), 0), dtype=np.float32)
            cat_feature_names = []

        X = np.hstack([X_num, X_cat])
        feature_names = num_cols + cat_feature_names

        super().__init__(
            X=X, y=y, A=A,
            feature_names=feature_names,
            protected_attrs=list(A.columns),
            name=name or "german",
            intersection_level=intersection_level,
        )
