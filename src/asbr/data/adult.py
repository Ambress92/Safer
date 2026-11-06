"""Adult Income dataset loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

from asbr.data.loader import FairnessDataset, _bin_age


class AdultDataset(FairnessDataset):
    """Load and preprocess the Adult Income dataset.

    Label: income >50K
    Protected: race, sex, age_bin, native_country_bin
    """

    def __init__(self, intersection_level: int = 3, name: str | None = None) -> None:
        data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
        df = data.frame.copy()

        # Drop rows with missing values
        df = df.dropna().reset_index(drop=True)

        # Label — column name varies across openml versions
        label_col = "income" if "income" in df.columns else "class"
        y = (df[label_col].astype(str).str.strip().str.contains(">50K")).astype(int).values

        # Protected attributes
        age_bins = [0, 25, 45, 65, 200]
        age_labels = ["<25", "25-45", "45-65", ">65"]
        A = pd.DataFrame({
            "race": df["race"].astype(str).str.strip().values,
            "sex": df["sex"].astype(str).str.strip().values,
            "age_bin": _bin_age(df["age"].astype(float), age_bins, age_labels).values,
            "native_country": np.where(
                df["native-country"].astype(str).str.strip() == "United-States",
                "US", "non-US"
            ),
        })

        # Features: all columns except label and protected attributes
        protected_raw = {"race", "sex", "age", "native-country", "income", "class"}
        feature_cols = [c for c in df.columns if c not in protected_raw]

        # Separate numeric and categorical features
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
            name=name or "adult",
            intersection_level=intersection_level,
        )
