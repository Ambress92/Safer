"""Diabetes 130-US Hospitals dataset loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

from asbr.data.loader import FairnessDataset


class DiabetesDataset(FairnessDataset):
    """Load and preprocess the Diabetes 130-US Hospitals dataset (1999-2008).

    Label: early readmission ("<30" -> 1) vs late readmission (">30" -> 0).
    "NO" cases are excluded. This is a harder, clinically richer task: both
    classes are readmitted patients, so the model must distinguish urgency
    rather than trivially separating sick from healthy.
    The dataset is subsampled to n_pos positives and n_neg negatives (balanced).

    Protected: race, gender, age_bin
    Source: OpenML name="diabetes130us"
    """

    def __init__(
        self,
        n_pos: int = 10_000,
        n_neg: int = 10_000,
        random_state: int = 42,
        intersection_level: int = 3,
        name: str | None = None,
    ) -> None:
        data = fetch_openml(name="diabetes130us", version=1, as_frame=True, parser="auto")

        # fetch_openml may place the target in data.target rather than data.frame
        df = data.frame.copy() if data.frame is not None else pd.DataFrame()
        if data.target is not None and isinstance(data.target, pd.Series):
            target_name = data.target.name or "readmitted"
            if target_name not in df.columns:
                df[target_name] = data.target.values

        # Find the readmitted column regardless of exact capitalisation
        from asbr.data.loader import _find_col
        readmitted_col = _find_col(df, ["readmitted", "Readmitted", "READMITTED"])
        if readmitted_col is None:
            raise RuntimeError(
                f"Could not find readmitted column in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.dropna(subset=[readmitted_col]).reset_index(drop=True)
        readmitted = df[readmitted_col].astype(str).str.strip()

        # --- Keep only "<30" (positive) and ">30" (negative); drop "NO" ---
        df_pos = df[readmitted == "<30"].reset_index(drop=True)
        df_neg = df[readmitted == ">30"].reset_index(drop=True)

        rng = np.random.default_rng(random_state)
        df_pos = df_pos.iloc[rng.choice(len(df_pos), size=min(n_pos, len(df_pos)), replace=False)]
        df_neg = df_neg.iloc[rng.choice(len(df_neg), size=min(n_neg, len(df_neg)), replace=False)]

        df_pos = df_pos.copy()
        df_neg = df_neg.copy()
        df_pos["_y"] = 1
        df_neg["_y"] = 0
        df = pd.concat([df_pos, df_neg], ignore_index=True)
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

        y = df["_y"].values.astype(int)

        # --- Protected attributes ---
        race = df["race"].astype(str).str.strip()
        # Collapse small groups for intersectional stability
        major_races = {"Caucasian", "AfricanAmerican", "Hispanic", "Asian"}
        race = race.where(race.isin(major_races), other="Other")

        gender = df["gender"].astype(str).str.strip()
        # Drop "Unknown/Invalid" rows if present (very few)
        valid_mask = ~gender.isin(["Unknown/Invalid"])
        df = df[valid_mask].reset_index(drop=True)
        y = y[valid_mask.values.astype(bool)]
        race = race[valid_mask].reset_index(drop=True)
        gender = gender[valid_mask].reset_index(drop=True)

        # age is already categorical like "[0-10)", "[10-20)", ..., "[90-100)"
        age = df["age"].astype(str).str.strip()
        # Bin into 3 coarser groups for intersectional tractability
        def _coarsen_age(a: str) -> str:
            young = {"[0-10)", "[10-20)", "[20-30)"}
            mid = {"[30-40)", "[40-50)", "[50-60)"}
            if a in young:
                return "<30"
            if a in mid:
                return "30-60"
            return "60+"

        age_bin = age.map(_coarsen_age)

        A = pd.DataFrame({
            "race": race.values,
            "gender": gender.values,
            "age_bin": age_bin.values,
        })

        # --- Features: all columns except target and protected ---
        protected_raw = {readmitted_col, "race", "gender", "age"}
        feature_cols = [c for c in df.columns if c not in protected_raw]

        df_feats = df[feature_cols].copy()

        # Replace "?" with NaN and drop columns with >30% missing
        df_feats = df_feats.replace("?", np.nan)
        missing_frac = df_feats.isnull().mean()
        df_feats = df_feats.loc[:, missing_frac <= 0.30]
        df_feats = df_feats.fillna(df_feats.mode().iloc[0])

        num_cols = df_feats.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_feats.select_dtypes(exclude=[np.number]).columns.tolist()

        X_num = (
            df_feats[num_cols].values.astype(np.float32)
            if num_cols
            else np.empty((len(df_feats), 0), dtype=np.float32)
        )

        if cat_cols:
            enc = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                drop="if_binary",
                max_categories=20,
            )
            X_cat = enc.fit_transform(df_feats[cat_cols]).astype(np.float32)
            cat_feature_names = list(enc.get_feature_names_out(cat_cols))
        else:
            X_cat = np.empty((len(df_feats), 0), dtype=np.float32)
            cat_feature_names = []

        X = np.hstack([X_num, X_cat])
        feature_names = num_cols + cat_feature_names

        super().__init__(
            X=X, y=y, A=A,
            feature_names=feature_names,
            protected_attrs=list(A.columns),
            name=name or "diabetes",
            intersection_level=intersection_level,
        )
