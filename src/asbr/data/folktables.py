"""ACS Income dataset loader via the folktables package (Ding et al., 2021).

Requires: pip install folktables
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from asbr.data.loader import FairnessDataset, _bin_age


_ALL_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

# RAC1P race codes — small groups collapsed to "Other"
_RAC1P_COLLAPSED = {1: "White", 2: "Black", 6: "Asian", 9: "TwoOrMore"}


class ACSIncomeDataset(FairnessDataset):
    """ACS Income dataset (Ding et al., 2021) with optional stratified subsampling.

    Label: annual income > $50K (PINCP > 50000)
    Protected: race (RAC1P, collapsed to 5 groups), sex (SEX), age_bin (AGEP)

    Downloads the full ACS data for the requested states, then optionally draws
    a stratified random sample of ``n_samples`` rows (stratified by label) so
    that the effective dataset size is comparable to other benchmarks.

    Args:
        survey_year: ACS survey year string, e.g. '2018'.
        states: List of US state abbreviations, or None for all 50 states + DC.
        horizon: '1-Year' or '5-Year'.
        n_samples: If set, draw a stratified subsample of this many rows after
            loading. None keeps the full dataset. Recommended: 50_000 for
            quick experiments, None for final paper runs.
        random_state: Seed for the stratified subsample (ignored if n_samples
            is None).
        intersection_level: Max intersection depth for subgroup masks.
        name: Dataset name override.
    """

    def __init__(
        self,
        survey_year: str = "2018",
        states: list[str] | None = None,
        horizon: str = "1-Year",
        n_samples: int | None = 50_000,
        random_state: int = 42,
        intersection_level: int = 3,
        name: str | None = None,
    ) -> None:
        try:
            from folktables import ACSDataSource, ACSIncome
        except ImportError as exc:
            raise ImportError(
                "folktables is required for ACSIncomeDataset. "
                "Install it with: pip install folktables"
            ) from exc

        # Use a fixed cache directory so Hydra's chdir=true doesn't cause
        # each job to re-download all state files into its own output folder.
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "folktables")
        os.makedirs(cache_dir, exist_ok=True)

        data_source = ACSDataSource(
            survey_year=survey_year,
            horizon=horizon,
            survey="person",
            root_dir=cache_dir,
        )
        acs_data = data_source.get_data(
            states=states if states is not None else _ALL_STATES,
            download=True,
        )
        X_raw, y_raw, _ = ACSIncome.df_to_numpy(acs_data)

        feature_cols = ACSIncome.features
        df = pd.DataFrame(X_raw, columns=feature_cols)
        y_full = y_raw.astype(int)

        # --- Stratified subsample ---
        if n_samples is not None and n_samples < len(y_full):
            rng = np.random.default_rng(random_state)
            idx_pos = np.where(y_full == 1)[0]
            idx_neg = np.where(y_full == 0)[0]
            pos_frac = len(idx_pos) / len(y_full)
            n_pos = int(round(n_samples * pos_frac))
            n_neg = n_samples - n_pos
            chosen = np.concatenate([
                rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False),
                rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False),
            ])
            chosen = rng.permutation(chosen)
            df = df.iloc[chosen].reset_index(drop=True)
            y = y_full[chosen]
        else:
            y = y_full

        # --- Protected attributes ---
        rac1p = df["RAC1P"].astype(int)
        race = rac1p.map(lambda r: _RAC1P_COLLAPSED.get(r, "Other"))
        sex = df["SEX"].astype(int).map({1: "Male", 2: "Female"}).fillna("Male")
        age_bin = _bin_age(
            df["AGEP"].astype(float),
            bins=[0, 25, 45, 65, 200],
            labels=["<25", "25-45", "45-65", ">65"],
        )

        A = pd.DataFrame({
            "race": race.values,
            "sex": sex.values,
            "age_bin": age_bin.values,
        })

        # --- Features: drop protected columns ---
        protected_cols = {"RAC1P", "SEX", "AGEP"}
        feat_cols = [c for c in feature_cols if c not in protected_cols]
        df_feats = df[feat_cols].copy()

        num_cols = df_feats.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_feats.select_dtypes(exclude=[np.number]).columns.tolist()

        X_num = (
            df_feats[num_cols].values.astype(np.float32)
            if num_cols
            else np.empty((len(df_feats), 0), dtype=np.float32)
        )

        if cat_cols:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary")
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
            name=name or "acs_income",
            intersection_level=intersection_level,
        )
