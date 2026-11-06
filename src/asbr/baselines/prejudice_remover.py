"""AIF360 Prejudice Remover Regularizer wrapper for intersectional fairness.

Wraps AIF360's classic (non-sklearn) PrejudiceRemover algorithm, which adds
a regularization term to logistic regression that penalizes mutual information
between predictions and the sensitive attribute (Kamishima et al. 2012).

PrejudiceRemover only supports a single protected attribute, so we
integer-encode the intersectional composite column.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.datasets import BinaryLabelDataset

from asbr.baselines.aif360_expgrad import build_intersectional_column


@dataclass
class PrejudiceRemoverConfig:
    """Configuration for PrejudiceRemover."""

    eta: float = 1.0  # fairness penalty strength


class PrejudiceRemoverWrapper:
    """Wrapper around AIF360's PrejudiceRemover.

    Converts numpy/pandas inputs to ``BinaryLabelDataset`` for the classic
    AIF360 API and exposes a familiar ``fit`` / ``predict`` / ``predict_proba``
    interface.
    """

    _PROT_COL = "_intersect_group"
    _LABEL_COL = "_label"

    def __init__(self, config: PrejudiceRemoverConfig | None = None):
        self.config = config or PrejudiceRemoverConfig()
        self.model_: PrejudiceRemover | None = None
        self._fallback_model: LogisticRegression | DummyClassifier | None = None
        self._group_map: dict[str, int] | None = None
        self._feature_names: list[str] | None = None

    @staticmethod
    def _has_two_classes(labels: np.ndarray) -> bool:
        """Return whether a label slice contains both binary classes."""
        return len(np.unique(labels)) >= 2

    @staticmethod
    def _ensure_current_python_on_path() -> None:
        """Make AIF360's ``python`` subprocess resolve to this interpreter."""
        python_dir = os.path.dirname(sys.executable)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if python_dir and python_dir not in path_parts:
            os.environ["PATH"] = os.pathsep.join([python_dir, *path_parts])

    def _encode_train_groups(self, intersect_col: pd.Series, y_train: np.ndarray) -> np.ndarray:
        """Encode groups, coalescing one-class groups AIF360 cannot initialize."""
        group_labels = intersect_col.astype(str)
        unsafe_mask = pd.Series(False, index=group_labels.index)
        safe_group_sizes: dict[str, int] = {}

        for group, idx in group_labels.groupby(group_labels).groups.items():
            if not self._has_two_classes(y_train[list(idx)]):
                unsafe_mask.loc[list(idx)] = True
            else:
                safe_group_sizes[str(group)] = len(idx)

        if unsafe_mask.any():
            pooled_labels = y_train[unsafe_mask.to_numpy()]
            if self._has_two_classes(pooled_labels):
                group_labels = group_labels.mask(unsafe_mask, "__pooled_sparse__")
                print(
                    "  [PrejudiceRemover] pooled "
                    f"{int(unsafe_mask.sum())} sample(s) from one-class protected groups"
                )
            elif safe_group_sizes:
                target_group = max(safe_group_sizes, key=safe_group_sizes.get)
                group_labels = group_labels.mask(unsafe_mask, target_group)
                print(
                    "  [PrejudiceRemover] merged "
                    f"{int(unsafe_mask.sum())} one-class protected-group sample(s) "
                    f"into {target_group!r}"
                )
            else:
                group_labels = pd.Series("__all_groups__", index=group_labels.index)
                print(
                    "  [PrejudiceRemover] all sparse protected groups had one label class; "
                    "using one pooled protected group for this fold"
                )

        unique_groups = sorted(group_labels.unique())
        self._group_map = {group: i for i, group in enumerate(unique_groups)}
        return group_labels.map(self._group_map).to_numpy()

    def _make_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_ints: np.ndarray,
    ) -> BinaryLabelDataset:
        """Build a BinaryLabelDataset from arrays."""
        n, d = X.shape
        feat_names = [f"f{i}" for i in range(d)]
        self._feature_names = feat_names

        df = pd.DataFrame(X, columns=feat_names)
        df[self._LABEL_COL] = y.astype(float)
        df[self._PROT_COL] = group_ints.astype(float)

        return BinaryLabelDataset(
            df=df,
            label_names=[self._LABEL_COL],
            protected_attribute_names=[self._PROT_COL],
            favorable_label=1.0,
            unfavorable_label=0.0,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        A_train: pd.DataFrame,
        intersection_level: int = 3,
    ) -> PrejudiceRemoverWrapper:
        """Fit the PrejudiceRemover model.

        Args:
            X_train: Training features, shape (n, d).
            y_train: Training labels, shape (n,).
            A_train: Protected attributes DataFrame.
            intersection_level: Maximum intersection level for group construction.

        Returns:
            self
        """
        intersect_col = build_intersectional_column(A_train, level=intersection_level)

        group_ints = self._encode_train_groups(intersect_col, y_train)

        dataset = self._make_dataset(X_train, y_train, group_ints)

        # AIF360's bundled Kamishima implementation only supports a binary
        # sensitive feature and initializes one LR per sensitive value. If one
        # side is absent in a sparse fold, it fits LogisticRegression on 0 rows.
        privileged = dataset.privileged_protected_attributes[0]
        sensitive_binary = np.isin(group_ints.astype(float), privileged)
        if len(np.unique(sensitive_binary)) < 2:
            if self._has_two_classes(y_train):
                self._fallback_model = LogisticRegression(
                    solver="liblinear", max_iter=1000, random_state=0,
                )
            else:
                self._fallback_model = DummyClassifier(strategy="most_frequent")
            self._fallback_model.fit(X_train, y_train)
            print(
                "  [PrejudiceRemover] only one binary protected side is present "
                "after AIF360 conversion; using logistic fallback for this fold"
            )
            return self

        self.model_ = PrejudiceRemover(
            eta=self.config.eta,
            sensitive_attr=self._PROT_COL,
            class_attr=self._LABEL_COL,
        )
        self._ensure_current_python_on_path()
        self.model_.fit(dataset)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for the favorable class.

        Args:
            X: Features, shape (n, d).

        Returns:
            y_prob: Predicted probabilities, shape (n,).
        """
        if self._fallback_model is not None:
            proba = self._fallback_model.predict_proba(X)
            classes = list(self._fallback_model.classes_)
            return proba[:, classes.index(1)] if 1 in classes else np.zeros(X.shape[0])

        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        dummy_y = np.zeros(X.shape[0])
        dummy_group = np.zeros(X.shape[0])
        dataset = self._make_dataset(X, dummy_y, dummy_group)

        self._ensure_current_python_on_path()
        result = self.model_.predict(dataset)
        return np.asarray(result.scores).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels.

        Args:
            X: Features, shape (n, d).

        Returns:
            y_pred: Binary predictions, shape (n,).
        """
        if self._fallback_model is not None:
            return self._fallback_model.predict(X).astype(int)

        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        dummy_y = np.zeros(X.shape[0])
        dummy_group = np.zeros(X.shape[0])
        dataset = self._make_dataset(X, dummy_y, dummy_group)

        self._ensure_current_python_on_path()
        result = self.model_.predict(dataset)
        return np.asarray(result.labels).reshape(-1).astype(int)
