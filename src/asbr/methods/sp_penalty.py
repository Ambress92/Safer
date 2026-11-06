"""SP-Penalty and EO-Penalty baseline methods.

SP-Penalty
----------
BCE + λ·Σ_S SP²(S) — raw squared statistical parity, no significance gate.
Uses the Wald-based ASBRLoss with use_hinge=False, which routes to the
``sp_est**2`` branch — a genuine SP² penalty.

EO-Penalty
----------
BCE + λ·Σ_S EO²(S) — raw squared TPR-disparity, no significance gate.
Uses the LRT-based ASBRLRTLoss with use_sp_magnitude=True, use_hinge=False,
which routes to the ``metric_tensor**2`` branch — a genuine EO² penalty.
Without use_sp_magnitude=True, use_hinge=False falls through to the raw LRT
statistic Λ_S (already size- and variance-scaled), which is NOT EO².
Subgroups with no positive ground-truth labels on either side are silently skipped.

Usage::

    python -m asbr.experiments.run_hydra method=eo_penalty dataset=adult
    python -m asbr.experiments.run_hydra method=eo_penalty method.lambd=10.0 dataset=adult
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asbr.data.intersections import build_subgroup_masks, masks_to_torch
from asbr.evaluation.evaluator import compute_predictions
from asbr.experiments.config import ASBRConfig, TrainingConfig
from asbr.fairness.asbr_lrt_loss import ASBRLRTLoss
from asbr.methods.base import TrainedMethod
from asbr.models.base_mlp import FairnessMLP
from asbr.training.trainer import JointTrainer, Trainer


class SPPenaltyMethod(TrainedMethod):
    """SP squared penalty baseline (ASBR with all ablation flags off)."""

    def __init__(self, name: str = "SP-Penalty", lambd: float = 1.0):
        self.name = name
        self._lambd = lambd

    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_vl: np.ndarray,
        y_vl: np.ndarray,
        A_tr: pd.DataFrame,
        intersection_level: int,
        training_config: TrainingConfig,
    ) -> None:
        masks_tr = build_subgroup_masks(A_tr, level=intersection_level, min_size=2)
        masks_tr_torch = masks_to_torch(masks_tr)

        self._model = FairnessMLP(
            input_dim=X_tr.shape[1],
            hidden_dims=training_config.hidden_dims,
            dropout=training_config.dropout,
        )
        sp_config = ASBRConfig.sp_penalty_preset()
        sp_config.lambd = self._lambd

        trainer = JointTrainer(
            model=self._model,
            asbr_config=sp_config,
            training_config=training_config,
        )
        trainer.train(X_tr, y_tr, masks_tr_torch, X_vl, y_vl)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        return compute_predictions(self._model, X, threshold=threshold)


class EOPenaltyMethod(TrainedMethod):
    """EO squared penalty baseline (BCE + λ·Σ_S EO²(S), no significance gate).

    The EO-Penalty is the Equal Opportunity analogue of SP-Penalty.
    It uses the LRT loss path (required for EO support) with use_sp_magnitude=True
    and use_hinge=False, which routes to the ``metric_tensor**2`` branch —
    a genuine EO² penalty per subgroup.  No significance hinge, no hybrid
    gradient, no size weight.
    Subgroups with no positive ground-truth labels on either side are silently skipped.
    """

    def __init__(
        self,
        name: str = "EO-Penalty",
        lambd: float | None = None,
        asbr: dict | None = None,
        trainer: str = "joint",
    ):
        self.name = name
        p = asbr or {}
        # lambd can be passed either as a top-level kwarg (from the YAML sweep field)
        # or nested inside the asbr dict; top-level takes precedence.
        self._lambd = float(lambd if lambd is not None else p.get("lambd", 100.0))
        self._trainer_type = trainer
        self._use_size_weight = bool(p.get("use_size_weight", False))

    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_vl: np.ndarray,
        y_vl: np.ndarray,
        A_tr: pd.DataFrame,
        intersection_level: int,
        training_config: TrainingConfig,
    ) -> None:
        masks_tr = build_subgroup_masks(A_tr, level=intersection_level, min_size=2)
        masks_tr_torch = masks_to_torch(masks_tr)

        self._model = FairnessMLP(
            input_dim=X_tr.shape[1],
            hidden_dims=training_config.hidden_dims,
            dropout=training_config.dropout,
        )
        eo_config = ASBRConfig.eo_penalty_preset()
        eo_config.lambd = self._lambd
        eo_config.use_size_weight = self._use_size_weight

        if self._trainer_type == "two_phase":
            trainer_obj = Trainer(
                model=self._model,
                asbr_config=eo_config,
                training_config=training_config,
                loss_class=ASBRLRTLoss,
            )
        else:
            trainer_obj = JointTrainer(
                model=self._model,
                asbr_config=eo_config,
                training_config=training_config,
                loss_class=ASBRLRTLoss,
            )
        trainer_obj.train(X_tr, y_tr, masks_tr_torch, X_vl, y_vl)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        return compute_predictions(self._model, X, threshold=threshold)
