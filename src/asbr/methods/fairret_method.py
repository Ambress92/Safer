"""fairret NormLoss(PositiveRate()/TruePositiveRate(), p=1 and p=2) in-processing baselines.

Bridges the fairret library (Buyl et al., ICLR 2024) into the JointTrainer
interface so it can be evaluated under the same training protocol as SP-Penalty
and ASBR.

Penalty formulas (SP variant, fairness_metric="sp"):
    p=1 (norm1): sum_S | p_hat_S / p_hat_overall - 1 |
    p=2 (norm2): sqrt( sum_S ( p_hat_S / p_hat_overall - 1 )^2 )

Penalty formulas (EO variant, fairness_metric="eo"):
    Same norms but over TruePositiveRate (TPR) instead of PositiveRate.
    Only samples with y=1 contribute; y_true must be passed.

Reference:
    Buyl, Defrance & De Bie. "fairret: a Framework for Differentiable
    Fairness Regularization Terms." ICLR 2024.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from asbr.data.intersections import build_subgroup_masks, masks_to_torch
from asbr.evaluation.evaluator import compute_predictions
from asbr.experiments.config import ASBRConfig, TrainingConfig
from asbr.methods.base import TrainedMethod
from asbr.models.base_mlp import FairnessMLP
from asbr.training.trainer import JointTrainer


def _build_fairret_loss(p: int, fairness_metric: str):
    """Instantiate a fairret NormLoss for the given norm order and fairness metric.

    Args:
        p: Norm order (1 or 2).
        fairness_metric: ``"sp"`` for PositiveRate (Statistical Parity) or
            ``"eo"`` for TruePositiveRate (Equal Opportunity).

    Returns:
        A fairret NormLoss instance.
    """
    from fairret.loss.violation import NormLoss
    from fairret.statistic.linear_fractional import PositiveRate, TruePositiveRate

    if fairness_metric == "eo":
        return NormLoss(TruePositiveRate(), p=p)
    return NormLoss(PositiveRate(), p=p)


class _FairretNormLoss:
    """Shared adapter wrapping fairret NormLoss for JointTrainer.

    JointTrainer calls ``compute_fairness_loss(y_pred, subgroup_masks, y_true)``
    and expects ``(loss_scalar, diagnostics_dict)``.  This class converts the
    masks dict to the ``(N, S)`` sens tensor that fairret requires.

    The sens tensor is built once on the first call and cached; subsequent
    epochs reuse it (same dataset, same subgroup structure).

    Args:
        cfg: ASBRConfig — only ``cfg.lambd`` is used (externally, by JointTrainer).
        p: Norm order (1 or 2).
        fairness_metric: ``"sp"`` (PositiveRate) or ``"eo"`` (TruePositiveRate).
    """

    def __init__(self, cfg: ASBRConfig, p: int, fairness_metric: str) -> None:
        self._loss_fn = _build_fairret_loss(p, fairness_metric)
        self._fairness_metric = fairness_metric
        self._sens: torch.Tensor | None = None

    def compute_fairness_loss(
        self,
        y_pred: torch.Tensor,
        subgroup_masks: dict[str, torch.Tensor],
        y_true: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        # Build and cache (N, S) float indicator matrix
        if self._sens is None or self._sens.shape[0] != y_pred.shape[0]:
            self._sens = torch.stack(
                [m.float() for m in subgroup_masks.values()], dim=1
            ).to(y_pred.device)

        # fairret expects pred shape (N, 1); pred is already probabilities, not logits
        kwargs: dict = {"pred_as_logit": False}
        if self._fairness_metric == "eo":
            if y_true is None:
                raise RuntimeError(
                    "fairness_metric='eo' requires y_true to be passed, but got None."
                )
            # TruePositiveRate needs labels shape (N, 1)
            kwargs["label"] = y_true.unsqueeze(-1)

        loss = self._loss_fn(y_pred.unsqueeze(-1), self._sens, **kwargs)
        return loss, {}


class FairretNorm2Loss(_FairretNormLoss):
    """Adapter wrapping fairret NormLoss(PositiveRate(), p=2) for JointTrainer (SP)."""

    def __init__(self, cfg: ASBRConfig) -> None:
        super().__init__(cfg, p=2, fairness_metric="sp")


class FairretNorm1Loss(_FairretNormLoss):
    """Adapter wrapping fairret NormLoss(PositiveRate(), p=1) for JointTrainer (SP)."""

    def __init__(self, cfg: ASBRConfig) -> None:
        super().__init__(cfg, p=1, fairness_metric="sp")


class _FairretNormMethod(TrainedMethod):
    """Shared base for fairret NormLoss methods.

    Args:
        name: Method name shown in result CSVs.
        lambd: Fairness regularization weight.
        p: Norm order (1 or 2).
        fairness_metric: ``"sp"`` (PositiveRate) or ``"eo"`` (TruePositiveRate).
    """

    def __init__(
        self,
        name: str,
        lambd: float,
        p: int,
        fairness_metric: str,
    ) -> None:
        self.name = name
        self._lambd = lambd
        self._p = p
        self._fairness_metric = fairness_metric

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
        cfg = ASBRConfig()
        cfg.lambd = self._lambd

        p = self._p
        fm = self._fairness_metric

        class _Loss(_FairretNormLoss):
            def __init__(self, cfg: ASBRConfig) -> None:
                super().__init__(cfg, p=p, fairness_metric=fm)

        JointTrainer(
            model=self._model,
            asbr_config=cfg,
            training_config=training_config,
            loss_class=_Loss,
        ).train(X_tr, y_tr, masks_tr_torch, X_vl, y_vl)

    def predict(
        self, X: np.ndarray, threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        return compute_predictions(self._model, X, threshold=threshold)


class FairretNorm2Method(_FairretNormMethod):
    """fairret NormLoss(p=2) — SP or EO variant.

    Penalty formula (SP):  sqrt( sum_S ( p_hat_S / p_hat_overall - 1 )^2 )
    Penalty formula (EO):  sqrt( sum_S ( tpr_hat_S / tpr_hat_overall - 1 )^2 )

    Args:
        name: Method name shown in result CSVs.
        lambd: Fairness regularization weight.
        fairness_metric: ``"sp"`` (default) or ``"eo"``.
    """

    def __init__(
        self,
        name: str = "fairret-norm2",
        lambd: float = 1.0,
        fairness_metric: str = "sp",
    ) -> None:
        super().__init__(name=name, lambd=lambd, p=2, fairness_metric=fairness_metric)


class FairretNorm1Method(_FairretNormMethod):
    """fairret NormLoss(p=1) — SP or EO variant.

    Penalty formula (SP):  sum_S | p_hat_S / p_hat_overall - 1 |
    Penalty formula (EO):  sum_S | tpr_hat_S / tpr_hat_overall - 1 |

    Args:
        name: Method name shown in result CSVs.
        lambd: Fairness regularization weight.
        fairness_metric: ``"sp"`` (default) or ``"eo"``.
    """

    def __init__(
        self,
        name: str = "fairret-norm1",
        lambd: float = 1.0,
        fairness_metric: str = "sp",
    ) -> None:
        super().__init__(name=name, lambd=lambd, p=1, fairness_metric=fairness_metric)
