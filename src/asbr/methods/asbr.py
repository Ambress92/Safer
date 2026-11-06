"""ASBR method (Wald/LRT, two-phase/joint)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from asbr.data.intersections import build_subgroup_masks, masks_to_torch
from asbr.evaluation.evaluator import compute_predictions
from asbr.experiments.config import ASBRConfig, TrainingConfig
from asbr.fairness.asbr_lrt_loss import ASBRLRTLoss
from asbr.methods.base import TrainedMethod
from asbr.models.base_mlp import FairnessMLP
from asbr.training.trainer import (
    JointTrainer,
    LambdaCheckpoint,
    ProgressiveTrainer,
    TargetProgressiveTrainer,
    Trainer,
    _compute_msee_hard,
)


class ASBRMethod(TrainedMethod):
    """ASBR fairness method (Wald or LRT, two-phase or joint training).

    Args:
        name: Method display name.
        trainer: ``"two_phase"`` or ``"joint"``.
        loss_type: ``"wald"`` (default ASBRLoss) or ``"lrt"`` (ASBRLRTLoss).
        asbr: Dict/DictConfig with ASBR hyperparameters (alpha, lambd, etc.).
    """

    def __init__(
        self,
        name: str,
        trainer: str = "two_phase",
        loss_type: str = "wald",
        asbr: dict | DictConfig | None = None,
        lambd: float | None = None,  # top-level mirror of asbr.lambd; passed by Hydra, unused here
        bce_weight: float = 0.0,
    ):
        self.name = name
        self._trainer_type = trainer
        self._loss_type = loss_type
        self._asbr_params = asbr or {}
        self._bce_weight = bce_weight
        self._train_msee: float | None = None
        self._lambd: float | None = None

    def _build_asbr_config(self) -> ASBRConfig:
        p = self._asbr_params
        return ASBRConfig(
            alpha=p.get("alpha", 0.05),
            lambd=p.get("lambd", 1.0),
            n_tilde=p.get("n_tilde", 30),
            kappa=p.get("kappa", 0.5),
            epsilon=p.get("epsilon", 1e-8),
            fairness_metric=p.get("fairness_metric", "sp"),
            use_hinge=p.get("use_hinge", True),
            use_hybrid=p.get("use_hybrid", True),
            use_size_weight=p.get("use_size_weight", True),
            use_sp_magnitude=p.get("use_sp_magnitude", False),
            gate_alpha=p.get("gate_alpha", None),
            hinge_alpha=p.get("hinge_alpha", None),
            use_squared_hinge=p.get("use_squared_hinge", True),
            dense_tail_weight=p.get("dense_tail_weight", 0.0),
            dense_tail_shape=p.get("dense_tail_shape", "lrt"),
            loss_shape=p.get("loss_shape", "lrt_hinge"),
            direct_tau=p.get("direct_tau", 0.01),
            direct_abs_eps=p.get("direct_abs_eps", 1e-3),
            direct_detach_se=p.get("direct_detach_se", True),
            direct_pred_temperature=p.get("direct_pred_temperature", None),
            direct_threshold=p.get("direct_threshold", 0.5),
            direct_lrt_detach=p.get("direct_lrt_detach", True),
            direct_lrt_sqrt_min=p.get("direct_lrt_sqrt_min", 1e-4),
            direct_lrt_se_min=p.get("direct_lrt_se_min", 0.0),
            direct_lrt_se_max=p.get("direct_lrt_se_max", 1.0),
            use_rms_aggregation=p.get("use_rms_aggregation", False),
            use_overall_reference=p.get("use_overall_reference", False),
        )

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
        asbr_config = self._build_asbr_config()
        loss_class = ASBRLRTLoss if self._loss_type == "lrt" else None

        trainer_cls = JointTrainer if self._trainer_type == "joint" else Trainer

        kwargs: dict = dict(
            model=self._model,
            asbr_config=asbr_config,
            training_config=training_config,
            bce_weight=self._bce_weight,
        )
        if loss_class is not None:
            kwargs["loss_class"] = loss_class

        trainer = trainer_cls(**kwargs)
        trainer.train(X_tr, y_tr, masks_tr_torch, X_vl, y_vl)
        self._lambd = asbr_config.lambd
        self._train_msee = _compute_msee_hard(
            self._model,
            torch.tensor(X_tr, dtype=torch.float32, device=next(self._model.parameters()).device),
            y_tr.astype(float),
            masks_tr_torch,
            fairness_metric=asbr_config.fairness_metric,
            alpha=asbr_config.alpha,
            threshold=asbr_config.direct_threshold,
        )

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        return compute_predictions(self._model, X, threshold=threshold)

    def get_checkpoint_meta(self, lambd: float) -> dict:
        """Return train-side MSEE metadata for the fixed-λ run."""
        if self._lambd is None or not np.isclose(float(lambd), float(self._lambd)):
            return {}
        return {
            "train_msee": self._train_msee,
            "stop_reason": None,
            "selected": True,
        }


class ProgressiveASBRMethod(TrainedMethod):
    """ASBR with progressive λ training: sweeps the full Pareto frontier in one run.

    Phase 1 trains BCE to convergence (identical to the two-phase trainer).
    Phase 2 iterates over ``lambda_schedule`` in increasing order, training
    ``BCE + λ_k · fairness`` at each step with warm-starting from the previous
    checkpoint. One model snapshot is produced per λ value.

    After training, ``predict_at(lambd, X)`` returns predictions from the
    snapshot at the requested λ. The experiment runner iterates over all
    snapshots to produce one metrics row per λ per (split, seed).

    Args:
        name: Display name (used in results CSV and plot legend).
        lambda_schedule: Increasing list of λ values to sweep.
        loss_type: ``"lrt"`` (default) or ``"wald"``.
        asbr: Dict/DictConfig with ASBR hyperparameters (alpha, n_tilde, etc.).
            ``lambd`` is ignored here — the schedule overrides it.
    """

    def __init__(
        self,
        name: str,
        lambda_schedule: list[float],
        loss_type: str = "lrt",
        bce_weight: float = 1e-6,
        checkpoint_mode: str = "combined",
        bce_guard_rel_tol: float = 0.0,
        reset_to_phase1: bool = False,
        asbr: dict | None = None,
    ):
        self.name = name
        self._lambda_schedule = list(lambda_schedule)
        self._loss_type = loss_type
        self._bce_weight = bce_weight
        self._checkpoint_mode = checkpoint_mode
        self._bce_guard_rel_tol = bce_guard_rel_tol
        self._reset_to_phase1 = reset_to_phase1
        self._asbr_params = asbr or {}
        self._checkpoints: list[LambdaCheckpoint] = []
        self._model: FairnessMLP | None = None

    def _build_base_config(self) -> ASBRConfig:
        p = self._asbr_params
        return ASBRConfig(
            alpha=p.get("alpha", 0.05),
            lambd=self._lambda_schedule[0],  # placeholder; overridden per step
            n_tilde=p.get("n_tilde", 30),
            kappa=p.get("kappa", 0.5),
            epsilon=p.get("epsilon", 1e-8),
            fairness_metric=p.get("fairness_metric", "sp"),
            use_hinge=p.get("use_hinge", True),
            use_hybrid=p.get("use_hybrid", False),
            use_size_weight=p.get("use_size_weight", False),
            use_sp_magnitude=p.get("use_sp_magnitude", True),
            gate_alpha=p.get("gate_alpha", None),
            hinge_alpha=p.get("hinge_alpha", None),
            use_squared_hinge=p.get("use_squared_hinge", True),
            dense_tail_weight=p.get("dense_tail_weight", 0.0),
            dense_tail_shape=p.get("dense_tail_shape", "lrt"),
            loss_shape=p.get("loss_shape", "lrt_hinge"),
            direct_tau=p.get("direct_tau", 0.01),
            direct_abs_eps=p.get("direct_abs_eps", 1e-3),
            direct_detach_se=p.get("direct_detach_se", True),
            direct_pred_temperature=p.get("direct_pred_temperature", None),
            direct_threshold=p.get("direct_threshold", 0.5),
            direct_lrt_detach=p.get("direct_lrt_detach", True),
            direct_lrt_sqrt_min=p.get("direct_lrt_sqrt_min", 1e-4),
            direct_lrt_se_min=p.get("direct_lrt_se_min", 0.0),
            direct_lrt_se_max=p.get("direct_lrt_se_max", 1.0),
            use_rms_aggregation=p.get("use_rms_aggregation", False),
            use_overall_reference=p.get("use_overall_reference", False),
        )

    @property
    def lambda_schedule(self) -> list[float]:
        """λ values for which snapshots are available after training."""
        return [c.lambd for c in self._checkpoints]

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
        from asbr.data.intersections import build_subgroup_masks, masks_to_torch
        masks_tr = build_subgroup_masks(A_tr, level=intersection_level, min_size=2)
        masks_tr_torch = masks_to_torch(masks_tr)

        self._model = FairnessMLP(
            input_dim=X_tr.shape[1],
            hidden_dims=training_config.hidden_dims,
            dropout=training_config.dropout,
        )
        base_config = self._build_base_config()
        loss_class = ASBRLRTLoss if self._loss_type == "lrt" else None

        trainer = ProgressiveTrainer(
            model=self._model,
            base_asbr_config=base_config,
            training_config=training_config,
            bce_weight=self._bce_weight,
            checkpoint_mode=self._checkpoint_mode,
            bce_guard_rel_tol=self._bce_guard_rel_tol,
            reset_to_phase1=self._reset_to_phase1,
            **({"loss_class": loss_class} if loss_class is not None else {}),
        )
        _, self._checkpoints = trainer.train(
            X_tr, y_tr, masks_tr_torch, X_vl, y_vl,
            lambda_schedule=self._lambda_schedule,
        )
        # Leave model at the last checkpoint's state (highest λ).

    def predict_at(
        self, lambd: float, X: np.ndarray, threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return predictions from the snapshot trained at ``lambd``."""
        for ckpt in self._checkpoints:
            if ckpt.lambd == lambd:
                self._model.load_state_dict(ckpt.state_dict)
                return compute_predictions(self._model, X, threshold=threshold)
        raise ValueError(f"No checkpoint for lambd={lambd}. Available: {self.lambda_schedule}")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Predict using the last (highest-λ) snapshot."""
        return compute_predictions(self._model, X, threshold=threshold)

    def get_checkpoint_meta(self, lambd: float) -> dict:
        """Return per-checkpoint metadata for ``lambd``."""
        for ckpt in self._checkpoints:
            if ckpt.lambd == lambd:
                return {
                    "train_msee": ckpt.train_msee,
                    "stop_reason": ckpt.stop_reason,
                    "selected": ckpt.selected,
                }
        return {}


class TargetProgressiveASBRMethod(TrainedMethod):
    """ASBR with target-based progressive λ training.

    Instead of specifying a fixed ``lambda_schedule``, the caller specifies a
    desired fairness reduction and the trainer searches for the minimum λ that
    achieves it. Exposes the same ``lambda_schedule`` / ``predict_at`` interface
    as :class:`ProgressiveASBRMethod` so the experiment runner can iterate over
    all checkpoints and produce one metrics row per step.

    Args:
        name: Display name.
        target_msee: Absolute MSEE/MSPE target. Mutually exclusive with
            ``target_reduction``.
        target_reduction: Fraction of ERM MSEE to eliminate (0–1). Used when
            ``target_msee`` is None.
        lambda_min: Starting λ (default 0.1).
        lambda_max: Maximum λ ceiling (default 500.0).
        lambda_multiplier: Geometric step factor > 1 (default 1.5).
        msee_patience: Steps without MSEE improvement before declaring plateau
            (default 2).
        loss_type: ``"lrt"`` (default) or ``"wald"``.
        bce_weight: BCE anchor weight in Phase 2.
        asbr: Dict/DictConfig with ASBR hyperparameters.
    """

    def __init__(
        self,
        name: str,
        target_msee: float | None = None,
        target_reduction: float | None = None,
        lambda_min: float = 0.1,
        lambda_max: float = 500.0,
        lambda_multiplier: float = 1.5,
        msee_patience: int = 2,
        loss_type: str = "lrt",
        bce_weight: float = 1e-6,
        asbr: dict | None = None,
    ):
        self.name = name
        self._target_msee = target_msee
        self._target_reduction = target_reduction
        self._lambda_min = lambda_min
        self._lambda_max = lambda_max
        self._lambda_multiplier = lambda_multiplier
        self._msee_patience = msee_patience
        self._loss_type = loss_type
        self._bce_weight = bce_weight
        self._asbr_params = asbr or {}
        self._checkpoints: list[LambdaCheckpoint] = []
        self._selected_idx: int = 0
        self._model: FairnessMLP | None = None

    def _build_base_config(self) -> ASBRConfig:
        p = self._asbr_params
        return ASBRConfig(
            alpha=p.get("alpha", 0.05),
            lambd=self._lambda_min,  # placeholder; overridden per step
            n_tilde=p.get("n_tilde", 30),
            kappa=p.get("kappa", 0.5),
            epsilon=p.get("epsilon", 1e-8),
            fairness_metric=p.get("fairness_metric", "eo"),
            use_hinge=p.get("use_hinge", True),
            use_hybrid=p.get("use_hybrid", False),
            use_size_weight=p.get("use_size_weight", False),
            use_sp_magnitude=p.get("use_sp_magnitude", True),
            gate_alpha=p.get("gate_alpha", None),
            hinge_alpha=p.get("hinge_alpha", None),
            use_squared_hinge=p.get("use_squared_hinge", True),
            dense_tail_weight=p.get("dense_tail_weight", 0.0),
            dense_tail_shape=p.get("dense_tail_shape", "lrt"),
            loss_shape=p.get("loss_shape", "lrt_hinge"),
            direct_tau=p.get("direct_tau", 0.01),
            direct_abs_eps=p.get("direct_abs_eps", 1e-3),
            direct_detach_se=p.get("direct_detach_se", True),
            direct_pred_temperature=p.get("direct_pred_temperature", None),
            direct_threshold=p.get("direct_threshold", 0.5),
            direct_lrt_detach=p.get("direct_lrt_detach", True),
            direct_lrt_sqrt_min=p.get("direct_lrt_sqrt_min", 1e-4),
            direct_lrt_se_min=p.get("direct_lrt_se_min", 0.0),
            direct_lrt_se_max=p.get("direct_lrt_se_max", 1.0),
            use_rms_aggregation=p.get("use_rms_aggregation", False),
            use_overall_reference=p.get("use_overall_reference", False),
        )

    @property
    def lambda_schedule(self) -> list[float]:
        """λ values for which snapshots are available after training."""
        return [c.lambd for c in self._checkpoints]

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
        from asbr.data.intersections import build_subgroup_masks, masks_to_torch
        masks_tr = build_subgroup_masks(A_tr, level=intersection_level, min_size=2)
        masks_tr_torch = masks_to_torch(masks_tr)

        self._model = FairnessMLP(
            input_dim=X_tr.shape[1],
            hidden_dims=training_config.hidden_dims,
            dropout=training_config.dropout,
        )
        base_config = self._build_base_config()
        loss_class = ASBRLRTLoss if self._loss_type == "lrt" else None

        trainer = TargetProgressiveTrainer(
            model=self._model,
            base_asbr_config=base_config,
            training_config=training_config,
            bce_weight=self._bce_weight,
            target_msee=self._target_msee,
            target_reduction=self._target_reduction,
            lambda_min=self._lambda_min,
            lambda_max=self._lambda_max,
            lambda_multiplier=self._lambda_multiplier,
            msee_patience=self._msee_patience,
            **({"loss_class": loss_class} if loss_class is not None else {}),
        )
        _, self._checkpoints, self._selected_idx = trainer.train(
            X_tr, y_tr, masks_tr_torch, X_vl, y_vl,
        )

    def predict_at(
        self, lambd: float, X: np.ndarray, threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return predictions from the snapshot trained at ``lambd``."""
        for ckpt in self._checkpoints:
            if ckpt.lambd == lambd:
                self._model.load_state_dict(ckpt.state_dict)
                return compute_predictions(self._model, X, threshold=threshold)
        raise ValueError(f"No checkpoint for lambd={lambd}. Available: {self.lambda_schedule}")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Predict using the selected (target-reaching or best) checkpoint."""
        self._model.load_state_dict(self._checkpoints[self._selected_idx].state_dict)
        return compute_predictions(self._model, X, threshold=threshold)

    def get_checkpoint_meta(self, lambd: float) -> dict:
        """Return ``train_msee``, ``stop_reason``, and ``selected`` for ``lambd``."""
        for ckpt in self._checkpoints:
            if ckpt.lambd == lambd:
                return {
                    "train_msee": ckpt.train_msee,
                    "stop_reason": ckpt.stop_reason,
                    "selected": ckpt.selected,
                }
        return {}
