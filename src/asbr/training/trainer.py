"""Training loop with full-dataset fairness computation and early stopping.

Key design decision: fairness loss is computed on the FULL training set
(not mini-batches) because the Wald test requires stable count estimates.
Prediction loss uses mini-batch SGD as usual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _default_device() -> torch.device:
    """Return CUDA if available, MPS (Apple Silicon) if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

from asbr.experiments.config import ASBRConfig, TrainingConfig
from asbr.fairness.asbr_loss import ASBRLoss
from asbr.models.base_mlp import FairnessMLP


@dataclass
class EpochLog:
    """Metrics recorded at the end of each epoch."""

    epoch: int
    train_pred_loss: float
    train_fair_loss: float
    train_total_loss: float
    val_pred_loss: float
    diagnostics: dict[str, SubgroupDiagnostics] = field(default_factory=dict)


class Trainer:
    """Training loop for ASBR and ablation variants.

    Training strategy:
    1. Mini-batch SGD for prediction loss (BCE)
    2. Full-dataset forward pass for fairness loss (stable Wald statistics)
    3. Combined backward pass each epoch
    4. Early stopping on validation prediction loss only

    Args:
        model: FairnessMLP instance.
        asbr_config: ASBRConfig for the fairness loss.
        training_config: TrainingConfig for the training loop.
        device: Torch device.
        loss_class: Fairness loss class to use. Defaults to ASBRLoss.
            Must accept ASBRConfig and implement compute_fairness_loss().
        bce_weight: BCE anchor weight in Phase 2. When > 0, Phase 2 optimizes
            ``bce_weight * BCE + lambda * fairness`` instead of pure fairness.
            Useful for datasets where pure fairness training causes model collapse.
    """

    def __init__(
        self,
        model: FairnessMLP,
        asbr_config: ASBRConfig | None = None,
        training_config: TrainingConfig | None = None,
        device: torch.device | str | None = None,
        loss_class: type | None = None,
        bce_weight: float = 0.0,
    ):
        device = _default_device() if device is None else torch.device(device)
        self.model = model.to(device)
        self.asbr_config = asbr_config or ASBRConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = device

        loss_cls = loss_class or ASBRLoss
        self.asbr_loss = loss_cls(self.asbr_config)
        self.ce_loss = nn.BCELoss()
        self.bce_weight = bce_weight
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay,
        )
        # SGD for Phase 2 fairness updates.  Adam normalises each
        # gradient by its running second moment, making the effective
        # step ≈ lr regardless of gradient magnitude.  This renders
        # lambda and max_grad_norm irrelevant.  SGD preserves the
        # gradient scale so lambda directly controls fairness strength.
        # No weight_decay on SGD: L2 regularization is already applied
        # in Phase 1 and would distort the calibrated fairness gradient scale.
        self.fair_optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.training_config.lr
        )

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.training_config.batch_size, shuffle=shuffle)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        subgroup_masks_train: dict[str, torch.Tensor],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[EpochLog]:
        """Run the sequential two-phase training loop.

        Phase 1: Mini-batch BCE training until early stopping triggers.
        Phase 2: Full-dataset fairness-only training (no BCE updates) with
        early stopping on combined validation loss.

        Args:
            X_train: Training features, shape (n_train, d).
            y_train: Training labels, shape (n_train,).
            subgroup_masks_train: Torch boolean masks for fairness on train set.
            X_val: Validation features, shape (n_val, d).
            y_val: Validation labels, shape (n_val,).

        Returns:
            history: List of EpochLog for each epoch.
        """
        cfg = self.training_config

        X_train_full = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_full = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        y_train_cpu = torch.tensor(y_train, dtype=torch.float32)  # CPU copy for fairness loss
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        # Keep masks on CPU — fairness computation runs on CPU to avoid per-subgroup
        # GPU kernel-launch overhead (176 tiny ops >> GPU kernel overhead for a small MLP).

        train_loader = self._make_loader(X_train, y_train, shuffle=True)

        history: list[EpochLog] = []

        # ===== Phase 1: BCE training until convergence =====
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(cfg.max_epochs):
            self.model.train()
            epoch_pred_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                y_pred_batch = self.model(X_batch)
                batch_loss = self.ce_loss(y_pred_batch, y_batch)
                batch_loss.backward()
                self.optimizer.step()
                epoch_pred_loss += batch_loss.item()
                n_batches += 1

            avg_pred_loss = epoch_pred_loss / max(n_batches, 1)

            self.model.eval()
            with torch.no_grad():
                y_pred_val = self.model(X_val_t)
                val_loss = self.ce_loss(y_pred_val, y_val_t).item()

            history.append(EpochLog(
                epoch=epoch,
                train_pred_loss=avg_pred_loss,
                train_fair_loss=0.0,
                train_total_loss=avg_pred_loss,
                val_pred_loss=val_loss,
            ))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                break

        # Restore best Phase 1 model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        phase1_epochs = len(history)
        print(f"  [P1] α={self.asbr_config.alpha} λ={self.asbr_config.lambd} | val_bce={best_val_loss:.4f} ({phase1_epochs} epochs)")

        # ===== Phase 2: Fairness-only training =====
        # Collapse guard: if val BCE exceeds Phase 1 optimum by >50%, the SGD
        # fairness step collapsed the model to a constant predictor.  Abort Phase 2
        # and return the Phase 1 checkpoint.  This can happen on large datasets
        # (e.g. CelebA) where the fairness gradient is very large.
        phase1_best_bce = best_val_loss
        phase1_best_state = best_state
        collapse_threshold = phase1_best_bce * 1.5

        # Normalization factor: scale fairness loss so that λ=1 means equal weight to BCE.
        # Compute fair_loss at Phase 1 optimum (one forward pass, no grad).
        if cfg.normalize_phase2:
            self.model.eval()
            with torch.no_grad():
                y_pred_init = self.model(X_train_full)
                fair_loss_init_cpu, _ = self.asbr_loss.compute_fairness_loss(
                    y_pred_init.cpu(), subgroup_masks_train, y_true=y_train_cpu
                )
            fair_loss_initial = fair_loss_init_cpu.item()
            norm_factor = phase1_best_bce / max(fair_loss_initial, 1e-8)
            print(f"  [P1→P2] bce={phase1_best_bce:.4f}  fair_init={fair_loss_initial:.2f}  norm_factor={norm_factor:.2e}")
        else:
            norm_factor = 1.0

        best_combined = float("inf")
        best_state = phase1_best_state  # default: Phase 1 if Phase 2 collapses
        patience_counter = 0
        phase2_epochs = 0
        phase2_stop_reason = "max_epochs"

        for epoch in range(phase1_epochs, cfg.max_epochs):
            self.model.train()
            self.fair_optimizer.zero_grad()
            y_pred_full = self.model(X_train_full)
            # Fairness runs on CPU: move predictions off GPU to avoid 176× kernel-launch overhead.
            # Autograd traces through .cpu() back to GPU model weights correctly.
            fair_loss_cpu, diagnostics = self.asbr_loss.compute_fairness_loss(
                y_pred_full.cpu(), subgroup_masks_train, y_true=y_train_cpu
            )
            fair_loss = fair_loss_cpu.to(self.device)
            # Normalize fairness term: λ=1 → fairness weighted equally to Phase 1 BCE.
            fairness_term = self.asbr_config.lambd * norm_factor * fair_loss
            if self.bce_weight > 0:
                pred_loss_full = self.ce_loss(y_pred_full, y_train_full)
                total_loss = self.bce_weight * pred_loss_full + fairness_term
            else:
                total_loss = fairness_term

            if total_loss.requires_grad:
                total_loss.backward()
                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.max_grad_norm
                    )
                self.fair_optimizer.step()

            fair_loss_val = fair_loss.item()

            self.model.eval()
            with torch.no_grad():
                y_pred_val = self.model(X_val_t)
                val_loss = self.ce_loss(y_pred_val, y_val_t).item()


            history.append(EpochLog(
                epoch=epoch,
                train_pred_loss=0.0,
                train_fair_loss=fair_loss_val,
                train_total_loss=self.asbr_config.lambd * fair_loss_val,
                val_pred_loss=val_loss,
                diagnostics=diagnostics,
            ))
            phase2_epochs += 1

            # Abort if model collapsed (val BCE ≫ Phase 1 optimum).
            if val_loss > collapse_threshold:
                print(f"  [COLLAPSE] epoch {epoch} | val_bce={val_loss:.4f} > thr={collapse_threshold:.4f}")
                phase2_stop_reason = "collapse"
                break

            combined_val = val_loss + self.asbr_config.lambd * norm_factor * fair_loss_val
            if combined_val < best_combined:
                best_combined = combined_val
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                phase2_stop_reason = "patience"
                break

        print(
            f"  [P2] α={self.asbr_config.alpha} λ={self.asbr_config.lambd} "
            f"| best_combined={best_combined:.4f} ({phase2_epochs} epochs, {phase2_stop_reason})"
        )

        # Restore best Phase 2 model (or Phase 1 if Phase 2 collapsed/didn't improve)
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history


class JointTrainer:
    """Single-phase training with combined BCE + fairness loss on the full dataset.

    Unlike the two-phase :class:`Trainer`, this computes **both** losses in a
    single forward pass over the full training set and applies one combined
    gradient step per epoch.  Early stopping monitors the combined training
    loss (BCE + lambd * fairness), so the fairness penalty directly influences
    model selection.

    Args:
        model: FairnessMLP instance.
        asbr_config: ASBRConfig for the fairness loss.
        training_config: TrainingConfig for the training loop.
        device: Torch device.
        loss_class: Fairness loss class to use. Defaults to ASBRLoss.
    """

    def __init__(
        self,
        model: FairnessMLP,
        asbr_config: ASBRConfig | None = None,
        training_config: TrainingConfig | None = None,
        device: torch.device | str | None = None,
        loss_class: type | None = None,
        bce_weight: float = 0.0,  # ignored for joint training, accepted for API compatibility
    ):
        device = _default_device() if device is None else torch.device(device)
        self.model = model.to(device)
        self.asbr_config = asbr_config or ASBRConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = device

        loss_cls = loss_class or ASBRLoss
        self.asbr_loss = loss_cls(self.asbr_config)
        self.ce_loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay,
        )

    @staticmethod
    def _fairness_ramp_factor(joint_epoch: int, ramp_epochs: int, shape: str) -> float:
        """Return the multiplicative fairness-weight ramp for a joint epoch."""
        if ramp_epochs <= 0:
            return 1.0
        progress = min(1.0, max(0.0, (joint_epoch + 1) / ramp_epochs))
        if shape == "linear":
            return progress
        if shape == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * progress)
        raise ValueError(
            "training.fairness_ramp_shape must be 'linear' or 'cosine', "
            f"got {shape!r}"
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        subgroup_masks_train: dict[str, torch.Tensor],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[EpochLog]:
        """Run the joint training loop.

        Each epoch performs a single full-dataset forward pass, computes
        BCE + lambd * fairness_loss, and does one backward/step.

        Convergence is managed by a ReduceLROnPlateau scheduler monitoring
        ``val_BCE + lambd * train_fairness``.  Training stops as soon as the
        learning rate reaches ``min_lr`` (meaning no more useful reductions
        are possible).  The best checkpoint is restored at the end.

        Monitoring metric rationale: val BCE is an unbiased estimator of test
        prediction loss; train fairness is a well-powered estimator of the
        model's fairness behaviour (val subgroups are too small — Z-scores
        ~0.33× of train — to carry a reliable fairness signal).
        """
        cfg = self.training_config
        if cfg.joint_checkpoint_mode not in ("val_bce", "bce_guarded_fairness"):
            raise ValueError(
                "training.joint_checkpoint_mode must be 'val_bce' or "
                f"'bce_guarded_fairness', got {cfg.joint_checkpoint_mode!r}"
            )
        if cfg.fairness_ramp_shape not in ("linear", "cosine"):
            raise ValueError(
                "training.fairness_ramp_shape must be 'linear' or 'cosine', "
                f"got {cfg.fairness_ramp_shape!r}"
            )

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        y_train_cpu = torch.tensor(y_train, dtype=torch.float32)  # CPU copy for fairness loss
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        # Keep masks on CPU — fairness computation runs on CPU to avoid per-subgroup
        # GPU kernel-launch overhead (176 tiny ops >> GPU kernel overhead for a small MLP).

        history: list[EpochLog] = []
        best_val_loss = float("inf")
        best_state = None

        warmup_epochs = max(0, min(cfg.bce_warmup_epochs, cfg.max_epochs))

        # --- Optional BCE-only warmup ---
        # This gives the classifier a meaningful prediction surface before the
        # fairness term sees threshold-sensitive or subgroup-conditioned counts.
        for epoch in range(warmup_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            y_pred_train = self.model(X_train_t)
            pred_loss = self.ce_loss(y_pred_train, y_train_t)
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                y_pred_val = self.model(X_val_t)
                val_bce = self.ce_loss(y_pred_val, y_val_t).item()

            history.append(EpochLog(
                epoch=epoch,
                train_pred_loss=pred_loss.item(),
                train_fair_loss=0.0,
                train_total_loss=pred_loss.item(),
                val_pred_loss=val_bce,
            ))

            if val_bce < best_val_loss:
                best_val_loss = val_bce
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if best_state is not None and warmup_epochs > 0:
            self.model.load_state_dict(best_state)
            warmup_best_val_loss = best_val_loss
            # Drop warmup Adam moments so the joint phase starts from the best
            # BCE checkpoint without stale momentum from discarded epochs.
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            warmup_best_val_loss = None

        if warmup_epochs < cfg.max_epochs:
            best_val_loss = float("inf")
            best_state = None
        use_guarded_checkpoint = (
            cfg.joint_checkpoint_mode == "bce_guarded_fairness"
            and self.asbr_config.lambd >= cfg.joint_bce_guard_min_lambd
        )
        best_guarded_score = float("inf")
        best_guarded_fair_loss = float("inf")
        best_guarded_val_bce = float("inf")
        guarded_best_state = None
        # With warmup, keep the BCE guard anchored to the restored BCE-only
        # checkpoint. Without warmup, the anchor is the best joint val BCE seen
        # so far. This lets high-lambda runs select a genuinely fairer epoch
        # without accepting a large predictive-loss collapse.
        guard_anchor_val_loss = warmup_best_val_loss

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.lr_reduce_factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )

        for epoch in range(warmup_epochs, cfg.max_epochs):
            joint_epoch = epoch - warmup_epochs
            ramp_epochs = cfg.fairness_ramp_epochs
            if self.asbr_config.lambd < cfg.fairness_ramp_min_lambd:
                ramp_epochs = 0
            ramp_factor = self._fairness_ramp_factor(
                joint_epoch,
                ramp_epochs,
                cfg.fairness_ramp_shape,
            )
            effective_lambd = self.asbr_config.lambd * ramp_factor

            # --- Forward on full training set ---
            self.model.train()
            self.optimizer.zero_grad()

            y_pred_train = self.model(X_train_t)
            pred_loss = self.ce_loss(y_pred_train, y_train_t)
            # Fairness runs on CPU: move predictions off GPU to avoid 176× kernel-launch overhead.
            # Autograd traces through .cpu() / .to(device) back to GPU model weights correctly.
            fair_loss_cpu, diagnostics = self.asbr_loss.compute_fairness_loss(
                y_pred_train.cpu(), subgroup_masks_train, y_true=y_train_cpu
            )
            fair_loss = fair_loss_cpu.to(self.device)
            total_loss = pred_loss + effective_lambd * fair_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            pred_loss_val_num = pred_loss.item()
            fair_loss_val_num = fair_loss.item()
            total_loss_val_num = total_loss.item()

            # --- Validation loss (BCE only, for comparability) ---
            self.model.eval()
            with torch.no_grad():
                y_pred_val = self.model(X_val_t)
                val_bce = self.ce_loss(y_pred_val, y_val_t).item()
            checkpoint_fair_loss = fair_loss_val_num
            if use_guarded_checkpoint:
                with torch.no_grad():
                    y_pred_train_ckpt = self.model(X_train_t)
                    fair_loss_ckpt_cpu, _ = self.asbr_loss.compute_fairness_loss(
                        y_pred_train_ckpt.cpu(), subgroup_masks_train, y_true=y_train_cpu
                    )
                checkpoint_fair_loss = fair_loss_ckpt_cpu.item()

            history.append(EpochLog(
                epoch=epoch,
                train_pred_loss=pred_loss_val_num,
                train_fair_loss=fair_loss_val_num,
                train_total_loss=total_loss_val_num,
                val_pred_loss=val_bce,
                diagnostics=diagnostics,
            ))

            # --- LR schedule + checkpoint ---
            # The default selector is val BCE only, preserving historical runs and
            # avoiding constant-predictor checkpoints. With BCE warmup, however,
            # high-lambda fairness training can make val BCE monotonically worse,
            # so best-BCE selection often saves the first post-warmup epoch. The
            # guarded selector fixes that by choosing the best λ-weighted
            # objective among epochs that remain close to the BCE anchor.
            if val_bce < best_val_loss:
                best_val_loss = val_bce
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            scheduler_monitor = val_bce
            if use_guarded_checkpoint:
                if guard_anchor_val_loss is None:
                    guard_anchor_val_loss = val_bce
                else:
                    guard_anchor_val_loss = min(guard_anchor_val_loss, val_bce)
                max_guard_bce = guard_anchor_val_loss * (1.0 + cfg.joint_bce_guard_rel_tol)
                if val_bce <= max_guard_bce:
                    checkpoint_score = (
                        val_bce + self.asbr_config.lambd * checkpoint_fair_loss
                    )
                    if (
                        checkpoint_score < best_guarded_score
                        or (
                            checkpoint_score == best_guarded_score
                            and (
                                checkpoint_fair_loss < best_guarded_fair_loss
                                or (
                                    checkpoint_fair_loss == best_guarded_fair_loss
                                    and val_bce < best_guarded_val_bce
                                )
                            )
                        )
                    ):
                        best_guarded_score = checkpoint_score
                        best_guarded_fair_loss = checkpoint_fair_loss
                        best_guarded_val_bce = val_bce
                        guarded_best_state = {
                            k: v.clone() for k, v in self.model.state_dict().items()
                        }
                    scheduler_monitor = checkpoint_score

            scheduler.step(scheduler_monitor)
            if self.optimizer.param_groups[0]["lr"] <= cfg.min_lr:
                break

        if use_guarded_checkpoint and guarded_best_state is not None:
            best_state = guarded_best_state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history


@dataclass
class LambdaCheckpoint:
    """Best model state at a specific lambda step in progressive training."""

    lambd: float
    state_dict: dict
    phase2_epochs: int
    train_msee: float | None = None   # train MSEE at end of this step (hard predictions)
    stop_reason: str | None = None    # "target_reached" | "plateau" | "lambda_max" | None
    selected: bool = False            # True for the checkpoint chosen by stopping criterion


class ProgressiveTrainer:
    """Phase 1: BCE to convergence. Phase 2: progressive λ schedule.

    Unlike the fixed-λ :class:`Trainer`, Phase 2 iterates over a list of λ values in
    increasing order.  One model snapshot is produced per λ value.

    Two Phase 2 strategies are supported via ``reset_to_phase1``:

    * ``False`` (default, **warm-start**): each λ step starts from the best checkpoint
      of the previous step.  Fewer epochs needed per step; the model builds on prior
      fairness adjustments.  Most efficient — a true single-run Pareto sweep.

    * ``True`` (**Phase-1-reset**): each λ step resets to the Phase 1 checkpoint before
      running Phase 2.  Equivalent to N independent TwoPhase runs sharing a single Phase 1.
      Avoids local-minimum trapping at the cost of losing warm-start efficiency.

    Args:
        model: FairnessMLP instance.
        base_asbr_config: ASBRConfig template. ``lambd`` will be overridden per step.
        training_config: TrainingConfig for the training loop.
        device: Torch device.
        loss_class: Fairness loss class. Defaults to ASBRLoss.
        bce_weight: Optional BCE anchor weight for Phase 2. When > 0, each λ-step
            optimizes ``bce_weight * BCE + λ * fairness`` instead of pure fairness.
        checkpoint_mode: ``"combined"`` (default) keeps the current selector
            based on ``val_bce + λ * fair_loss``. ``"bce_guarded_fairness"``
            chooses the lowest-fairness epoch whose validation BCE stays within
            ``(1 + bce_guard_rel_tol)`` of the Phase 1 optimum.
        bce_guard_rel_tol: Relative validation-BCE slack used only by
            ``checkpoint_mode="bce_guarded_fairness"``.
        reset_to_phase1: If True, reset to the Phase 1 checkpoint before each λ step
            instead of warm-starting from the previous step.
    """

    def __init__(
        self,
        model: FairnessMLP,
        base_asbr_config: ASBRConfig | None = None,
        training_config: TrainingConfig | None = None,
        device: torch.device | str | None = None,
        loss_class: type | None = None,
        bce_weight: float = 1e-6,
        checkpoint_mode: str = "combined",
        bce_guard_rel_tol: float = 0.0,
        reset_to_phase1: bool = False,
    ):
        device = _default_device() if device is None else torch.device(device)
        self.model = model.to(device)
        self.base_asbr_config = base_asbr_config or ASBRConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = device
        self.loss_class = loss_class or ASBRLoss
        self.bce_weight = bce_weight
        self.checkpoint_mode = checkpoint_mode
        self.bce_guard_rel_tol = bce_guard_rel_tol
        self.reset_to_phase1 = reset_to_phase1
        self.ce_loss = nn.BCELoss()

    def _make_loader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        subgroup_masks_train: dict[str, torch.Tensor],
        X_val: np.ndarray,
        y_val: np.ndarray,
        lambda_schedule: list[float],
    ) -> tuple[list[EpochLog], list[LambdaCheckpoint]]:
        """Run progressive two-phase training.

        Args:
            X_train: Training features, shape (n_train, d).
            y_train: Training labels, shape (n_train,).
            subgroup_masks_train: Torch boolean masks for fairness on train set.
            X_val: Validation features, shape (n_val, d).
            y_val: Validation labels, shape (n_val,).
            lambda_schedule: Increasing sequence of λ values. One Pareto snapshot
                is produced per value.

        Returns:
            phase1_history: EpochLog list for Phase 1.
            checkpoints: One LambdaCheckpoint per λ step.
        """
        cfg = self.training_config

        X_train_full = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_full = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        y_train_cpu = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        train_loader = self._make_loader(X_train, y_train)

        # ===== Phase 1: BCE training to convergence (patience-based) =====
        phase1_history: list[EpochLog] = []
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        best_val = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(cfg.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                loss = self.ce_loss(self.model(X_b), y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_pred = epoch_loss / max(n_batches, 1)

            self.model.eval()
            with torch.no_grad():
                val_bce = self.ce_loss(self.model(X_val_t), y_val_t).item()

            phase1_history.append(EpochLog(
                epoch=epoch,
                train_pred_loss=avg_pred,
                train_fair_loss=0.0,
                train_total_loss=avg_pred,
                val_pred_loss=val_bce,
            ))

            if val_bce < best_val:
                best_val = val_bce
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= cfg.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Save Phase 1 state for optional reset between λ steps.
        phase1_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # ===== Phase 2: SGD per λ step (no gradient clipping) =====
        #
        # Each step runs SGD on bce_weight·BCE + λ·fairness until patience fires on
        # val_bce + λ·fair_loss.
        # Starting point depends on reset_to_phase1:
        #   False (warm-start): continue from previous step's best checkpoint — fewer
        #       epochs needed, true single-run sweep.
        #   True (Phase-1-reset): reload Phase 1 checkpoint each step — equivalent to
        #       N independent TwoPhase runs sharing one Phase 1.
        #
        # SGD preserves gradient scale so λ directly controls fairness strength.
        # No gradient clipping: the two-phase auto-override in run_hydra.py
        # (max_grad_norm → 0) is NOT applied to ProgressiveASBRMethod.

        checkpoints: list[LambdaCheckpoint] = []

        for lambd in lambda_schedule:
            if self.reset_to_phase1:
                # Reset to Phase 1 checkpoint and fresh optimizer (no stale momentum).
                self.model.load_state_dict({k: v.clone() for k, v in phase1_state.items()})
            fair_optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr)
            step_config = ASBRConfig(
                alpha=self.base_asbr_config.alpha,
                lambd=lambd,
                n_tilde=self.base_asbr_config.n_tilde,
                kappa=self.base_asbr_config.kappa,
                epsilon=self.base_asbr_config.epsilon,
                fairness_metric=self.base_asbr_config.fairness_metric,
                use_hinge=self.base_asbr_config.use_hinge,
                use_hybrid=self.base_asbr_config.use_hybrid,
                use_size_weight=self.base_asbr_config.use_size_weight,
                use_sp_magnitude=self.base_asbr_config.use_sp_magnitude,
                gate_alpha=self.base_asbr_config.gate_alpha,
                hinge_alpha=self.base_asbr_config.hinge_alpha,
                use_squared_hinge=self.base_asbr_config.use_squared_hinge,
                dense_tail_weight=self.base_asbr_config.dense_tail_weight,
                dense_tail_shape=self.base_asbr_config.dense_tail_shape,
                loss_shape=self.base_asbr_config.loss_shape,
                direct_tau=self.base_asbr_config.direct_tau,
                direct_abs_eps=self.base_asbr_config.direct_abs_eps,
                direct_detach_se=self.base_asbr_config.direct_detach_se,
                direct_pred_temperature=self.base_asbr_config.direct_pred_temperature,
                direct_threshold=self.base_asbr_config.direct_threshold,
                direct_lrt_detach=self.base_asbr_config.direct_lrt_detach,
                direct_lrt_sqrt_min=self.base_asbr_config.direct_lrt_sqrt_min,
                direct_lrt_se_min=self.base_asbr_config.direct_lrt_se_min,
                direct_lrt_se_max=self.base_asbr_config.direct_lrt_se_max,
                use_rms_aggregation=self.base_asbr_config.use_rms_aggregation,
                use_overall_reference=self.base_asbr_config.use_overall_reference,
            )
            asbr_loss = self.loss_class(step_config)

            best_combined = float("inf")
            best_step_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            guarded_step_state = None
            best_guarded_fair = float("inf")
            best_guarded_bce = float("inf")
            patience_counter = 0
            step_epochs = 0

            for _ in range(cfg.max_epochs):
                self.model.train()
                fair_optimizer.zero_grad()

                y_pred_full = self.model(X_train_full)
                pred_loss_full = self.ce_loss(y_pred_full, y_train_full)
                fair_loss_cpu, _ = asbr_loss.compute_fairness_loss(
                    y_pred_full.cpu(), subgroup_masks_train, y_true=y_train_cpu
                )
                fair_loss = fair_loss_cpu.to(self.device)
                total_loss = self.bce_weight * pred_loss_full + lambd * fair_loss

                if total_loss.requires_grad:
                    total_loss.backward()
                    # No gradient clipping: SGD gradient scale must be preserved so
                    # λ directly controls the effective step size (same as Trainer
                    # Phase 2 with max_grad_norm=0).
                    fair_optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    val_bce = self.ce_loss(self.model(X_val_t), y_val_t).item()

                monitor = val_bce + lambd * fair_loss.item()
                step_epochs += 1

                if monitor < best_combined:
                    best_combined = monitor
                    best_step_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.checkpoint_mode == "bce_guarded_fairness":
                    max_guard_bce = best_val * (1.0 + self.bce_guard_rel_tol)
                    fair_loss_val = fair_loss.item()
                    if val_bce <= max_guard_bce and (
                        fair_loss_val < best_guarded_fair
                        or (
                            fair_loss_val == best_guarded_fair
                            and val_bce < best_guarded_bce
                        )
                    ):
                        best_guarded_fair = fair_loss_val
                        best_guarded_bce = val_bce
                        guarded_step_state = {
                            k: v.clone() for k, v in self.model.state_dict().items()
                        }
                if patience_counter >= cfg.patience:
                    break

            selected_state = best_step_state
            if self.checkpoint_mode == "bce_guarded_fairness" and guarded_step_state is not None:
                selected_state = guarded_step_state

            self.model.load_state_dict(selected_state)
            train_msee = _compute_msee_hard(
                self.model,
                X_train_full,
                y_train.astype(float),
                subgroup_masks_train,
                fairness_metric=step_config.fairness_metric,
                alpha=step_config.alpha,
                threshold=step_config.direct_threshold,
            )
            checkpoints.append(LambdaCheckpoint(
                lambd=lambd,
                state_dict=selected_state,
                phase2_epochs=step_epochs,
                train_msee=train_msee,
            ))

        return phase1_history, checkpoints


def _compute_msee_hard(
    model: FairnessMLP,
    X_full: torch.Tensor,
    y_true_np: np.ndarray,
    subgroup_masks: dict[str, torch.Tensor],
    fairness_metric: str = "eo",
    alpha: float = 0.05,
    threshold: float = 0.5,
) -> float:
    """Compute MSEE (EO) or MSPE (SP) with hard predictions on the full training set.

    Uses the same Delta Method SE formula as SAFT, computed directly without
    importing the saft module.  Intended for training-time stopping criteria
    where the validation set is too small to give reliable fairness statistics.

    Args:
        model: Trained FairnessMLP (eval mode is set internally).
        X_full: Full training features as a float32 tensor on the model device.
        y_true_np: Ground-truth labels as a numpy float array, shape (n,).
        subgroup_masks: Dict of boolean torch tensors, one per subgroup.
        fairness_metric: ``"eo"`` (TPR gap) or ``"sp"`` (positive-rate gap).
        alpha: Significance level; z_alpha = norm.ppf(1 - alpha/2).
        threshold: Hard-prediction threshold (default 0.5).

    Returns:
        Mean significant excess over all qualifying subgroups (≥ 0).
    """
    from scipy.stats import norm as _norm

    z_alpha = float(_norm.ppf(1.0 - alpha / 2.0))
    eps = 1e-9

    model.eval()
    with torch.no_grad():
        y_prob = model(X_full).cpu().numpy()
    y_hard = (y_prob >= threshold).astype(float)

    excesses: list[float] = []
    for mask_t in subgroup_masks.values():
        mask = mask_t.numpy().astype(bool)
        mask_c = ~mask

        if fairness_metric == "eo":
            pos = y_true_np == 1
            pos_S = pos & mask
            pos_Sbar = pos & mask_c
            m_S = int(pos_S.sum())
            m_Sbar = int(pos_Sbar.sum())
            if m_S < 2 or m_Sbar < 2:
                continue
            r_S = float(y_hard[pos_S].mean())
            r_Sbar = float(y_hard[pos_Sbar].mean())
            se = float(np.sqrt(r_S * (1 - r_S) / m_S + r_Sbar * (1 - r_Sbar) / m_Sbar + eps))
            excess = max(0.0, abs(r_S - r_Sbar) - z_alpha * se)
        else:
            n_S = int(mask.sum())
            n_Sbar = int(mask_c.sum())
            if n_S < 2 or n_Sbar < 2:
                continue
            p_S = float(y_hard[mask].mean())
            p_Sbar = float(y_hard[mask_c].mean())
            se = float(np.sqrt(p_S * (1 - p_S) / n_S + p_Sbar * (1 - p_Sbar) / n_Sbar + eps))
            excess = max(0.0, abs(p_S - p_Sbar) - z_alpha * se)

        excesses.append(excess)

    return float(np.mean(excesses)) if excesses else 0.0


class TargetProgressiveTrainer:
    """Progressive λ training that searches for a target MSEE/MSPE.

    Replaces the fixed lambda_schedule with an adaptive geometric search.
    Starting from ``lambda_min``, λ is multiplied by ``lambda_multiplier``
    each step until one of three stopping conditions fires (in priority order):

    1. ``train_msee ≤ target`` → ``"target_reached"``: minimum λ achieving goal.
    2. No MSEE improvement for ``msee_patience`` consecutive steps →
       ``"plateau"``: target unreachable; best checkpoint returned.
    3. ``λ > lambda_max`` → ``"lambda_max"``: ceiling hit; best checkpoint returned.

    The full search trajectory is returned as a list of
    :class:`LambdaCheckpoint` objects, each with ``train_msee`` and
    ``stop_reason`` populated. The ``selected`` flag is set to ``True``
    on the recommended checkpoint.

    Args:
        model: FairnessMLP instance.
        base_asbr_config: ASBRConfig template; ``lambd`` is overridden per step.
        training_config: TrainingConfig for the training loop.
        device: Torch device.
        loss_class: Fairness loss class. Defaults to ASBRLoss.
        bce_weight: BCE anchor weight added to the fairness loss in Phase 2.
        target_msee: Absolute MSEE/MSPE target to reach.
        target_reduction: Fraction of ERM MSEE to eliminate (0–1). Used when
            ``target_msee`` is None; target = erm_msee × (1 − target_reduction).
        lambda_min: Starting λ (default 0.1).
        lambda_max: Maximum λ; search stops if exceeded (default 500.0).
        lambda_multiplier: Geometric step factor > 1 (default 1.5).
        msee_patience: Steps without MSEE improvement before declaring plateau
            (default 2).
    """

    def __init__(
        self,
        model: FairnessMLP,
        base_asbr_config: ASBRConfig | None = None,
        training_config: TrainingConfig | None = None,
        device: torch.device | str | None = None,
        loss_class: type | None = None,
        bce_weight: float = 1e-6,
        target_msee: float | None = None,
        target_reduction: float | None = None,
        lambda_min: float = 0.1,
        lambda_max: float = 500.0,
        lambda_multiplier: float = 1.5,
        msee_patience: int = 2,
    ):
        if target_msee is None and target_reduction is None:
            raise ValueError("Either target_msee or target_reduction must be set.")
        if lambda_multiplier <= 1.0:
            raise ValueError("lambda_multiplier must be > 1.0.")

        device = _default_device() if device is None else torch.device(device)
        self.model = model.to(device)
        self.base_asbr_config = base_asbr_config or ASBRConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = device
        self.loss_class = loss_class or ASBRLoss
        self.bce_weight = bce_weight
        self.target_msee = target_msee
        self.target_reduction = target_reduction
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_multiplier = lambda_multiplier
        self.msee_patience = msee_patience
        self.ce_loss = nn.BCELoss()

    def _make_loader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        subgroup_masks_train: dict[str, torch.Tensor],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[list[EpochLog], list[LambdaCheckpoint], int]:
        """Run Phase 1 then adaptive Phase 2 target search.

        Returns:
            phase1_history: EpochLog list for Phase 1.
            checkpoints: One LambdaCheckpoint per λ step visited. The last
                entry carries the ``stop_reason`` that triggered termination;
                the selected checkpoint has ``selected=True``.
            selected_idx: Index into ``checkpoints`` of the recommended model.
        """
        cfg = self.training_config

        X_train_full = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_full = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        y_train_cpu = torch.tensor(y_train, dtype=torch.float32)
        y_train_np = y_train.astype(float)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        train_loader = self._make_loader(X_train, y_train)

        # ===== Phase 1: BCE to convergence =====
        phase1_history: list[EpochLog] = []
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        best_val = float("inf")
        best_p1_state = None
        patience_counter = 0

        for epoch in range(cfg.max_epochs):
            self.model.train()
            epoch_loss, n_batches = 0.0, 0
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                loss = self.ce_loss(self.model(X_b), y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_pred = epoch_loss / max(n_batches, 1)

            self.model.eval()
            with torch.no_grad():
                val_bce = self.ce_loss(self.model(X_val_t), y_val_t).item()

            phase1_history.append(EpochLog(
                epoch=epoch, train_pred_loss=avg_pred, train_fair_loss=0.0,
                train_total_loss=avg_pred, val_pred_loss=val_bce,
            ))
            if val_bce < best_val:
                best_val = val_bce
                best_p1_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= cfg.patience:
                break

        if best_p1_state is not None:
            self.model.load_state_dict(best_p1_state)

        # ===== ERM baseline MSEE (λ=0) — used for relative targets =====
        erm_msee = _compute_msee_hard(
            self.model, X_train_full, y_train_np, subgroup_masks_train,
            fairness_metric=self.base_asbr_config.fairness_metric,
            alpha=self.base_asbr_config.alpha,
        )
        if self.target_msee is not None:
            abs_target = self.target_msee
        else:
            abs_target = erm_msee * (1.0 - self.target_reduction)  # type: ignore[operator]

        print(
            f"  [Target PLT] ERM train_msee={erm_msee:.6f}, "
            f"target={abs_target:.6f} "
            f"({'absolute' if self.target_msee is not None else f'{self.target_reduction:.0%} reduction'})"
        )

        # ===== Phase 2: Adaptive geometric λ search =====
        checkpoints: list[LambdaCheckpoint] = []
        best_msee = erm_msee
        best_idx = 0
        plateau_count = 0
        found_first_improvement = False
        lambd = self.lambda_min

        while True:
            step_config = ASBRConfig(
                alpha=self.base_asbr_config.alpha,
                lambd=lambd,
                n_tilde=self.base_asbr_config.n_tilde,
                kappa=self.base_asbr_config.kappa,
                epsilon=self.base_asbr_config.epsilon,
                fairness_metric=self.base_asbr_config.fairness_metric,
                use_hinge=self.base_asbr_config.use_hinge,
                use_hybrid=self.base_asbr_config.use_hybrid,
                use_size_weight=self.base_asbr_config.use_size_weight,
                use_sp_magnitude=self.base_asbr_config.use_sp_magnitude,
                gate_alpha=self.base_asbr_config.gate_alpha,
                hinge_alpha=self.base_asbr_config.hinge_alpha,
                use_squared_hinge=self.base_asbr_config.use_squared_hinge,
                dense_tail_weight=self.base_asbr_config.dense_tail_weight,
                dense_tail_shape=self.base_asbr_config.dense_tail_shape,
                loss_shape=self.base_asbr_config.loss_shape,
                direct_tau=self.base_asbr_config.direct_tau,
                direct_abs_eps=self.base_asbr_config.direct_abs_eps,
                direct_detach_se=self.base_asbr_config.direct_detach_se,
                direct_pred_temperature=self.base_asbr_config.direct_pred_temperature,
                direct_threshold=self.base_asbr_config.direct_threshold,
                direct_lrt_detach=self.base_asbr_config.direct_lrt_detach,
                direct_lrt_sqrt_min=self.base_asbr_config.direct_lrt_sqrt_min,
                direct_lrt_se_min=self.base_asbr_config.direct_lrt_se_min,
                direct_lrt_se_max=self.base_asbr_config.direct_lrt_se_max,
                use_rms_aggregation=self.base_asbr_config.use_rms_aggregation,
                use_overall_reference=self.base_asbr_config.use_overall_reference,
            )
            asbr_loss_fn = self.loss_class(step_config)
            fair_optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr)

            best_combined = float("inf")
            best_step_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            step_patience = 0
            step_epochs = 0

            for _ in range(cfg.max_epochs):
                self.model.train()
                fair_optimizer.zero_grad()
                y_pred_full = self.model(X_train_full)
                pred_loss_full = self.ce_loss(y_pred_full, y_train_full)
                fair_loss_cpu, _ = asbr_loss_fn.compute_fairness_loss(
                    y_pred_full.cpu(), subgroup_masks_train, y_true=y_train_cpu
                )
                fair_loss = fair_loss_cpu.to(self.device)
                total_loss = self.bce_weight * pred_loss_full + lambd * fair_loss
                if total_loss.requires_grad:
                    total_loss.backward()
                    fair_optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    val_bce = self.ce_loss(self.model(X_val_t), y_val_t).item()

                monitor = val_bce + lambd * fair_loss.item()
                step_epochs += 1
                if monitor < best_combined:
                    best_combined = monitor
                    best_step_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    step_patience = 0
                else:
                    step_patience += 1
                if step_patience >= cfg.patience:
                    break

            self.model.load_state_dict(best_step_state)

            # Evaluate stopping criterion with hard predictions on full train set
            train_msee = _compute_msee_hard(
                self.model, X_train_full, y_train_np, subgroup_masks_train,
                fairness_metric=self.base_asbr_config.fairness_metric,
                alpha=self.base_asbr_config.alpha,
            )

            # Update global best before checking plateau.
            # Plateau is only counted after the first improvement has been seen —
            # otherwise the search stops prematurely in the flat low-λ region
            # before the fairness loss has had any measurable effect.
            if train_msee < best_msee:
                best_msee = train_msee
                best_idx = len(checkpoints)
                plateau_count = 0
                found_first_improvement = True
            elif found_first_improvement:
                plateau_count += 1

            # Determine stop reason (checked in priority order)
            stop_reason: str | None = None
            if train_msee <= abs_target:
                stop_reason = "target_reached"
            elif found_first_improvement and plateau_count >= self.msee_patience:
                stop_reason = "plateau"
            elif lambd >= self.lambda_max:
                stop_reason = "lambda_max"

            checkpoints.append(LambdaCheckpoint(
                lambd=lambd,
                state_dict=best_step_state,
                phase2_epochs=step_epochs,
                train_msee=train_msee,
                stop_reason=stop_reason,
            ))

            tag = stop_reason or "continue"
            print(
                f"  [Target PLT] λ={lambd:.4f}: train_msee={train_msee:.6f} "
                f"(target={abs_target:.6f}, best={best_msee:.6f}, "
                f"plateau={plateau_count}/{self.msee_patience}) → {tag}"
            )

            if stop_reason is not None:
                break

            lambd = min(lambd * self.lambda_multiplier, self.lambda_max)

        # selected_idx: last checkpoint on target_reached; best_idx on plateau/lambda_max
        if checkpoints[-1].stop_reason == "target_reached":
            selected_idx = len(checkpoints) - 1
        else:
            selected_idx = best_idx

        checkpoints[selected_idx].selected = True
        self.model.load_state_dict(checkpoints[selected_idx].state_dict)

        return phase1_history, checkpoints, selected_idx


class BaselineTrainer:
    """Vanilla training loop without fairness loss (ERM baseline).

    Uses the same training protocol (mini-batch SGD, early stopping)
    but without any fairness regularization.
    """

    def __init__(
        self,
        model: FairnessMLP,
        training_config: TrainingConfig | None = None,
        device: torch.device | str | None = None,
    ):
        device = _default_device() if device is None else torch.device(device)
        self.model = model.to(device)
        self.training_config = training_config or TrainingConfig()
        self.device = device
        self.ce_loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[EpochLog]:
        """Run vanilla training loop (no fairness loss)."""
        cfg = self.training_config

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=cfg.batch_size, shuffle=True,
        )

        history: list[EpochLog] = []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(cfg.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.ce_loss(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            self.model.eval()
            with torch.no_grad():
                y_pred_val = self.model(X_val_t)
                val_loss = self.ce_loss(y_pred_val, y_val_t).item()

            history.append(EpochLog(
                epoch=epoch,
                train_pred_loss=avg_loss,
                train_fair_loss=0.0,
                train_total_loss=avg_loss,
                val_pred_loss=val_loss,
            ))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history
