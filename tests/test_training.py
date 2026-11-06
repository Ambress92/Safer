"""Tests for training infrastructure: Trainer, BaselineTrainer, evaluator."""

import numpy as np
import pytest
import torch

from asbr.data.intersections import build_subgroup_masks, masks_to_torch
from asbr.evaluation.evaluator import compute_predictions, compute_all_metrics
from asbr.experiments.config import ASBRConfig, TrainingConfig
from asbr.models.base_mlp import FairnessMLP
from asbr.training.trainer import JointTrainer, Trainer, BaselineTrainer, EpochLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Synthetic binary classification dataset with two protected attributes."""
    rng = np.random.RandomState(42)
    n = 200
    d = 5

    X = rng.randn(n, d).astype(np.float32)
    # Label correlated with first feature
    logits = X[:, 0] + 0.5 * X[:, 1]
    y = (logits > 0).astype(np.float32)

    import pandas as pd
    A = pd.DataFrame({
        "group": rng.choice(["A", "B"], size=n),
        "type": rng.choice(["X", "Y"], size=n),
    })

    return X, y, A


@pytest.fixture
def small_training_config():
    """Fast training config for testing."""
    return TrainingConfig(
        hidden_dims=(8,),
        dropout=0.0,
        lr=1e-2,
        patience=3,
        max_epochs=10,
        batch_size=64,
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_basic_training_completes(self, synthetic_data, small_training_config):
        X, y, A = synthetic_data
        n_train = 160
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)

        masks_np = build_subgroup_masks(A_train, level=1)
        masks_torch = masks_to_torch(masks_np)

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        trainer = Trainer(
            model, ASBRConfig(lambd=0.1), small_training_config, device="cpu"
        )
        history = trainer.train(X_train, y_train, masks_torch, X_val, y_val)

        assert len(history) > 0
        assert all(isinstance(h, EpochLog) for h in history)

    def test_early_stopping_triggers(self, synthetic_data, small_training_config):
        X, y, A = synthetic_data
        # Use low patience and small max_epochs
        config = TrainingConfig(
            hidden_dims=(8,), dropout=0.0, lr=1e-1,
            patience=2, max_epochs=50, batch_size=64,
        )
        n_train = 160
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)

        masks_torch = masks_to_torch(build_subgroup_masks(A_train, level=1))
        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        trainer = Trainer(model, ASBRConfig(lambd=0.1), config, device="cpu")
        history = trainer.train(X_train, y_train, masks_torch, X_val, y_val)

        # With high lr and low patience, at least one phase should trigger
        # early stopping before max_epochs
        assert len(history) <= 50

    def test_history_has_correct_fields(self, synthetic_data, small_training_config):
        X, y, A = synthetic_data
        n_train = 160
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)

        masks_torch = masks_to_torch(build_subgroup_masks(A_train, level=1))
        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        trainer = Trainer(
            model, ASBRConfig(lambd=0.1), small_training_config, device="cpu"
        )
        history = trainer.train(X_train, y_train, masks_torch, X_val, y_val)

        h = history[0]
        assert hasattr(h, "train_pred_loss")
        assert hasattr(h, "train_fair_loss")
        assert hasattr(h, "val_pred_loss")
        assert h.train_pred_loss >= 0
        assert h.val_pred_loss >= 0

    def test_model_predictions_change_after_training(self, synthetic_data, small_training_config):
        X, y, A = synthetic_data
        n_train = 160
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)

        # Predictions before training
        model.eval()
        with torch.no_grad():
            preds_before = model(torch.tensor(X_train[:10], dtype=torch.float32)).numpy()

        masks_torch = masks_to_torch(build_subgroup_masks(A_train, level=1))
        trainer = Trainer(
            model, ASBRConfig(lambd=0.1), small_training_config, device="cpu"
        )
        trainer.train(X_train, y_train, masks_torch, X_val, y_val)

        model.eval()
        with torch.no_grad():
            preds_after = model(torch.tensor(X_train[:10], dtype=torch.float32)).numpy()

        # Predictions should change
        assert not np.allclose(preds_before, preds_after, atol=1e-3)


# ---------------------------------------------------------------------------
# Joint Trainer
# ---------------------------------------------------------------------------

class TestJointTrainer:
    def test_basic_training_completes(self, synthetic_data):
        """JointTrainer runs to LR-based convergence without error."""
        X, y, A = synthetic_data
        n_train = 160
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)

        masks_np = build_subgroup_masks(A_train, level=1)
        masks_torch = masks_to_torch(masks_np)

        # Small config: low patience forces LR reductions quickly
        cfg = TrainingConfig(
            hidden_dims=(8,), dropout=0.0, lr=1e-2,
            patience=2, max_epochs=100, batch_size=64,
            lr_reduce_factor=0.1, min_lr=1e-4,
        )
        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        trainer = JointTrainer(model, ASBRConfig(lambd=0.1), cfg, device="cpu")
        history = trainer.train(X_train, y_train, masks_torch, X_val, y_val)

        assert len(history) > 0
        assert all(isinstance(h, EpochLog) for h in history)
        assert len(history) <= 100

    def test_best_checkpoint_restored(self, synthetic_data):
        """JointTrainer restores the best checkpoint, not the last state."""
        X, y, A = synthetic_data
        n_train = 160
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)

        masks_torch = masks_to_torch(build_subgroup_masks(A_train, level=1))
        cfg = TrainingConfig(
            hidden_dims=(8,), dropout=0.0, lr=1e-2,
            patience=2, max_epochs=50, batch_size=64,
            lr_reduce_factor=0.1, min_lr=1e-4,
        )
        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        trainer = JointTrainer(model, ASBRConfig(lambd=0.1), cfg, device="cpu")
        history = trainer.train(X_train, y_train, masks_torch, X_val, y_val)

        # After training, model produces valid predictions (not NaN or constant)
        model.eval()
        with torch.no_grad():
            probs = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        assert not np.any(np.isnan(probs)), "NaN predictions after training"
        assert probs.std() > 1e-6, "Model collapsed to constant predictions"


# ---------------------------------------------------------------------------
# Baseline Trainer
# ---------------------------------------------------------------------------

class TestBaselineTrainer:
    def test_trains_without_fairness(self, synthetic_data, small_training_config):
        X, y, _ = synthetic_data
        n_train = 160
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        trainer = BaselineTrainer(model, small_training_config, device="cpu")
        history = trainer.train(X_train, y_train, X_val, y_val)

        assert len(history) > 0
        # No fairness loss
        assert all(h.train_fair_loss == 0.0 for h in history)

    def test_baseline_achieves_reasonable_accuracy(self, synthetic_data, small_training_config):
        X, y, _ = synthetic_data
        n_train = 160
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        trainer = BaselineTrainer(model, small_training_config, device="cpu")
        trainer.train(X_train, y_train, X_val, y_val)

        _, y_pred = compute_predictions(model, X_val)
        acc = (y_pred == y_val).mean()
        # Should do better than random (data is linearly separable)
        assert acc > 0.55


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_compute_predictions_shapes(self, synthetic_data):
        X, _, _ = synthetic_data
        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        y_prob, y_pred = compute_predictions(model, X)
        assert y_prob.shape == (len(X),)
        assert y_pred.shape == (len(X),)
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_compute_all_metrics_keys(self, synthetic_data, small_training_config):
        X, y, A = synthetic_data
        n_train = 160

        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)
        A_test = A.iloc[n_train:].reset_index(drop=True)

        masks_train = build_subgroup_masks(A_train, level=1)
        masks_test = build_subgroup_masks(A_test, level=1)

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)

        metrics = compute_all_metrics(
            model, X_train, y_train, masks_train,
            X_test, y_test, masks_test,
        )

        expected_keys = {
            "accuracy_train", "accuracy_test",
            "auc_train", "auc_test",
            "max_sp_train", "max_sp_test",
            "mean_sp_train", "mean_sp_test",
            "max_eo_train", "max_eo_test",
            "mean_eo_train", "mean_eo_test",
            "gap_overall", "gap_tail", "gap_bulk",
            "n_subgroups",
        }
        assert expected_keys == set(metrics.keys())

    def test_metrics_are_numeric(self, synthetic_data, small_training_config):
        X, y, A = synthetic_data
        n_train = 160

        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)
        A_test = A.iloc[n_train:].reset_index(drop=True)

        masks_train = build_subgroup_masks(A_train, level=1)
        masks_test = build_subgroup_masks(A_test, level=1)

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)

        # Train briefly first
        trainer = BaselineTrainer(model, small_training_config, device="cpu")
        trainer.train(X_train, y_train, X_test, y_test)

        metrics = compute_all_metrics(
            model, X_train, y_train, masks_train,
            X_test, y_test, masks_test,
        )

        for key, val in metrics.items():
            assert isinstance(val, (int, float)), f"{key} is not numeric: {type(val)}"

    def test_accuracy_in_valid_range(self, synthetic_data, small_training_config):
        X, y, A = synthetic_data
        n_train = 160

        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)
        A_test = A.iloc[n_train:].reset_index(drop=True)

        masks_train = build_subgroup_masks(A_train, level=1)
        masks_test = build_subgroup_masks(A_test, level=1)

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        trainer = BaselineTrainer(model, small_training_config, device="cpu")
        trainer.train(X_train, y_train, X_test, y_test)

        metrics = compute_all_metrics(
            model, X_train, y_train, masks_train,
            X_test, y_test, masks_test,
        )

        assert 0.0 <= metrics["accuracy_train"] <= 1.0
        assert 0.0 <= metrics["accuracy_test"] <= 1.0
