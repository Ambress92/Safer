"""Tests for fairness baselines: ExponGrad wrapper, evaluator refactor, config presets."""

import numpy as np
import pandas as pd
import pytest

from asbr.data.intersections import build_subgroup_masks
from asbr.evaluation.evaluator import (
    compute_predictions,
    compute_all_metrics,
    compute_all_metrics_from_predictions,
)
from asbr.experiments.config import ASBRConfig, TrainingConfig
from asbr.models.base_mlp import FairnessMLP
from asbr.training.trainer import BaselineTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Synthetic binary classification dataset with two protected attributes."""
    rng = np.random.RandomState(42)
    n = 300
    d = 5

    X = rng.randn(n, d).astype(np.float32)
    logits = X[:, 0] + 0.5 * X[:, 1]
    y = (logits > 0).astype(np.float32)

    A = pd.DataFrame({
        "group": rng.choice(["A", "B"], size=n),
        "type": rng.choice(["X", "Y"], size=n),
    })

    return X, y, A


@pytest.fixture
def small_training_config():
    return TrainingConfig(
        hidden_dims=(8,),
        dropout=0.0,
        lr=1e-2,
        patience=3,
        max_epochs=10,
        batch_size=64,
    )


# ---------------------------------------------------------------------------
# ASBRConfig preset
# ---------------------------------------------------------------------------

class TestSPPenaltyPreset:
    def test_preset_disables_all_flags(self):
        config = ASBRConfig.sp_penalty_preset()
        assert config.use_hinge is False
        assert config.use_hybrid is False
        assert config.use_size_weight is False

    def test_preset_keeps_defaults_for_other_params(self):
        config = ASBRConfig.sp_penalty_preset()
        default = ASBRConfig()
        assert config.alpha == default.alpha
        assert config.lambd == default.lambd
        assert config.epsilon == default.epsilon


class TestEOPenaltyPreset:
    def test_preset_disables_all_flags(self):
        config = ASBRConfig.eo_penalty_preset()
        assert config.use_hinge is False
        assert config.use_hybrid is False
        assert config.use_size_weight is False

    def test_preset_sets_eo_metric(self):
        config = ASBRConfig.eo_penalty_preset()
        assert config.fairness_metric == "eo"

    def test_preset_keeps_defaults_for_other_params(self):
        config = ASBRConfig.eo_penalty_preset()
        default = ASBRConfig()
        assert config.alpha == default.alpha
        assert config.lambd == default.lambd
        assert config.epsilon == default.epsilon


# ---------------------------------------------------------------------------
# Model-agnostic evaluator
# ---------------------------------------------------------------------------

class TestComputeAllMetricsFromPredictions:
    def test_matches_compute_all_metrics(self, synthetic_data, small_training_config):
        """compute_all_metrics_from_predictions should match compute_all_metrics."""
        X, y, A = synthetic_data
        n_train = 240
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)
        A_test = A.iloc[n_train:].reset_index(drop=True)

        masks_train = build_subgroup_masks(A_train, level=1)
        masks_test = build_subgroup_masks(A_test, level=1)

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        trainer = BaselineTrainer(model, small_training_config, device="cpu")
        trainer.train(X_train, y_train, X_test, y_test)

        # Method 1: original
        metrics_original = compute_all_metrics(
            model, X_train, y_train, masks_train,
            X_test, y_test, masks_test,
        )

        # Method 2: from predictions
        y_prob_train, y_pred_train = compute_predictions(model, X_train)
        y_prob_test, y_pred_test = compute_predictions(model, X_test)
        metrics_from_preds = compute_all_metrics_from_predictions(
            y_prob_train, y_pred_train, y_train, masks_train,
            y_prob_test, y_pred_test, y_test, masks_test,
        )

        assert set(metrics_original.keys()) == set(metrics_from_preds.keys())
        for key in metrics_original:
            assert metrics_original[key] == pytest.approx(
                metrics_from_preds[key], abs=1e-10
            ), f"Mismatch on {key}"

    def test_returns_expected_keys(self, synthetic_data):
        X, y, A = synthetic_data
        n_train = 240
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)
        A_test = A.iloc[n_train:].reset_index(drop=True)

        masks_train = build_subgroup_masks(A_train, level=1)
        masks_test = build_subgroup_masks(A_test, level=1)

        # Use random predictions
        rng = np.random.RandomState(0)
        y_prob_train = rng.rand(n_train).astype(np.float32)
        y_pred_train = (y_prob_train >= 0.5).astype(int)
        y_prob_test = rng.rand(len(X_test)).astype(np.float32)
        y_pred_test = (y_prob_test >= 0.5).astype(int)

        metrics = compute_all_metrics_from_predictions(
            y_prob_train, y_pred_train, y_train, masks_train,
            y_prob_test, y_pred_test, y_test, masks_test,
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


# ---------------------------------------------------------------------------
# ExponGrad wrapper
# ---------------------------------------------------------------------------

class TestExponGradWrapper:
    @pytest.fixture(autouse=True)
    def _skip_if_no_fairlearn(self):
        pytest.importorskip("fairlearn")

    def test_fit_predict(self, synthetic_data):
        from asbr.baselines.expgrad import ExponGradWrapper, ExponGradConfig

        X, y, A = synthetic_data
        wrapper = ExponGradWrapper(ExponGradConfig(max_iter=10))
        wrapper.fit(X, y.astype(int), A, intersection_level=1)

        y_pred = wrapper.predict(X)
        assert y_pred.shape == (len(X),)
        assert set(np.unique(y_pred)).issubset({0, 1})

        y_prob = wrapper.predict_proba(X)
        assert y_prob.shape == (len(X),)
        assert np.all(y_prob >= -1e-10)
        assert np.all(y_prob <= 1 + 1e-10)

    def test_reduces_disparity_vs_unconstrained(self, synthetic_data):
        """ExponGrad should reduce max SP violation compared to unconstrained LR."""
        from sklearn.linear_model import LogisticRegression
        from asbr.baselines.expgrad import ExponGradWrapper, ExponGradConfig
        from asbr.fairness.metrics import max_sp_violation

        X, y, A = synthetic_data
        y_int = y.astype(int)
        masks = build_subgroup_masks(A, level=1)

        # Unconstrained
        lr = LogisticRegression(solver="liblinear", max_iter=1000)
        lr.fit(X, y_int)
        y_pred_lr = lr.predict(X)
        sp_lr = max_sp_violation(y_pred_lr, masks)

        # ExponGrad
        wrapper = ExponGradWrapper(ExponGradConfig(epsilon=0.01, max_iter=30))
        wrapper.fit(X, y_int, A, intersection_level=1)
        y_pred_eg = wrapper.predict(X)
        sp_eg = max_sp_violation(y_pred_eg, masks)

        # ExponGrad should not be worse (allow small tolerance)
        assert sp_eg <= sp_lr + 0.05

    def test_predict_before_fit_raises(self):
        from asbr.baselines.expgrad import ExponGradWrapper

        wrapper = ExponGradWrapper()
        with pytest.raises(RuntimeError, match="not fitted"):
            wrapper.predict(np.zeros((10, 5)))
        with pytest.raises(RuntimeError, match="not fitted"):
            wrapper.predict_proba(np.zeros((10, 5)))


# ---------------------------------------------------------------------------
# build_sensitive_features
# ---------------------------------------------------------------------------

class TestBuildSensitiveFeatures:
    @pytest.fixture(autouse=True)
    def _skip_if_no_fairlearn(self):
        pytest.importorskip("fairlearn")

    def test_level_1_uses_first_column(self):
        from asbr.baselines.expgrad import build_sensitive_features

        A = pd.DataFrame({"race": ["W", "B", "W"], "sex": ["M", "F", "F"]})
        sf = build_sensitive_features(A, level=1)
        assert list(sf) == ["W", "B", "W"]

    def test_level_2_concatenates(self):
        from asbr.baselines.expgrad import build_sensitive_features

        A = pd.DataFrame({"race": ["W", "B", "W"], "sex": ["M", "F", "F"]})
        sf = build_sensitive_features(A, level=2)
        assert list(sf) == ["W&M", "B&F", "W&F"]

    def test_level_3_with_three_columns(self):
        from asbr.baselines.expgrad import build_sensitive_features

        A = pd.DataFrame({
            "race": ["W", "B"],
            "sex": ["M", "F"],
            "age": ["<25", ">25"],
        })
        sf = build_sensitive_features(A, level=3)
        assert list(sf) == ["W&M&<25", "B&F&>25"]


# ---------------------------------------------------------------------------
# Trainer with custom loss_class
# ---------------------------------------------------------------------------

class TestTrainerLossClass:
    def test_trainer_accepts_lrt_loss(self, synthetic_data, small_training_config):
        from asbr.fairness.asbr_lrt_loss import ASBRLRTLoss
        from asbr.training.trainer import Trainer
        from asbr.data.intersections import masks_to_torch

        X, y, A = synthetic_data
        n_train = 240
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        A_train = A.iloc[:n_train].reset_index(drop=True)

        masks_torch = masks_to_torch(build_subgroup_masks(A_train, level=1))
        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        trainer = Trainer(
            model, ASBRConfig(lambd=0.1), small_training_config,
            device="cpu", loss_class=ASBRLRTLoss,
        )
        history = trainer.train(X_train, y_train, masks_torch, X_val, y_val)
        assert len(history) > 0

    def test_trainer_defaults_to_asbr_loss(self, synthetic_data, small_training_config):
        from asbr.fairness.asbr_loss import ASBRLoss
        from asbr.training.trainer import Trainer

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        trainer = Trainer(model, ASBRConfig(), small_training_config, device="cpu")
        assert isinstance(trainer.asbr_loss, ASBRLoss)


# ---------------------------------------------------------------------------
# PyTorchMLPWrapper (sklearn-compatible MLP)
# ---------------------------------------------------------------------------

class TestPyTorchMLPWrapper:
    def test_fit_predict_no_weights(self, synthetic_data):
        from asbr.baselines.pytorch_mlp_wrapper import PyTorchMLPWrapper

        X, y, _ = synthetic_data
        wrapper = PyTorchMLPWrapper(
            input_dim=5, hidden_dims=(8,), max_epochs=5, batch_size=64,
        )
        wrapper.fit(X, y.astype(int))

        y_pred = wrapper.predict(X)
        assert y_pred.shape == (len(X),)
        assert set(np.unique(y_pred)).issubset({0, 1})

        y_proba = wrapper.predict_proba(X)
        assert y_proba.shape == (len(X), 2)
        np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, atol=1e-6)

    def test_fit_with_sample_weights(self, synthetic_data):
        from asbr.baselines.pytorch_mlp_wrapper import PyTorchMLPWrapper

        X, y, _ = synthetic_data
        rng = np.random.RandomState(0)
        weights = rng.rand(len(X)).astype(np.float32) + 0.1

        wrapper = PyTorchMLPWrapper(
            input_dim=5, hidden_dims=(8,), max_epochs=5, batch_size=64,
        )
        wrapper.fit(X, y.astype(int), sample_weight=weights)

        y_pred = wrapper.predict(X)
        assert y_pred.shape == (len(X),)

    def test_sklearn_attributes_set(self, synthetic_data):
        from asbr.baselines.pytorch_mlp_wrapper import PyTorchMLPWrapper

        X, y, _ = synthetic_data
        wrapper = PyTorchMLPWrapper(input_dim=5, max_epochs=2)
        wrapper.fit(X, y.astype(int))

        assert hasattr(wrapper, "classes_")
        assert list(wrapper.classes_) == [0, 1]
        assert wrapper.n_features_in_ == 5

    def test_get_params_set_params(self):
        """Sklearn clone() compatibility via get_params/set_params."""
        from asbr.baselines.pytorch_mlp_wrapper import PyTorchMLPWrapper

        wrapper = PyTorchMLPWrapper(input_dim=10, hidden_dims=(32,), lr=0.01)
        params = wrapper.get_params()
        assert params["input_dim"] == 10
        assert params["hidden_dims"] == (32,)
        assert params["lr"] == 0.01

        wrapper.set_params(lr=0.005)
        assert wrapper.lr == 0.005


# ---------------------------------------------------------------------------
# ExponGrad with MLP base estimator
# ---------------------------------------------------------------------------

class TestExponGradMLP:
    @pytest.fixture(autouse=True)
    def _skip_if_no_fairlearn(self):
        pytest.importorskip("fairlearn")

    def test_fit_predict_with_mlp_base(self, synthetic_data):
        from asbr.baselines.expgrad import ExponGradWrapper, ExponGradConfig
        from asbr.baselines.pytorch_mlp_wrapper import PyTorchMLPWrapper

        X, y, A = synthetic_data
        base = PyTorchMLPWrapper(
            input_dim=5, hidden_dims=(8,), max_epochs=3, batch_size=64,
        )
        wrapper = ExponGradWrapper(
            ExponGradConfig(max_iter=5), base_estimator=base,
        )
        wrapper.fit(X, y.astype(int), A, intersection_level=1)

        y_pred = wrapper.predict(X)
        assert y_pred.shape == (len(X),)
        assert set(np.unique(y_pred)).issubset({0, 1})

        y_prob = wrapper.predict_proba(X)
        assert y_prob.shape == (len(X),)

    def test_custom_base_is_used(self):
        """base_estimator parameter is forwarded to ExponentiatedGradient."""
        from asbr.baselines.expgrad import ExponGradWrapper, ExponGradConfig
        from asbr.baselines.pytorch_mlp_wrapper import PyTorchMLPWrapper

        base = PyTorchMLPWrapper(input_dim=5, max_epochs=1)
        wrapper = ExponGradWrapper(ExponGradConfig(), base_estimator=base)
        assert wrapper.base_estimator is base


# ---------------------------------------------------------------------------
# AIF360 ExponentiatedGradientReduction wrapper
# ---------------------------------------------------------------------------

class TestAIF360ExponGradWrapper:
    @pytest.fixture(autouse=True)
    def _skip_if_no_aif360(self):
        pytest.importorskip("aif360")

    def test_fit_predict(self, synthetic_data):
        from asbr.baselines.aif360_expgrad import AIF360ExponGradWrapper, AIF360ExponGradConfig

        X, y, A = synthetic_data
        wrapper = AIF360ExponGradWrapper(AIF360ExponGradConfig(max_iter=10))
        wrapper.fit(X, y.astype(int), A, intersection_level=1)

        y_pred = wrapper.predict(X)
        assert y_pred.shape == (len(X),)
        assert set(np.unique(y_pred)).issubset({0, 1})

        y_prob = wrapper.predict_proba(X)
        assert y_prob.shape == (len(X),)
        assert np.all(y_prob >= -1e-10)
        assert np.all(y_prob <= 1 + 1e-10)

    def test_reduces_disparity_vs_unconstrained(self, synthetic_data):
        """AIF360 ExponGrad should reduce max SP violation compared to unconstrained LR."""
        from sklearn.linear_model import LogisticRegression
        from asbr.baselines.aif360_expgrad import AIF360ExponGradWrapper, AIF360ExponGradConfig
        from asbr.fairness.metrics import max_sp_violation

        X, y, A = synthetic_data
        y_int = y.astype(int)
        masks = build_subgroup_masks(A, level=1)

        # Unconstrained
        lr = LogisticRegression(solver="liblinear", max_iter=1000)
        lr.fit(X, y_int)
        y_pred_lr = lr.predict(X)
        sp_lr = max_sp_violation(y_pred_lr, masks)

        # AIF360 ExponGrad
        wrapper = AIF360ExponGradWrapper(AIF360ExponGradConfig(eps=0.01, max_iter=30))
        wrapper.fit(X, y_int, A, intersection_level=1)
        y_pred_eg = wrapper.predict(X)
        sp_eg = max_sp_violation(y_pred_eg, masks)

        assert sp_eg <= sp_lr + 0.05

    def test_predict_before_fit_raises(self):
        from asbr.baselines.aif360_expgrad import AIF360ExponGradWrapper

        wrapper = AIF360ExponGradWrapper()
        with pytest.raises(RuntimeError, match="not fitted"):
            wrapper.predict(np.zeros((10, 5)))
        with pytest.raises(RuntimeError, match="not fitted"):
            wrapper.predict_proba(np.zeros((10, 5)))

    def test_eps_controls_tradeoff(self, synthetic_data):
        """Smaller eps should enforce tighter fairness constraints."""
        from asbr.baselines.aif360_expgrad import AIF360ExponGradWrapper, AIF360ExponGradConfig
        from asbr.fairness.metrics import max_sp_violation

        X, y, A = synthetic_data
        y_int = y.astype(int)
        masks = build_subgroup_masks(A, level=1)

        # Tight constraint
        wrapper_tight = AIF360ExponGradWrapper(AIF360ExponGradConfig(eps=0.001, max_iter=50))
        wrapper_tight.fit(X, y_int, A, intersection_level=1)
        sp_tight = max_sp_violation(wrapper_tight.predict(X), masks)

        # Loose constraint
        wrapper_loose = AIF360ExponGradWrapper(AIF360ExponGradConfig(eps=0.5, max_iter=50))
        wrapper_loose.fit(X, y_int, A, intersection_level=1)
        sp_loose = max_sp_violation(wrapper_loose.predict(X), masks)

        # Tight constraint should have less or equal disparity
        assert sp_tight <= sp_loose + 0.05

    def test_intersectional_level(self, synthetic_data):
        """Should work with different intersection levels."""
        from asbr.baselines.aif360_expgrad import AIF360ExponGradWrapper, AIF360ExponGradConfig

        X, y, A = synthetic_data
        for level in [1, 2]:
            wrapper = AIF360ExponGradWrapper(AIF360ExponGradConfig(max_iter=5))
            wrapper.fit(X, y.astype(int), A, intersection_level=level)
            y_pred = wrapper.predict(X)
            assert y_pred.shape == (len(X),)


# ---------------------------------------------------------------------------
# Adversarial Debiasing wrapper
# ---------------------------------------------------------------------------

class TestAdversarialDebiasingWrapper:
    @pytest.fixture(autouse=True)
    def _skip_if_no_fairlearn(self):
        pytest.importorskip("fairlearn")

    def test_fit_predict(self, synthetic_data):
        from asbr.baselines.adversarial import AdversarialDebiasingWrapper, AdversarialDebiasingConfig

        X, y, A = synthetic_data
        wrapper = AdversarialDebiasingWrapper(
            AdversarialDebiasingConfig(
                epochs=5, predictor_model=[8, "relu"], adversary_model=[3, "relu"],
            )
        )
        wrapper.fit(X, y.astype(int), A, intersection_level=1)

        y_pred = wrapper.predict(X)
        assert y_pred.shape == (len(X),)
        assert set(np.unique(y_pred)).issubset({0, 1})

        y_prob = wrapper.predict_proba(X)
        assert y_prob.shape == (len(X),)
        assert np.all(y_prob >= -1e-10)
        assert np.all(y_prob <= 1 + 1e-10)

    def test_reduces_disparity_vs_unconstrained(self, synthetic_data):
        """High alpha adversarial debiasing should reduce SP vs unconstrained LR."""
        from sklearn.linear_model import LogisticRegression
        from asbr.baselines.adversarial import AdversarialDebiasingWrapper, AdversarialDebiasingConfig
        from asbr.fairness.metrics import max_sp_violation

        X, y, A = synthetic_data
        y_int = y.astype(int)
        masks = build_subgroup_masks(A, level=1)

        # Unconstrained
        lr = LogisticRegression(solver="liblinear", max_iter=1000)
        lr.fit(X, y_int)
        y_pred_lr = lr.predict(X)
        sp_lr = max_sp_violation(y_pred_lr, masks)

        # Adversarial Debiasing with high alpha
        wrapper = AdversarialDebiasingWrapper(
            AdversarialDebiasingConfig(
                alpha=10.0, epochs=30, predictor_model=[16, "relu"],
            )
        )
        wrapper.fit(X, y_int, A, intersection_level=1)
        y_pred_ad = wrapper.predict(X)
        sp_ad = max_sp_violation(y_pred_ad, masks)

        # Should not be worse (allow tolerance for stochastic training)
        assert sp_ad <= sp_lr + 0.10

    def test_predict_before_fit_raises(self):
        from asbr.baselines.adversarial import AdversarialDebiasingWrapper

        wrapper = AdversarialDebiasingWrapper()
        with pytest.raises(RuntimeError, match="not fitted"):
            wrapper.predict(np.zeros((10, 5)))
        with pytest.raises(RuntimeError, match="not fitted"):
            wrapper.predict_proba(np.zeros((10, 5)))

    def test_alpha_controls_tradeoff(self, synthetic_data):
        """Higher alpha should produce lower or equal SP disparity."""
        from asbr.baselines.adversarial import AdversarialDebiasingWrapper, AdversarialDebiasingConfig
        from asbr.fairness.metrics import max_sp_violation

        X, y, A = synthetic_data
        y_int = y.astype(int)
        masks = build_subgroup_masks(A, level=1)

        # Low alpha
        wrapper_low = AdversarialDebiasingWrapper(
            AdversarialDebiasingConfig(
                alpha=0.01, epochs=20, predictor_model=[8, "relu"],
            )
        )
        wrapper_low.fit(X, y_int, A, intersection_level=1)
        sp_low = max_sp_violation(wrapper_low.predict(X), masks)

        # High alpha
        wrapper_high = AdversarialDebiasingWrapper(
            AdversarialDebiasingConfig(
                alpha=50.0, epochs=20, predictor_model=[8, "relu"],
            )
        )
        wrapper_high.fit(X, y_int, A, intersection_level=1)
        sp_high = max_sp_violation(wrapper_high.predict(X), masks)

        # High alpha should enforce tighter fairness (allow tolerance)
        assert sp_high <= sp_low + 0.10


# ---------------------------------------------------------------------------
# Prejudice Remover wrapper
# ---------------------------------------------------------------------------

class TestPrejudiceRemoverWrapper:
    @pytest.fixture(autouse=True)
    def _skip_if_no_aif360(self):
        pytest.importorskip("aif360")

    def test_fit_predict(self, synthetic_data):
        from asbr.baselines.prejudice_remover import PrejudiceRemoverWrapper, PrejudiceRemoverConfig

        X, y, A = synthetic_data
        wrapper = PrejudiceRemoverWrapper(PrejudiceRemoverConfig(eta=1.0))
        wrapper.fit(X, y.astype(int), A, intersection_level=1)

        y_pred = wrapper.predict(X)
        assert y_pred.shape == (len(X),)
        assert set(np.unique(y_pred)).issubset({0, 1})

        y_prob = wrapper.predict_proba(X)
        assert y_prob.shape == (len(X),)
        assert np.all(y_prob >= -1e-10)
        assert np.all(y_prob <= 1 + 1e-10)

    def test_reduces_disparity_vs_unconstrained(self, synthetic_data):
        """High eta PrejudiceRemover should reduce SP vs unconstrained LR."""
        from sklearn.linear_model import LogisticRegression
        from asbr.baselines.prejudice_remover import PrejudiceRemoverWrapper, PrejudiceRemoverConfig
        from asbr.fairness.metrics import max_sp_violation

        X, y, A = synthetic_data
        y_int = y.astype(int)
        masks = build_subgroup_masks(A, level=1)

        # Unconstrained
        lr = LogisticRegression(solver="liblinear", max_iter=1000)
        lr.fit(X, y_int)
        y_pred_lr = lr.predict(X)
        sp_lr = max_sp_violation(y_pred_lr, masks)

        # PrejudiceRemover with high eta
        wrapper = PrejudiceRemoverWrapper(PrejudiceRemoverConfig(eta=100.0))
        wrapper.fit(X, y_int, A, intersection_level=1)
        y_pred_pr = wrapper.predict(X)
        sp_pr = max_sp_violation(y_pred_pr, masks)

        assert sp_pr <= sp_lr + 0.05

    def test_predict_before_fit_raises(self):
        from asbr.baselines.prejudice_remover import PrejudiceRemoverWrapper

        wrapper = PrejudiceRemoverWrapper()
        with pytest.raises(RuntimeError, match="not fitted"):
            wrapper.predict(np.zeros((10, 5)))
        with pytest.raises(RuntimeError, match="not fitted"):
            wrapper.predict_proba(np.zeros((10, 5)))

    def test_eta_controls_tradeoff(self, synthetic_data):
        """Higher eta should produce lower or equal SP disparity."""
        from asbr.baselines.prejudice_remover import PrejudiceRemoverWrapper, PrejudiceRemoverConfig
        from asbr.fairness.metrics import max_sp_violation

        X, y, A = synthetic_data
        y_int = y.astype(int)
        masks = build_subgroup_masks(A, level=1)

        # Low eta
        wrapper_low = PrejudiceRemoverWrapper(PrejudiceRemoverConfig(eta=0.1))
        wrapper_low.fit(X, y_int, A, intersection_level=1)
        sp_low = max_sp_violation(wrapper_low.predict(X), masks)

        # High eta
        wrapper_high = PrejudiceRemoverWrapper(PrejudiceRemoverConfig(eta=100.0))
        wrapper_high.fit(X, y_int, A, intersection_level=1)
        sp_high = max_sp_violation(wrapper_high.predict(X), masks)

        # High eta should enforce tighter fairness (allow tolerance)
        assert sp_high <= sp_low + 0.05


class TestAIF360ExponGradMLP:
    @pytest.fixture(autouse=True)
    def _skip_if_no_aif360(self):
        pytest.importorskip("aif360")

    def test_fit_predict_with_mlp_base(self, synthetic_data):
        from asbr.baselines.aif360_expgrad import AIF360ExponGradWrapper, AIF360ExponGradConfig
        from asbr.baselines.pytorch_mlp_wrapper import PyTorchMLPWrapper

        X, y, A = synthetic_data
        base = PyTorchMLPWrapper(
            input_dim=5, hidden_dims=(8,), max_epochs=3, batch_size=64,
        )
        wrapper = AIF360ExponGradWrapper(
            AIF360ExponGradConfig(max_iter=5), base_estimator=base,
        )
        wrapper.fit(X, y.astype(int), A, intersection_level=1)

        y_pred = wrapper.predict(X)
        assert y_pred.shape == (len(X),)
        assert set(np.unique(y_pred)).issubset({0, 1})

        y_prob = wrapper.predict_proba(X)
        assert y_prob.shape == (len(X),)


# ---------------------------------------------------------------------------
# FairretNorm2Method
# ---------------------------------------------------------------------------

class TestFairretNorm2Method:
    @pytest.fixture(autouse=True)
    def _skip_if_no_fairret(self):
        pytest.importorskip("fairret")

    def test_train_predict(self, synthetic_data, small_training_config):
        """Basic fit/predict: correct shapes, probabilities in [0, 1]."""
        from asbr.methods.fairret_method import FairretNorm2Method

        X, y, A = synthetic_data
        n_train = 240
        method = FairretNorm2Method(lambd=1.0)
        method.train(
            X[:n_train], y[:n_train],
            X[n_train:], y[n_train:],
            A.iloc[:n_train].reset_index(drop=True),
            intersection_level=1,
            training_config=small_training_config,
        )
        y_prob, y_pred = method.predict(X[n_train:])
        assert y_prob.shape == (len(X) - n_train,)
        assert np.all((y_prob >= 0) & (y_prob <= 1))
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_predict_before_train_raises(self):
        """predict() before train() raises AttributeError."""
        from asbr.methods.fairret_method import FairretNorm2Method

        method = FairretNorm2Method()
        with pytest.raises(AttributeError):
            method.predict(np.zeros((10, 5)))

    def test_high_lambda_reduces_disparity(self, synthetic_data, small_training_config):
        """At high λ the mean SP on training data should be lower than at λ=0."""
        from asbr.methods.fairret_method import FairretNorm2Method
        from asbr.fairness.metrics import mean_sp_violation

        X, y, A = synthetic_data
        n_train = 240
        masks = build_subgroup_masks(A.iloc[:n_train].reset_index(drop=True), level=1)

        # No fairness pressure
        method_erm = FairretNorm2Method(lambd=0.0)
        method_erm.train(
            X[:n_train], y[:n_train],
            X[n_train:], y[n_train:],
            A.iloc[:n_train].reset_index(drop=True),
            intersection_level=1,
            training_config=TrainingConfig(hidden_dims=(8,), dropout=0.0, max_epochs=20),
        )
        _, y_pred_erm = method_erm.predict(X[:n_train])
        sp_erm = mean_sp_violation(y_pred_erm, masks)

        # High fairness pressure
        method_fair = FairretNorm2Method(lambd=10.0)
        method_fair.train(
            X[:n_train], y[:n_train],
            X[n_train:], y[n_train:],
            A.iloc[:n_train].reset_index(drop=True),
            intersection_level=1,
            training_config=TrainingConfig(hidden_dims=(8,), dropout=0.0, max_epochs=20),
        )
        _, y_pred_fair = method_fair.predict(X[:n_train])
        sp_fair = mean_sp_violation(y_pred_fair, masks)

        assert sp_fair <= sp_erm + 0.05


# ---------------------------------------------------------------------------
# FairretNorm1Method
# ---------------------------------------------------------------------------

class TestFairretNorm1Method:
    @pytest.fixture(autouse=True)
    def _skip_if_no_fairret(self):
        pytest.importorskip("fairret")

    def test_train_predict(self, synthetic_data, small_training_config):
        """Basic fit/predict: correct shapes, probabilities in [0, 1]."""
        from asbr.methods.fairret_method import FairretNorm1Method

        X, y, A = synthetic_data
        n_train = 240
        method = FairretNorm1Method(lambd=1.0)
        method.train(
            X[:n_train], y[:n_train],
            X[n_train:], y[n_train:],
            A.iloc[:n_train].reset_index(drop=True),
            intersection_level=1,
            training_config=small_training_config,
        )
        y_prob, y_pred = method.predict(X[n_train:])
        assert y_prob.shape == (len(X) - n_train,)
        assert np.all((y_prob >= 0) & (y_prob <= 1))
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_predict_before_train_raises(self):
        """predict() before train() raises AttributeError."""
        from asbr.methods.fairret_method import FairretNorm1Method

        method = FairretNorm1Method()
        with pytest.raises(AttributeError):
            method.predict(np.zeros((10, 5)))

    def test_high_lambda_reduces_disparity(self, synthetic_data):
        """At high λ the mean SP should be lower than at λ=0."""
        from asbr.methods.fairret_method import FairretNorm1Method
        from asbr.fairness.metrics import mean_sp_violation

        X, y, A = synthetic_data
        n_train = 240
        masks = build_subgroup_masks(A.iloc[:n_train].reset_index(drop=True), level=1)
        cfg = TrainingConfig(hidden_dims=(8,), dropout=0.0, max_epochs=20)

        method_erm = FairretNorm1Method(lambd=0.0)
        method_erm.train(
            X[:n_train], y[:n_train], X[n_train:], y[n_train:],
            A.iloc[:n_train].reset_index(drop=True), intersection_level=1,
            training_config=cfg,
        )
        _, y_pred_erm = method_erm.predict(X[:n_train])
        sp_erm = mean_sp_violation(y_pred_erm, masks)

        method_fair = FairretNorm1Method(lambd=10.0)
        method_fair.train(
            X[:n_train], y[:n_train], X[n_train:], y[n_train:],
            A.iloc[:n_train].reset_index(drop=True), intersection_level=1,
            training_config=cfg,
        )
        _, y_pred_fair = method_fair.predict(X[:n_train])
        sp_fair = mean_sp_violation(y_pred_fair, masks)

        assert sp_fair <= sp_erm + 0.05


# ---------------------------------------------------------------------------
# EOPenaltyMethod — two-phase trainer (Blocker C control)
# ---------------------------------------------------------------------------

class TestEOPenaltyTwoPhase:
    """Smoke tests for EOPenaltyMethod with trainer='two_phase'."""

    def test_train_predict_shapes(self, synthetic_data):
        """Basic fit/predict: correct output shapes and valid probability range."""
        from asbr.methods.sp_penalty import EOPenaltyMethod

        X, y, A = synthetic_data
        n_train = 240
        cfg = TrainingConfig(hidden_dims=(8,), dropout=0.0, max_epochs=10, patience=3)

        method = EOPenaltyMethod(lambd=1.0, trainer="two_phase")
        method.train(
            X[:n_train], y[:n_train], X[n_train:], y[n_train:],
            A.iloc[:n_train].reset_index(drop=True), intersection_level=1,
            training_config=cfg,
        )
        y_prob, y_pred = method.predict(X[n_train:])
        assert y_pred.shape == (len(X) - n_train,)
        assert y_prob.shape == (len(X) - n_train,)
        assert np.all((y_prob >= 0) & (y_prob <= 1))
        assert set(np.unique(y_pred)).issubset({0, 1})

    def test_trainer_type_stored(self):
        """trainer kwarg is stored correctly on the method object."""
        from asbr.methods.sp_penalty import EOPenaltyMethod

        m = EOPenaltyMethod(trainer="two_phase")
        assert m._trainer_type == "two_phase"

        m_joint = EOPenaltyMethod()
        assert m_joint._trainer_type == "joint"

    def test_joint_default_unchanged(self, synthetic_data):
        """Default (joint) trainer still works after the refactor."""
        from asbr.methods.sp_penalty import EOPenaltyMethod

        X, y, A = synthetic_data
        n_train = 240
        cfg = TrainingConfig(hidden_dims=(8,), dropout=0.0, max_epochs=5, patience=3)

        method = EOPenaltyMethod(lambd=1.0)
        method.train(
            X[:n_train], y[:n_train], X[n_train:], y[n_train:],
            A.iloc[:n_train].reset_index(drop=True), intersection_level=1,
            training_config=cfg,
        )
        y_pred, _ = method.predict(X[n_train:])
        assert y_pred.shape == (len(X) - n_train,)
