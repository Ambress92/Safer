"""Tests for the ASBR fairness modules."""

import math

import numpy as np
import pytest
import torch

from asbr.fairness.metrics import (
    equal_opportunity,
    max_eo_violation,
    mean_eo_violation,
)

from asbr.fairness.soft_counts import compute_soft_counts, compute_hard_counts
from asbr.fairness.wald import (
    compute_gradient_vector,
    compute_covariance_matrix,
    compute_asymptotic_variance,
    compute_statistical_parity,
    compute_wald_statistic,
)
from asbr.fairness.bayesian import bayesian_sp_estimate
from asbr.fairness.hybrid_gradient import compute_transition_weight, compute_hybrid_estimate
from asbr.fairness.asbr_loss import ASBRLoss
from asbr.experiments.config import ASBRConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_scenario():
    """A simple scenario with 100 samples: 40 in group S, 60 in Sbar.

    Group S: 30/40 predicted positive (rate = 0.75)
    Group Sbar: 30/60 predicted positive (rate = 0.50)
    SP = 0.75 - 0.50 = 0.25
    """
    n = 100
    mask_S = torch.zeros(n, dtype=torch.bool)
    mask_S[:40] = True

    # Predictions: first 30 of S are positive, last 10 negative
    # First 30 of Sbar are positive, last 30 negative
    y_pred = torch.zeros(n)
    y_pred[:30] = 1.0   # S: 30 positive
    y_pred[40:70] = 1.0  # Sbar: 30 positive

    y_true = torch.zeros(n)
    y_true[:30] = 1.0
    y_true[40:70] = 1.0

    return y_pred, y_true, mask_S, n


@pytest.fixture
def balanced_scenario():
    """A balanced scenario where SP = 0 (no disparity).

    Group S: 20/40 positive (rate = 0.50)
    Group Sbar: 30/60 positive (rate = 0.50)
    """
    n = 100
    mask_S = torch.zeros(n, dtype=torch.bool)
    mask_S[:40] = True

    y_pred = torch.zeros(n)
    y_pred[:20] = 1.0   # S: 20/40 positive
    y_pred[40:70] = 1.0  # Sbar: 30/60 positive

    y_true = y_pred.clone()
    return y_pred, y_true, mask_S, n


# ---------------------------------------------------------------------------
# Soft Counts
# ---------------------------------------------------------------------------

class TestSoftCounts:
    def test_sum_to_one(self, simple_scenario):
        y_pred, _, mask_S, _ = simple_scenario
        p_hat = compute_soft_counts(y_pred, mask_S)
        assert torch.allclose(p_hat.sum(), torch.tensor(1.0), atol=1e-6)

    def test_correct_values_binary_predictions(self, simple_scenario):
        y_pred, _, mask_S, n = simple_scenario
        p_hat = compute_soft_counts(y_pred, mask_S)

        # p_{1,S} = 30/100, p_{0,S} = 10/100, p_{1,Sbar} = 30/100, p_{0,Sbar} = 30/100
        expected = torch.tensor([30.0 / n, 10.0 / n, 30.0 / n, 30.0 / n])
        assert torch.allclose(p_hat, expected, atol=1e-6)

    def test_hard_counts_match_for_binary_preds(self, simple_scenario):
        y_pred, _, mask_S, _ = simple_scenario
        counts = compute_hard_counts(y_pred, mask_S)
        expected = torch.tensor([30.0, 10.0, 30.0, 30.0])
        assert torch.allclose(counts, expected, atol=1e-6)

    def test_differentiable(self):
        """Soft counts should propagate gradients through y_pred."""
        y_pred = torch.tensor([0.7, 0.3, 0.8, 0.2], requires_grad=True)
        mask_S = torch.tensor([True, True, False, False])
        p_hat = compute_soft_counts(y_pred, mask_S)
        # p_hat.sum() == 1 always, so test a single component instead
        p_hat[0].backward()
        assert y_pred.grad is not None
        assert not torch.all(y_pred.grad == 0)


# ---------------------------------------------------------------------------
# Wald Components
# ---------------------------------------------------------------------------

class TestWaldComponents:
    def test_gradient_vector_shape(self, simple_scenario):
        y_pred, _, mask_S, _ = simple_scenario
        p_hat = compute_soft_counts(y_pred, mask_S)
        V = compute_gradient_vector(p_hat)
        assert V.shape == (4,)

    def test_covariance_matrix_shape_and_symmetry(self, simple_scenario):
        y_pred, _, mask_S, _ = simple_scenario
        p_hat = compute_soft_counts(y_pred, mask_S)
        Sigma4 = compute_covariance_matrix(p_hat)
        assert Sigma4.shape == (4, 4)
        assert torch.allclose(Sigma4, Sigma4.T, atol=1e-7)

    def test_covariance_positive_semidefinite(self, simple_scenario):
        y_pred, _, mask_S, _ = simple_scenario
        p_hat = compute_soft_counts(y_pred, mask_S)
        Sigma4 = compute_covariance_matrix(p_hat)
        eigenvalues = torch.linalg.eigvalsh(Sigma4)
        assert torch.all(eigenvalues >= -1e-7)

    def test_asymptotic_variance_positive(self, simple_scenario):
        y_pred, _, mask_S, _ = simple_scenario
        p_hat = compute_soft_counts(y_pred, mask_S)
        sigma2 = compute_asymptotic_variance(p_hat)
        assert sigma2.item() > 0

    def test_statistical_parity_value(self, simple_scenario):
        y_pred, _, mask_S, _ = simple_scenario
        p_hat = compute_soft_counts(y_pred, mask_S)
        sp = compute_statistical_parity(p_hat)
        # SP = 30/40 - 30/60 = 0.75 - 0.50 = 0.25
        assert abs(sp.item() - 0.25) < 1e-5

    def test_statistical_parity_zero_when_balanced(self, balanced_scenario):
        y_pred, _, mask_S, _ = balanced_scenario
        p_hat = compute_soft_counts(y_pred, mask_S)
        sp = compute_statistical_parity(p_hat)
        assert abs(sp.item()) < 1e-5

    def test_wald_statistic_sign(self, simple_scenario):
        y_pred, _, mask_S, n = simple_scenario
        p_hat = compute_soft_counts(y_pred, mask_S)
        sp = compute_statistical_parity(p_hat)
        sigma2 = compute_asymptotic_variance(p_hat)
        z_s = compute_wald_statistic(sp, sigma2, n)
        # SP > 0 so Z should be positive
        assert z_s.item() > 0

    def test_wald_differentiable(self):
        """Full chain from y_pred to Z_S should be differentiable."""
        y_pred = torch.tensor([0.8, 0.2, 0.6, 0.4, 0.5, 0.5], requires_grad=True)
        mask_S = torch.tensor([True, True, True, False, False, False])
        p_hat = compute_soft_counts(y_pred, mask_S)
        sp = compute_statistical_parity(p_hat)
        sigma2 = compute_asymptotic_variance(p_hat)
        z_s = compute_wald_statistic(sp, sigma2, 6)
        z_s.backward()
        assert y_pred.grad is not None


# ---------------------------------------------------------------------------
# Bayesian Estimator
# ---------------------------------------------------------------------------

class TestBayesianEstimator:
    def test_returns_two_tensors(self):
        counts = torch.tensor([10.0, 5.0, 20.0, 15.0])
        sp, sigma2 = bayesian_sp_estimate(counts)
        assert sp.shape == ()
        assert sigma2.shape == ()

    def test_variance_positive(self):
        counts = torch.tensor([10.0, 5.0, 20.0, 15.0])
        _, sigma2 = bayesian_sp_estimate(counts)
        assert sigma2.item() > 0

    def test_converges_to_frequentist_with_large_counts(self):
        """With large counts, the prior's influence vanishes."""
        counts = torch.tensor([3000.0, 1000.0, 3000.0, 3000.0])
        sp_bayes, _ = bayesian_sp_estimate(counts)
        # Frequentist: p = counts / sum = [0.3, 0.1, 0.3, 0.3]
        # SP = 0.3/0.4 - 0.3/0.6 = 0.75 - 0.50 = 0.25
        assert abs(sp_bayes.item() - 0.25) < 0.01

    def test_prior_smooths_small_counts(self):
        """With very small counts, the prior pulls toward a moderate SP."""
        counts_extreme = torch.tensor([5.0, 0.0, 0.0, 5.0])
        sp_bayes, _ = bayesian_sp_estimate(counts_extreme)
        # Without prior, SP would be 5/5 - 0/5 = 1.0
        # With uniform prior (1,1,1,1), counts become (6,1,1,6):
        # SP = 6/7 - 1/7 ≈ 0.714, pulled from 1.0
        assert sp_bayes.item() < 0.95  # Prior should moderate the extreme

    def test_variance_consistent_with_frequentist_scale(self):
        """Bayesian sigma2 should be on population-variance scale.

        With proportionally identical counts, the posterior mean p is the same,
        so the population-scale sigma2 = V^T (diag(p) - pp^T) V is identical.
        This confirms sigma2_bayes matches sigma2_freq's scale (not estimator-scale).
        """
        counts_small = torch.tensor([5.0, 5.0, 5.0, 5.0])
        counts_large = counts_small * 100
        _, sigma2_small = bayesian_sp_estimate(counts_small)
        _, sigma2_large = bayesian_sp_estimate(counts_large)
        # Population-scale variance is the same for proportional counts
        assert abs(sigma2_large.item() - sigma2_small.item()) < 0.01


# ---------------------------------------------------------------------------
# Hybrid Gradient
# ---------------------------------------------------------------------------

class TestHybridGradient:
    def test_transition_weight_small_group(self):
        beta = compute_transition_weight(n_S=5, n_tilde=30, kappa=0.5)
        assert beta.item() < 0.01  # Should be very Bayesian

    def test_transition_weight_large_group(self):
        beta = compute_transition_weight(n_S=100, n_tilde=30, kappa=0.5)
        assert beta.item() > 0.99  # Should be very frequentist

    def test_transition_weight_at_threshold(self):
        beta = compute_transition_weight(n_S=30, n_tilde=30, kappa=0.5)
        assert abs(beta.item() - 0.5) < 0.01  # At threshold, should be ~0.5

    def test_hybrid_returns_three_tensors(self, simple_scenario):
        y_pred, _, mask_S, n = simple_scenario
        p_hat = compute_soft_counts(y_pred, mask_S)
        counts = compute_hard_counts(y_pred, mask_S)
        sp, sigma2, z_s = compute_hybrid_estimate(p_hat, counts, n_S=40, n_total=n)
        assert sp.shape == ()
        assert sigma2.shape == ()
        assert z_s.shape == ()

    def test_force_frequentist_matches_wald(self, simple_scenario):
        """When force_frequentist=True, should match pure Wald computation."""
        y_pred, _, mask_S, n = simple_scenario
        p_hat = compute_soft_counts(y_pred, mask_S)
        counts = compute_hard_counts(y_pred, mask_S)

        sp_hybrid, _, z_hybrid = compute_hybrid_estimate(
            p_hat, counts, n_S=40, n_total=n, force_frequentist=True
        )
        sp_wald = compute_statistical_parity(p_hat)
        assert torch.allclose(sp_hybrid, sp_wald, atol=1e-6)


# ---------------------------------------------------------------------------
# ASBR Loss
# ---------------------------------------------------------------------------

class TestASBRLoss:
    def test_returns_loss_and_diagnostics(self, simple_scenario):
        y_pred, y_true, mask_S, _ = simple_scenario
        loss_fn = ASBRLoss()
        subgroups = {"group_S": mask_S}
        total_loss, diag = loss_fn(y_pred, y_true, subgroups)
        assert total_loss.shape == ()
        assert "group_S" in diag
        assert diag["group_S"].n_S == 40

    def test_loss_is_finite(self, simple_scenario):
        y_pred, y_true, mask_S, _ = simple_scenario
        loss_fn = ASBRLoss()
        subgroups = {"group_S": mask_S}
        total_loss, _ = loss_fn(y_pred, y_true, subgroups)
        assert torch.isfinite(total_loss)

    def test_near_zero_fairness_when_balanced(self, balanced_scenario):
        """When SP ≈ 0, the hinge should not activate (Z < z_alpha).

        With softplus hinge, the value is small but nonzero (softplus(-1.96)² ≈ 0.018).
        """
        y_pred, y_true, mask_S, _ = balanced_scenario
        config = ASBRConfig(lambd=1.0, alpha=0.05)
        loss_fn = ASBRLoss(config)
        subgroups = {"group_S": mask_S}
        _, diag = loss_fn(y_pred, y_true, subgroups)
        assert not diag["group_S"].active
        assert diag["group_S"].hinge_value < 0.05  # Small but nonzero with softplus

    def test_hinge_activates_for_large_disparity(self, simple_scenario):
        """With SP = 0.25 and n=100, the Z-score should exceed z_0.05."""
        y_pred, y_true, mask_S, _ = simple_scenario
        config = ASBRConfig(lambd=1.0, alpha=0.05)
        loss_fn = ASBRLoss(config)
        subgroups = {"group_S": mask_S}
        _, diag = loss_fn(y_pred, y_true, subgroups)
        assert diag["group_S"].active
        assert diag["group_S"].hinge_value > 0

    def test_backward_propagates(self):
        """Loss.backward() should produce non-zero gradients on model params."""
        from asbr.models.base_mlp import FairnessMLP

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        X = torch.randn(50, 5)
        y = torch.randint(0, 2, (50,)).float()
        mask = torch.zeros(50, dtype=torch.bool)
        mask[:20] = True

        loss_fn = ASBRLoss(ASBRConfig(lambd=1.0))
        y_pred = model(X)
        total_loss, _ = loss_fn(y_pred, y, {"test_group": mask})
        total_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_ablation_no_hinge_always_active(self, simple_scenario):
        """Without hinge, all subgroups should always contribute penalty."""
        y_pred, y_true, mask_S, _ = simple_scenario
        config = ASBRConfig(use_hinge=False)
        loss_fn = ASBRLoss(config)
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        assert diag["group_S"].active

    def test_ablation_no_size_weight(self, simple_scenario):
        """Without size-weight, omega = 1 regardless of group size."""
        y_pred, y_true, mask_S, _ = simple_scenario
        config_with = ASBRConfig(use_size_weight=True, lambd=1.0)
        config_without = ASBRConfig(use_size_weight=False, lambd=1.0)

        loss_with = ASBRLoss(config_with)
        loss_without = ASBRLoss(config_without)

        l1, _ = loss_with(y_pred, y_true, {"group_S": mask_S})
        l2, _ = loss_without(y_pred, y_true, {"group_S": mask_S})

        # With omega=log(41)≈3.71, fairness contribution is ~3.71x larger than omega=1
        # So loss_with should be larger (assuming fairness penalty is positive)
        # Just check they're different
        assert not torch.allclose(l1, l2)

    def test_skips_degenerate_subgroups(self):
        """Groups with n_S < 2 or n_Sbar < 2 should be skipped."""
        y_pred = torch.rand(10)
        y_true = torch.randint(0, 2, (10,)).float()
        # Only 1 member in group
        mask_tiny = torch.zeros(10, dtype=torch.bool)
        mask_tiny[0] = True

        loss_fn = ASBRLoss()
        _, diag = loss_fn(y_pred, y_true, {"tiny": mask_tiny})
        assert "tiny" not in diag

    def test_multiple_subgroups(self):
        """Loss should accumulate across multiple subgroups."""
        n = 200
        y_pred = torch.rand(n)
        y_true = torch.randint(0, 2, (n,)).float()

        mask_a = torch.zeros(n, dtype=torch.bool)
        mask_a[:50] = True
        mask_b = torch.zeros(n, dtype=torch.bool)
        mask_b[50:120] = True

        loss_fn = ASBRLoss(ASBRConfig(lambd=1.0))
        _, diag = loss_fn(y_pred, y_true, {"group_a": mask_a, "group_b": mask_b})
        assert "group_a" in diag
        assert "group_b" in diag


# ---------------------------------------------------------------------------
# Equal Opportunity evaluation metrics (metrics.py)
# ---------------------------------------------------------------------------

class TestEqualOpportunityMetrics:
    """Tests for equal_opportunity, max_eo_violation, mean_eo_violation."""

    def _make_arrays(self):
        """100 samples: 40 in S, 60 in Sbar.

        Ground-truth positives: first 20 of S, first 30 of Sbar.
        Predictions:            first 16 of S (TPR_S=0.80),
                                first 15 of Sbar (TPR_Sbar=0.50).
        EO = 0.80 - 0.50 = 0.30
        """
        n = 100
        mask_S = np.zeros(n, dtype=bool)
        mask_S[:40] = True

        y_true = np.zeros(n, dtype=int)
        y_true[:20] = 1      # 20 positives in S
        y_true[40:70] = 1    # 30 positives in Sbar

        y_pred = np.zeros(n, dtype=int)
        y_pred[:16] = 1      # 16/20 TP in S  → TPR_S = 0.80
        y_pred[40:55] = 1    # 15/30 TP in Sbar → TPR_Sbar = 0.50

        return y_pred, y_true, mask_S, n

    def test_equal_opportunity_value(self):
        y_pred, y_true, mask_S, _ = self._make_arrays()
        eo = equal_opportunity(y_pred, y_true, mask_S)
        assert eo is not None
        assert eo == pytest.approx(0.30, abs=1e-5)

    def test_eo_zero_when_tpr_equal(self):
        n = 80
        mask_S = np.zeros(n, dtype=bool)
        mask_S[:40] = True
        y_true = np.zeros(n, dtype=int)
        y_true[:20] = 1
        y_true[40:60] = 1
        y_pred = np.zeros(n, dtype=int)
        y_pred[:10] = 1    # 10/20 = 0.50
        y_pred[40:50] = 1  # 10/20 = 0.50
        eo = equal_opportunity(y_pred, y_true, mask_S)
        assert eo is not None
        assert eo == pytest.approx(0.0, abs=1e-5)

    def test_eo_returns_none_no_positives_in_S(self):
        n = 40
        mask_S = np.zeros(n, dtype=bool)
        mask_S[:10] = True
        y_true = np.zeros(n, dtype=int)
        y_true[10:] = 1  # positives only in Sbar
        y_pred = np.zeros(n, dtype=int)
        result = equal_opportunity(y_pred, y_true, mask_S)
        assert result is None

    def test_eo_returns_none_no_positives_in_Sbar(self):
        n = 40
        mask_S = np.zeros(n, dtype=bool)
        mask_S[:10] = True
        y_true = np.zeros(n, dtype=int)
        y_true[:10] = 1  # positives only in S
        y_pred = np.zeros(n, dtype=int)
        result = equal_opportunity(y_pred, y_true, mask_S)
        assert result is None

    def test_eo_can_be_negative(self):
        """EO is negative when TPR_S < TPR_Sbar."""
        n = 80
        mask_S = np.zeros(n, dtype=bool)
        mask_S[:40] = True
        y_true = np.zeros(n, dtype=int)
        y_true[:20] = 1
        y_true[40:60] = 1
        y_pred = np.zeros(n, dtype=int)
        y_pred[:5] = 1    # TPR_S  = 5/20 = 0.25
        y_pred[40:55] = 1  # TPR_Sbar = 15/20 = 0.75  → EO = -0.50
        eo = equal_opportunity(y_pred, y_true, mask_S)
        assert eo is not None
        assert eo == pytest.approx(-0.50, abs=1e-5)

    def test_max_eo_violation(self):
        y_pred, y_true, mask_S, _ = self._make_arrays()
        masks = {"group_A": mask_S, "group_B": ~mask_S}
        max_eo = max_eo_violation(y_pred, y_true, masks)
        # group_B mask = Sbar → EO(Sbar) = -EO(S) = -0.30, |EO| = 0.30
        assert max_eo == pytest.approx(0.30, abs=1e-5)

    def test_mean_eo_violation(self):
        y_pred, y_true, mask_S, n = self._make_arrays()
        masks = {"group_A": mask_S}
        mean_eo = mean_eo_violation(y_pred, y_true, masks)
        assert mean_eo == pytest.approx(0.30, abs=1e-5)

    def test_max_eo_zero_when_all_equal(self):
        n = 80
        mask_S = np.zeros(n, dtype=bool)
        mask_S[:40] = True
        y_true = np.zeros(n, dtype=int)
        y_true[:20] = 1
        y_true[40:60] = 1
        y_pred = np.zeros(n, dtype=int)
        y_pred[:10] = 1
        y_pred[40:50] = 1
        masks = {"g": mask_S}
        assert max_eo_violation(y_pred, y_true, masks) == pytest.approx(0.0, abs=1e-5)

    def test_eo_excludes_subgroups_with_no_positives(self):
        """Subgroups with undefined EO should not affect the aggregate."""
        n = 60
        mask_valid = np.zeros(n, dtype=bool)
        mask_valid[:20] = True
        mask_no_pos = np.zeros(n, dtype=bool)
        mask_no_pos[20:30] = True  # will have no positives

        y_true = np.zeros(n, dtype=int)
        y_true[:10] = 1
        y_true[30:50] = 1  # positives outside mask_no_pos

        y_pred = np.zeros(n, dtype=int)
        y_pred[:5] = 1    # TPR_valid_S = 5/10 = 0.50
        y_pred[30:40] = 1  # TPR in Sbar

        masks = {"valid": mask_valid, "no_pos": mask_no_pos}
        max_eo = max_eo_violation(y_pred, y_true, masks)
        mean_eo = mean_eo_violation(y_pred, y_true, masks)

        # Only "valid" contributes; "no_pos" is excluded (undefined EO)
        eo_valid = equal_opportunity(y_pred, y_true, mask_valid)
        assert eo_valid is not None
        assert max_eo == pytest.approx(abs(eo_valid), abs=1e-5)
        assert mean_eo == pytest.approx(abs(eo_valid), abs=1e-5)

    def test_mean_eo_empty_masks(self):
        """An empty subgroup dict should return 0.0."""
        y_pred = np.zeros(10, dtype=int)
        y_true = np.ones(10, dtype=int)
        assert mean_eo_violation(y_pred, y_true, {}) == 0.0
        assert max_eo_violation(y_pred, y_true, {}) == 0.0
