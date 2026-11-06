"""Tests for the LRT fairness module and ASBR-LRT loss."""

import math

import pytest
import torch

from asbr.fairness.lrt import (
    compute_lrt_statistic,
    compute_soft_rates,
    compute_eo_lrt_statistic,
    compute_eo_soft_rates,
)
from asbr.fairness.asbr_lrt_loss import ASBRLRTLoss, LRTDiagnostics
from asbr.fairness.wald import (
    compute_asymptotic_variance,
    compute_statistical_parity,
    compute_wald_statistic,
)
from asbr.fairness.soft_counts import compute_soft_counts
from asbr.experiments.config import ASBRConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_scenario():
    """100 samples: 40 in S, 60 in Sbar.

    Group S: 30/40 predicted positive (rate = 0.75)
    Group Sbar: 30/60 predicted positive (rate = 0.50)
    SP = 0.25
    """
    n = 100
    mask_S = torch.zeros(n, dtype=torch.bool)
    mask_S[:40] = True

    y_pred = torch.zeros(n)
    y_pred[:30] = 1.0   # S: 30 positive
    y_pred[40:70] = 1.0  # Sbar: 30 positive

    y_true = torch.zeros(n)
    y_true[:30] = 1.0
    y_true[40:70] = 1.0

    return y_pred, y_true, mask_S, n


@pytest.fixture
def balanced_scenario():
    """SP = 0: both groups have rate 0.50."""
    n = 100
    mask_S = torch.zeros(n, dtype=torch.bool)
    mask_S[:40] = True

    y_pred = torch.zeros(n)
    y_pred[:20] = 1.0   # S: 20/40 = 0.50
    y_pred[40:70] = 1.0  # Sbar: 30/60 = 0.50

    y_true = y_pred.clone()
    return y_pred, y_true, mask_S, n


@pytest.fixture
def soft_scenario():
    """Scenario with soft (non-binary) predictions for gradient tests."""
    torch.manual_seed(42)
    n = 80
    mask_S = torch.zeros(n, dtype=torch.bool)
    mask_S[:30] = True

    y_pred = torch.sigmoid(torch.randn(n))
    y_true = torch.randint(0, 2, (n,)).float()
    return y_pred, y_true, mask_S, n


# ---------------------------------------------------------------------------
# compute_soft_rates
# ---------------------------------------------------------------------------

class TestSoftRates:
    def test_returns_three_tensors(self, simple_scenario):
        y_pred, _, mask_S, _ = simple_scenario
        p_S, p_Sbar, p_pooled = compute_soft_rates(y_pred, mask_S)
        assert p_S.shape == ()
        assert p_Sbar.shape == ()
        assert p_pooled.shape == ()

    def test_correct_values(self, simple_scenario):
        y_pred, _, mask_S, _ = simple_scenario
        p_S, p_Sbar, p_pooled = compute_soft_rates(y_pred, mask_S)
        assert p_S.item() == pytest.approx(0.75, abs=1e-5)
        assert p_Sbar.item() == pytest.approx(0.50, abs=1e-5)
        assert p_pooled.item() == pytest.approx(0.60, abs=1e-5)

    def test_balanced_equal_rates(self, balanced_scenario):
        y_pred, _, mask_S, _ = balanced_scenario
        p_S, p_Sbar, p_pooled = compute_soft_rates(y_pred, mask_S)
        assert p_S.item() == pytest.approx(p_Sbar.item(), abs=1e-5)
        assert p_pooled.item() == pytest.approx(0.50, abs=1e-5)

    def test_clamped_in_bounds(self):
        """Even with extreme predictions, rates stay in (epsilon, 1-epsilon)."""
        n = 20
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:10] = True
        y_pred = torch.zeros(n)  # All zeros
        eps = 1e-8
        p_S, p_Sbar, p_pooled = compute_soft_rates(y_pred, mask_S, epsilon=eps)
        # Allow small float tolerance on clamp boundary
        assert p_S.item() >= eps - 1e-15
        assert p_Sbar.item() >= eps - 1e-15
        assert p_S.item() <= 1.0 - eps + 1e-15
        assert p_Sbar.item() <= 1.0 - eps + 1e-15


# ---------------------------------------------------------------------------
# compute_lrt_statistic
# ---------------------------------------------------------------------------

class TestLRTStatistic:
    def test_non_negative(self, simple_scenario):
        y_pred, _, mask_S, _ = simple_scenario
        lrt = compute_lrt_statistic(y_pred, mask_S)
        assert lrt.item() >= 0.0

    def test_zero_when_balanced(self, balanced_scenario):
        """LRT should be ~0 when groups have equal rates."""
        y_pred, _, mask_S, _ = balanced_scenario
        lrt = compute_lrt_statistic(y_pred, mask_S)
        assert lrt.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_when_disparate(self, simple_scenario):
        """LRT should be positive when groups have different rates."""
        y_pred, _, mask_S, _ = simple_scenario
        lrt = compute_lrt_statistic(y_pred, mask_S)
        assert lrt.item() > 1.0  # Should be substantial with SP=0.25, n=100

    def test_increases_with_disparity(self):
        """Larger disparities should produce larger LRT statistics."""
        n = 200
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:100] = True

        # Small disparity: 55% vs 50%
        y_small = torch.zeros(n)
        y_small[:55] = 1.0
        y_small[100:150] = 1.0

        # Large disparity: 80% vs 30%
        y_large = torch.zeros(n)
        y_large[:80] = 1.0
        y_large[100:130] = 1.0

        lrt_small = compute_lrt_statistic(y_small, mask_S)
        lrt_large = compute_lrt_statistic(y_large, mask_S)
        assert lrt_large.item() > lrt_small.item()

    def test_scalar_output(self, simple_scenario):
        y_pred, _, mask_S, _ = simple_scenario
        lrt = compute_lrt_statistic(y_pred, mask_S)
        assert lrt.dim() == 0

    def test_differentiable(self, soft_scenario):
        """LRT statistic should propagate gradients through y_pred."""
        y_pred, _, mask_S, _ = soft_scenario
        y_pred = y_pred.clone().requires_grad_(True)
        lrt = compute_lrt_statistic(y_pred, mask_S)
        lrt.backward()
        assert y_pred.grad is not None
        assert not torch.all(y_pred.grad == 0)


# ---------------------------------------------------------------------------
# Asymptotic concordance: sqrt(Lambda) ≈ |Z_S| for large samples
# ---------------------------------------------------------------------------

class TestAsymptoticConcordance:
    def test_sqrt_lambda_close_to_abs_z_large_n(self):
        """For large n with moderate proportions, sqrt(Lambda) ≈ |Z_S|."""
        torch.manual_seed(123)
        n = 5000
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:2000] = True

        # S: rate 0.60, Sbar: rate 0.50
        y_pred = torch.zeros(n)
        y_pred[:1200] = 1.0   # 1200/2000 = 0.60
        y_pred[2000:3500] = 1.0  # 1500/3000 = 0.50

        # LRT
        lrt = compute_lrt_statistic(y_pred, mask_S)
        sqrt_lrt = torch.sqrt(lrt).item()

        # Wald
        p_hat = compute_soft_counts(y_pred, mask_S)
        sp = compute_statistical_parity(p_hat)
        sigma2 = compute_asymptotic_variance(p_hat)
        z_s = compute_wald_statistic(sp, sigma2, n)
        abs_z = abs(z_s.item())

        # Should agree within ~5% for n=5000
        assert abs(sqrt_lrt - abs_z) / abs_z < 0.05

    def test_diverge_for_small_extreme_groups(self):
        """For small groups with extreme proportions, LRT and Wald may differ."""
        n = 50
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:5] = True  # Only 5 in S

        # S: rate ~1.0 (all positive), Sbar: rate ~0.50
        y_pred = torch.zeros(n)
        y_pred[:5] = 0.99   # S: nearly all positive
        y_pred[5:28] = 1.0  # Sbar: ~23/45 = 0.51

        lrt = compute_lrt_statistic(y_pred, mask_S)
        sqrt_lrt = torch.sqrt(lrt).item()

        p_hat = compute_soft_counts(y_pred, mask_S)
        sp = compute_statistical_parity(p_hat)
        sigma2 = compute_asymptotic_variance(p_hat)
        z_s = compute_wald_statistic(sp, sigma2, n)
        abs_z = abs(z_s.item())

        # Just verify both are computed without errors — divergence is expected
        assert sqrt_lrt > 0
        assert abs_z > 0


# ---------------------------------------------------------------------------
# ASBRLRTLoss
# ---------------------------------------------------------------------------

class TestASBRLRTLoss:
    def test_returns_loss_and_diagnostics(self, simple_scenario):
        y_pred, y_true, mask_S, _ = simple_scenario
        loss_fn = ASBRLRTLoss()
        subgroups = {"group_S": mask_S}
        total_loss, diag = loss_fn(y_pred, y_true, subgroups)
        assert total_loss.shape == ()
        assert "group_S" in diag
        assert diag["group_S"].n_S == 40

    def test_loss_is_finite(self, simple_scenario):
        y_pred, y_true, mask_S, _ = simple_scenario
        loss_fn = ASBRLRTLoss()
        subgroups = {"group_S": mask_S}
        total_loss, _ = loss_fn(y_pred, y_true, subgroups)
        assert torch.isfinite(total_loss)

    def test_diagnostics_fields(self, simple_scenario):
        y_pred, y_true, mask_S, _ = simple_scenario
        loss_fn = ASBRLRTLoss()
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        d = diag["group_S"]
        assert isinstance(d, LRTDiagnostics)
        assert d.n_S == 40
        assert isinstance(d.sp, float)
        assert isinstance(d.lambda_stat, float)
        assert isinstance(d.sqrt_lambda, float)
        assert isinstance(d.active, bool)
        assert isinstance(d.hinge_value, float)
        assert d.lambda_stat >= 0
        assert d.sqrt_lambda >= 0

    def test_hinge_activates_for_large_disparity(self, simple_scenario):
        y_pred, y_true, mask_S, _ = simple_scenario
        config = ASBRConfig(lambd=1.0, alpha=0.05)
        loss_fn = ASBRLRTLoss(config)
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        assert diag["group_S"].active
        assert diag["group_S"].hinge_value > 0

    def test_hinge_inactive_when_balanced(self, balanced_scenario):
        """With softplus hinge, value is small but nonzero when below threshold."""
        y_pred, y_true, mask_S, _ = balanced_scenario
        config = ASBRConfig(lambd=1.0, alpha=0.05)
        loss_fn = ASBRLRTLoss(config)
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        assert not diag["group_S"].active
        assert diag["group_S"].hinge_value < 0.05  # Small but nonzero with softplus

    def test_backward_propagates(self):
        """Loss.backward() should produce non-zero gradients on model params."""
        from asbr.models.base_mlp import FairnessMLP

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        X = torch.randn(50, 5)
        y = torch.randint(0, 2, (50,)).float()
        mask = torch.zeros(50, dtype=torch.bool)
        mask[:20] = True

        loss_fn = ASBRLRTLoss(ASBRConfig(lambd=1.0))
        y_pred = model(X)
        total_loss, _ = loss_fn(y_pred, y, {"test_group": mask})
        total_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_skips_degenerate_subgroups(self):
        y_pred = torch.rand(10)
        y_true = torch.randint(0, 2, (10,)).float()
        mask_tiny = torch.zeros(10, dtype=torch.bool)
        mask_tiny[0] = True

        loss_fn = ASBRLRTLoss()
        _, diag = loss_fn(y_pred, y_true, {"tiny": mask_tiny})
        assert "tiny" not in diag

    def test_multiple_subgroups(self):
        n = 200
        torch.manual_seed(0)
        y_pred = torch.rand(n)
        y_true = torch.randint(0, 2, (n,)).float()

        mask_a = torch.zeros(n, dtype=torch.bool)
        mask_a[:50] = True
        mask_b = torch.zeros(n, dtype=torch.bool)
        mask_b[50:120] = True

        loss_fn = ASBRLRTLoss(ASBRConfig(lambd=1.0))
        _, diag = loss_fn(y_pred, y_true, {"group_a": mask_a, "group_b": mask_b})
        assert "group_a" in diag
        assert "group_b" in diag

    def test_ablation_no_hinge_always_active(self, simple_scenario):
        y_pred, y_true, mask_S, _ = simple_scenario
        config = ASBRConfig(use_hinge=False)
        loss_fn = ASBRLRTLoss(config)
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        assert diag["group_S"].active

    def test_ablation_no_hinge_uses_raw_lambda(self, simple_scenario):
        """Without hinge, penalty is raw Lambda (not sqrt-hinge)."""
        y_pred, y_true, mask_S, _ = simple_scenario
        config = ASBRConfig(use_hinge=False, use_size_weight=False, lambd=1.0)
        loss_fn = ASBRLRTLoss(config)
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        # hinge_value should equal lambda_stat when hinge is off
        assert diag["group_S"].hinge_value == pytest.approx(
            diag["group_S"].lambda_stat, abs=1e-5
        )

    def test_ablation_no_size_weight(self, simple_scenario):
        y_pred, y_true, mask_S, _ = simple_scenario
        config_with = ASBRConfig(use_size_weight=True, lambd=1.0)
        config_without = ASBRConfig(use_size_weight=False, lambd=1.0)

        loss_with = ASBRLRTLoss(config_with)
        loss_without = ASBRLRTLoss(config_without)

        l1, _ = loss_with(y_pred, y_true, {"group_S": mask_S})
        l2, _ = loss_without(y_pred, y_true, {"group_S": mask_S})
        assert not torch.allclose(l1, l2)

    def test_compute_fairness_loss_standalone(self, simple_scenario):
        """compute_fairness_loss can be called without prediction loss."""
        y_pred, _, mask_S, _ = simple_scenario
        loss_fn = ASBRLRTLoss()
        fair_loss, diag = loss_fn.compute_fairness_loss(y_pred, {"group_S": mask_S})
        assert torch.isfinite(fair_loss)
        assert fair_loss.item() >= 0

    def test_sp_diagnostic_correct(self, simple_scenario):
        """The SP diagnostic should match the raw statistical parity."""
        y_pred, y_true, mask_S, _ = simple_scenario
        loss_fn = ASBRLRTLoss()
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        # SP = 0.75 - 0.50 = 0.25
        assert diag["group_S"].sp == pytest.approx(0.25, abs=1e-4)

    def test_lambda_zero_when_equal_rates(self, balanced_scenario):
        """Lambda should be ~0 when SP = 0."""
        y_pred, y_true, mask_S, _ = balanced_scenario
        loss_fn = ASBRLRTLoss()
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        assert diag["group_S"].lambda_stat == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Gated-SP variant (use_sp_magnitude=True)
# ---------------------------------------------------------------------------

class TestGatedSP:
    """Tests for use_sp_magnitude=True: sigmoid gate, SP² magnitude."""

    def _gated_config(self, **kwargs) -> ASBRConfig:
        defaults = dict(lambd=1.0, alpha=0.05, use_hinge=True, use_sp_magnitude=True,
                        use_hybrid=False, use_size_weight=False)
        defaults.update(kwargs)
        return ASBRConfig(**defaults)

    def test_gated_sp_active_for_large_disparity(self, simple_scenario):
        """With SP=0.25 and n=100, gate > 0 and hinge_value > 0."""
        y_pred, y_true, mask_S, _ = simple_scenario
        loss_fn = ASBRLRTLoss(self._gated_config())
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        d = diag["group_S"]
        assert d.gate_value > 0.0
        assert d.hinge_value > 0.0
        assert d.active

    def test_gated_sp_inactive_for_zero_disparity(self, balanced_scenario):
        """With SP≈0, hinge_value ≈ 0 (gate is small and SP²=0)."""
        y_pred, y_true, mask_S, _ = balanced_scenario
        loss_fn = ASBRLRTLoss(self._gated_config())
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        d = diag["group_S"]
        assert d.hinge_value == pytest.approx(0.0, abs=1e-5)

    def test_gated_sp_magnitude_is_sp_squared_when_gate_saturated(self):
        """When sqrt(Lambda) >> z_alpha, gate ≈ 1 and hinge ≈ SP²."""
        # n=5000, SP=0.25 → sqrt_lambda >> 1.96
        n = 5000
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:2500] = True
        y_pred = torch.zeros(n)
        y_pred[:625] = 1.0      # S: 625/2500 = 0.25
        y_pred[2500:5000] = 0.0  # Sbar: 0/2500 = 0.0 → SP = 0.25

        # Use kappa=10 for sharp gate saturation
        config = ASBRConfig(lambd=1.0, alpha=0.05, kappa=10.0,
                            use_hinge=True, use_sp_magnitude=True,
                            use_hybrid=False, use_size_weight=False)
        loss_fn = ASBRLRTLoss(config)
        _, diag = loss_fn(y_pred, y_pred, {"group_S": mask_S})
        d = diag["group_S"]
        assert d.gate_value == pytest.approx(1.0, abs=1e-3)
        expected_sp2 = d.sp ** 2
        assert d.hinge_value == pytest.approx(expected_sp2, rel=0.01)

    def test_gated_sp_backward_propagates(self):
        """total_loss.backward() succeeds and produces non-zero gradients."""
        from asbr.models.base_mlp import FairnessMLP

        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        X = torch.randn(80, 5)
        y = torch.randint(0, 2, (80,)).float()
        mask = torch.zeros(80, dtype=torch.bool)
        mask[:30] = True

        config = ASBRConfig(lambd=1.0, use_hinge=True, use_sp_magnitude=True,
                            use_hybrid=False, use_size_weight=False)
        loss_fn = ASBRLRTLoss(config)
        y_pred = model(X)
        total_loss, _ = loss_fn(y_pred, y, {"group": mask})
        total_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_gated_sp_less_amplification_than_hinge(self):
        """Gated-SP penalty ratio (large/small n) << original hinge ratio.

        Both groups have the same SP=0.25. Larger n makes sqrt(Lambda)
        proportionally larger, causing the original hinge to grow as n,
        while the gated-SP hinge stays bounded (gate saturates at 1).
        """
        sp = 0.25

        def make_scenario(n_S: int):
            n_Sbar = n_S
            n = n_S + n_Sbar
            mask_S = torch.zeros(n, dtype=torch.bool)
            mask_S[:n_S] = True
            y_pred = torch.zeros(n)
            # S: rate = sp, Sbar: rate = 0
            y_pred[:int(n_S * sp)] = 1.0
            return y_pred, mask_S

        y_small, mask_small = make_scenario(n_S=50)
        y_large, mask_large = make_scenario(n_S=500)

        # Original hinge
        hinge_config = ASBRConfig(lambd=1.0, alpha=0.05, use_hinge=True,
                                   use_sp_magnitude=False, use_hybrid=False,
                                   use_size_weight=False)
        loss_hinge = ASBRLRTLoss(hinge_config)
        dummy_y = torch.zeros_like(y_small)
        _, d_small_h = loss_hinge(y_small, dummy_y, {"g": mask_small})
        dummy_y_l = torch.zeros_like(y_large)
        _, d_large_h = loss_hinge(y_large, dummy_y_l, {"g": mask_large})

        # Gated-SP
        gated_config = ASBRConfig(lambd=1.0, alpha=0.05, kappa=0.5,
                                   use_hinge=True, use_sp_magnitude=True,
                                   use_hybrid=False, use_size_weight=False)
        loss_gated = ASBRLRTLoss(gated_config)
        _, d_small_g = loss_gated(y_small, dummy_y, {"g": mask_small})
        _, d_large_g = loss_gated(y_large, dummy_y_l, {"g": mask_large})

        hinge_small = d_small_h["g"].hinge_value
        hinge_large = d_large_h["g"].hinge_value
        gated_small = d_small_g["g"].hinge_value
        gated_large = d_large_g["g"].hinge_value

        hinge_ratio = hinge_large / hinge_small
        gated_ratio = gated_large / gated_small

        # Original hinge amplifies with n (ratio >> 1)
        assert hinge_ratio > 5.0, f"Expected hinge_ratio > 5, got {hinge_ratio:.2f}"
        # Gated-SP stays bounded (ratio ≈ 1 when gate saturates in both cases)
        assert gated_ratio < hinge_ratio, (
            f"Expected gated_ratio ({gated_ratio:.2f}) < hinge_ratio ({hinge_ratio:.2f})"
        )

    def test_gate_alpha_none_defaults_to_alpha(self, simple_scenario):
        """gate_alpha=None should give identical results to gate_alpha=alpha."""
        y_pred, y_true, mask_S, _ = simple_scenario
        alpha = 0.05
        cfg_none = ASBRConfig(lambd=1.0, alpha=alpha, kappa=5.0,
                              use_hinge=True, use_sp_magnitude=True,
                              use_hybrid=False, use_size_weight=False,
                              gate_alpha=None)
        cfg_explicit = ASBRConfig(lambd=1.0, alpha=alpha, kappa=5.0,
                                  use_hinge=True, use_sp_magnitude=True,
                                  use_hybrid=False, use_size_weight=False,
                                  gate_alpha=alpha)
        loss_none = ASBRLRTLoss(cfg_none)
        loss_explicit = ASBRLRTLoss(cfg_explicit)
        assert loss_none.z_gate == pytest.approx(loss_explicit.z_gate, rel=1e-6)

        _, d_none = loss_none(y_pred, y_true, {"g": mask_S})
        _, d_exp = loss_explicit(y_pred, y_true, {"g": mask_S})
        assert d_none["g"].hinge_value == pytest.approx(d_exp["g"].hinge_value, rel=1e-5)
        assert d_none["g"].gate_value == pytest.approx(d_exp["g"].gate_value, rel=1e-5)

    def test_gate_alpha_shifts_threshold(self):
        """Higher gate_alpha → lower z_gate → gate activates at lower |Z|.

        A subgroup at |Z|≈1.5 (borderline — below z_0.05=1.96 but above z_0.20=1.28)
        should have a strictly higher gate value with gate_alpha=0.20 than with
        gate_alpha=0.05.
        """
        # Construct a scenario with SP ≈ 0.10, n_S=100 → |Z| ≈ 1.4–1.6
        n = 200
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:100] = True
        y_pred = torch.zeros(n)
        y_pred[:10] = 1.0       # S: 10/100 = 0.10
        y_pred[100:] = 0.0      # Sbar: 0/100 = 0.00  → SP ≈ 0.10

        cfg_strict = ASBRConfig(lambd=1.0, alpha=0.05, kappa=5.0,
                                use_hinge=True, use_sp_magnitude=True,
                                use_hybrid=False, use_size_weight=False,
                                gate_alpha=0.05)   # z_gate = 1.96
        cfg_loose = ASBRConfig(lambd=1.0, alpha=0.05, kappa=5.0,
                               use_hinge=True, use_sp_magnitude=True,
                               use_hybrid=False, use_size_weight=False,
                               gate_alpha=0.20)   # z_gate = 1.28

        loss_strict = ASBRLRTLoss(cfg_strict)
        loss_loose = ASBRLRTLoss(cfg_loose)

        # Stricter z_gate should be higher
        assert loss_strict.z_gate > loss_loose.z_gate

        dummy_y = torch.zeros(n)
        _, d_strict = loss_strict(y_pred, dummy_y, {"g": mask_S})
        _, d_loose = loss_loose(y_pred, dummy_y, {"g": mask_S})

        gate_strict = d_strict["g"].gate_value
        gate_loose = d_loose["g"].gate_value
        # Looser threshold → higher gate for a borderline group
        assert gate_loose > gate_strict, (
            f"Expected gate_loose ({gate_loose:.4f}) > gate_strict ({gate_strict:.4f})"
        )

    def test_high_kappa_no_attenuation(self):
        """With kappa=20, a clearly significant group (|Z| >> z_alpha) gets gate ≈ 1.0.

        High kappa → near-binary gate → no attenuation for penalized groups.
        """
        # n=2000, SP=0.25 → |Z| >> 1.96
        n = 2000
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:1000] = True
        y_pred = torch.zeros(n)
        y_pred[:250] = 1.0      # S: 250/1000 = 0.25
        y_pred[1000:] = 0.0    # Sbar: 0/1000 = 0.00

        cfg_high_kappa = ASBRConfig(lambd=1.0, alpha=0.05, kappa=20.0,
                                    use_hinge=True, use_sp_magnitude=True,
                                    use_hybrid=False, use_size_weight=False)
        cfg_low_kappa = ASBRConfig(lambd=1.0, alpha=0.05, kappa=0.5,
                                   use_hinge=True, use_sp_magnitude=True,
                                   use_hybrid=False, use_size_weight=False)
        loss_high = ASBRLRTLoss(cfg_high_kappa)
        loss_low = ASBRLRTLoss(cfg_low_kappa)

        dummy_y = torch.zeros(n)
        _, d_high = loss_high(y_pred, dummy_y, {"g": mask_S})
        _, d_low = loss_low(y_pred, dummy_y, {"g": mask_S})

        gate_high = d_high["g"].gate_value
        gate_low = d_low["g"].gate_value

        # High kappa → gate saturates near 1.0 (tolerance 1%)
        assert gate_high == pytest.approx(1.0, abs=0.01), (
            f"Expected gate ≈ 1.0 with kappa=20, got {gate_high:.4f}"
        )
        # Low kappa → more attenuation (gate < 1.0 even for large |Z|)
        assert gate_low < gate_high, (
            f"Expected gate_low ({gate_low:.4f}) < gate_high ({gate_high:.4f})"
        )

    def test_high_gate_alpha_approaches_sp_penalty(self):
        """gate_alpha → 1.0 (z_gate → 0) makes gate ≈ 1 for all groups → SP-Penalty.

        When gate_alpha is very high and kappa is large, every group with nonzero
        SP gets gate ≈ 1.0, so gated-SP ≈ SP² per group (same as SP-Penalty).
        """
        n = 200
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:100] = True
        y_pred = torch.zeros(n)
        y_pred[:15] = 1.0      # S: 15/100 = 0.15
        y_pred[100:] = 0.0    # Sbar: 0.00

        # Very high gate_alpha with high kappa → gate ≈ 1.0 for all nonzero SP
        cfg_permissive = ASBRConfig(lambd=1.0, alpha=0.05, kappa=20.0,
                                    use_hinge=True, use_sp_magnitude=True,
                                    use_hybrid=False, use_size_weight=False,
                                    gate_alpha=0.99)  # z_gate ≈ 0
        loss_fn = ASBRLRTLoss(cfg_permissive)
        dummy_y = torch.zeros(n)
        _, diag = loss_fn(y_pred, dummy_y, {"g": mask_S})
        d = diag["g"]

        # Gate should saturate near 1.0
        assert d.gate_value == pytest.approx(1.0, abs=0.01), (
            f"Expected gate ≈ 1.0 with gate_alpha=0.99, got {d.gate_value:.4f}"
        )
        # Hinge should be ≈ SP²
        expected_sp2 = d.sp ** 2
        assert d.hinge_value == pytest.approx(expected_sp2, rel=0.05), (
            f"Expected hinge ≈ SP²={expected_sp2:.4f}, got {d.hinge_value:.4f}"
        )


# ---------------------------------------------------------------------------
# compute_eo_soft_rates
# ---------------------------------------------------------------------------

class TestEOSoftRates:
    """Tests for compute_eo_soft_rates (TPR rates conditioned on y=1)."""

    def test_returns_three_tensors(self, simple_scenario):
        """Should return (tpr_S, tpr_Sbar, tpr_pooled) when positives exist."""
        y_pred, y_true, mask_S, _ = simple_scenario
        result = compute_eo_soft_rates(y_pred, mask_S, y_true)
        assert result is not None
        tpr_S, tpr_Sbar, tpr_pooled = result
        assert tpr_S.shape == ()
        assert tpr_Sbar.shape == ()
        assert tpr_pooled.shape == ()

    def test_correct_values(self, simple_scenario):
        """simple_scenario: S has 30 positive labels, all predicted 1 (TPR=1.0).
        Sbar has 30 positive labels, all predicted 1 (TPR=1.0).
        """
        y_pred, y_true, mask_S, _ = simple_scenario
        result = compute_eo_soft_rates(y_pred, mask_S, y_true)
        assert result is not None
        tpr_S, tpr_Sbar, _ = result
        assert tpr_S.item() == pytest.approx(1.0, abs=1e-5)
        assert tpr_Sbar.item() == pytest.approx(1.0, abs=1e-5)

    def test_returns_none_when_no_positives_in_S(self):
        """Should return None when S contains no ground-truth positives."""
        n = 40
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:10] = True
        y_pred = torch.rand(n)
        y_true = torch.zeros(n)   # all negatives in S
        y_true[10:] = 1.0         # positives only in Sbar
        result = compute_eo_soft_rates(y_pred, mask_S, y_true)
        assert result is None

    def test_returns_none_when_no_positives_in_Sbar(self):
        """Should return None when Sbar contains no ground-truth positives."""
        n = 40
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:10] = True
        y_pred = torch.rand(n)
        y_true = torch.zeros(n)
        y_true[:10] = 1.0   # positives only in S
        result = compute_eo_soft_rates(y_pred, mask_S, y_true)
        assert result is None

    def test_clamped_in_bounds(self):
        """TPR rates should stay in (epsilon, 1-epsilon) even at extremes."""
        n = 20
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:5] = True
        y_pred = torch.zeros(n)  # all zeros → soft TPR = 0
        y_true = torch.ones(n)   # all positives
        result = compute_eo_soft_rates(y_pred, mask_S, y_true, epsilon=1e-8)
        assert result is not None
        tpr_S, tpr_Sbar, _ = result
        # float32 may represent the clamped epsilon slightly below 1e-6
        assert tpr_S.item() >= 1e-7

    def test_no_gradient_through_y_true(self):
        """Gradient should flow through y_pred but not y_true."""
        n = 40
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:20] = True
        y_pred = torch.rand(n, requires_grad=True)
        y_true = torch.randint(0, 2, (n,)).float()
        result = compute_eo_soft_rates(y_pred, mask_S, y_true)
        assert result is not None
        tpr_S, _, _ = result
        tpr_S.backward()
        assert y_pred.grad is not None


# ---------------------------------------------------------------------------
# compute_eo_lrt_statistic
# ---------------------------------------------------------------------------

class TestEOLRTStatistic:
    """Tests for compute_eo_lrt_statistic (LRT on TPR disparity)."""

    @pytest.fixture
    def eo_disparity_scenario(self):
        """Scenario with EO disparity: different TPR between S and Sbar.

        S:    10 positives, 8 predicted positive → TPR_S = 0.80
        Sbar: 20 positives, 8 predicted positive → TPR_Sbar = 0.40
        EO = 0.40, LRT should be positive.
        """
        n = 60
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:20] = True  # 20 samples in S

        y_true = torch.zeros(n)
        y_true[:10] = 1.0   # 10 positives in S
        y_true[20:40] = 1.0  # 20 positives in Sbar

        y_pred = torch.zeros(n)
        y_pred[:8] = 1.0    # 8/10 TP in S → TPR_S = 0.80
        y_pred[20:28] = 1.0  # 8/20 TP in Sbar → TPR_Sbar = 0.40

        return y_pred, y_true, mask_S

    @pytest.fixture
    def eo_balanced_scenario(self):
        """Scenario with no EO disparity: identical TPR in S and Sbar."""
        n = 60
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:20] = True

        y_true = torch.zeros(n)
        y_true[:10] = 1.0
        y_true[20:40] = 1.0

        y_pred = torch.zeros(n)
        y_pred[:5] = 1.0    # 5/10 TP in S → TPR_S = 0.50
        y_pred[20:30] = 1.0  # 10/20 TP in Sbar → TPR_Sbar = 0.50

        return y_pred, y_true, mask_S

    def test_non_negative(self, eo_disparity_scenario):
        y_pred, y_true, mask_S = eo_disparity_scenario
        stat = compute_eo_lrt_statistic(y_pred, mask_S, y_true)
        assert stat is not None
        assert stat.item() >= 0.0

    def test_zero_when_equal_tpr(self, eo_balanced_scenario):
        y_pred, y_true, mask_S = eo_balanced_scenario
        stat = compute_eo_lrt_statistic(y_pred, mask_S, y_true)
        assert stat is not None
        assert stat.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_when_disparate(self, eo_disparity_scenario):
        y_pred, y_true, mask_S = eo_disparity_scenario
        stat = compute_eo_lrt_statistic(y_pred, mask_S, y_true)
        assert stat is not None
        assert stat.item() > 0.5

    def test_returns_none_no_positives_in_S(self):
        n = 40
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:10] = True
        y_pred = torch.rand(n)
        y_true = torch.zeros(n)
        y_true[10:] = 1.0  # positives only in Sbar
        result = compute_eo_lrt_statistic(y_pred, mask_S, y_true)
        assert result is None

    def test_scalar_output(self, eo_disparity_scenario):
        y_pred, y_true, mask_S = eo_disparity_scenario
        stat = compute_eo_lrt_statistic(y_pred, mask_S, y_true)
        assert stat is not None
        assert stat.dim() == 0

    def test_differentiable(self, eo_disparity_scenario):
        """EO LRT should propagate gradients through y_pred."""
        y_pred, y_true, mask_S = eo_disparity_scenario
        y_pred = y_pred.clone().detach().requires_grad_(True)
        stat = compute_eo_lrt_statistic(y_pred, mask_S, y_true)
        assert stat is not None
        stat.backward()
        assert y_pred.grad is not None
        assert not torch.all(y_pred.grad == 0)

    def test_larger_disparity_larger_stat(self):
        """Larger EO disparity should produce a larger statistic."""
        n = 80
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:40] = True
        y_true = torch.zeros(n)
        y_true[:20] = 1.0   # 20 positives in S
        y_true[40:60] = 1.0  # 20 positives in Sbar

        # Small disparity: TPR_S=0.6, TPR_Sbar=0.5
        y_small = torch.zeros(n)
        y_small[:12] = 1.0
        y_small[40:50] = 1.0

        # Large disparity: TPR_S=0.9, TPR_Sbar=0.1
        y_large = torch.zeros(n)
        y_large[:18] = 1.0
        y_large[40:42] = 1.0

        stat_small = compute_eo_lrt_statistic(y_small, mask_S, y_true)
        stat_large = compute_eo_lrt_statistic(y_large, mask_S, y_true)
        assert stat_small is not None and stat_large is not None
        assert stat_large.item() > stat_small.item()


# ---------------------------------------------------------------------------
# ASBRLRTLoss with fairness_metric="eo"
# ---------------------------------------------------------------------------

class TestASBRLRTLossEO:
    """Tests for ASBRLRTLoss with fairness_metric='eo'."""

    @pytest.fixture
    def eo_scenario(self):
        """60 samples; clear EO disparity, no SP disparity by design."""
        n = 60
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:20] = True

        y_true = torch.zeros(n)
        y_true[:10] = 1.0
        y_true[20:40] = 1.0

        y_pred = torch.zeros(n)
        y_pred[:8] = 1.0    # TPR_S = 0.80
        y_pred[20:28] = 1.0  # TPR_Sbar = 0.40

        return y_pred, y_true, mask_S

    def test_eo_loss_is_finite(self, eo_scenario):
        y_pred, y_true, mask_S = eo_scenario
        cfg = ASBRConfig(fairness_metric="eo", lambd=1.0)
        loss_fn = ASBRLRTLoss(cfg)
        total_loss, _ = loss_fn(y_pred, y_true, {"group_S": mask_S})
        assert torch.isfinite(total_loss)

    def test_eo_diagnostics_have_eo_field(self, eo_scenario):
        y_pred, y_true, mask_S = eo_scenario
        cfg = ASBRConfig(fairness_metric="eo", lambd=1.0)
        loss_fn = ASBRLRTLoss(cfg)
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        assert "group_S" in diag
        assert diag["group_S"].eo == pytest.approx(0.40, abs=0.01)

    def test_eo_sp_always_recorded(self, eo_scenario):
        """SP diagnostic should be recorded even in EO mode."""
        y_pred, y_true, mask_S = eo_scenario
        cfg = ASBRConfig(fairness_metric="eo", lambd=1.0)
        loss_fn = ASBRLRTLoss(cfg)
        _, diag = loss_fn(y_pred, y_true, {"group_S": mask_S})
        assert isinstance(diag["group_S"].sp, float)

    def test_eo_requires_y_true(self, eo_scenario):
        """fairness_metric='eo' must raise RuntimeError when y_true is None."""
        y_pred, _, mask_S = eo_scenario
        cfg = ASBRConfig(fairness_metric="eo")
        loss_fn = ASBRLRTLoss(cfg)
        with pytest.raises(RuntimeError, match="y_true"):
            loss_fn.compute_fairness_loss(y_pred, {"group_S": mask_S}, y_true=None)

    def test_invalid_metric_raises(self, eo_scenario):
        y_pred, y_true, mask_S = eo_scenario
        cfg = ASBRConfig(fairness_metric="invalid")
        loss_fn = ASBRLRTLoss(cfg)
        with pytest.raises(ValueError, match="fairness_metric"):
            loss_fn.compute_fairness_loss(y_pred, {"group_S": mask_S}, y_true=y_true)

    def test_eo_skips_subgroup_with_no_positives(self):
        """Subgroups with no positives on either side should be silently skipped."""
        n = 40
        mask_S = torch.zeros(n, dtype=torch.bool)
        mask_S[:10] = True
        y_pred = torch.rand(n)
        y_true = torch.zeros(n)
        y_true[10:] = 1.0  # positives only in Sbar → EO undefined for S

        cfg = ASBRConfig(fairness_metric="eo")
        loss_fn = ASBRLRTLoss(cfg)
        _, diag = loss_fn.compute_fairness_loss(
            y_pred, {"group_S": mask_S}, y_true=y_true
        )
        assert "group_S" not in diag

    def test_eo_backward_propagates(self):
        """EO loss.backward() should produce non-zero gradients on model params."""
        from asbr.models.base_mlp import FairnessMLP

        torch.manual_seed(7)
        model = FairnessMLP(input_dim=5, hidden_dims=(8,), dropout=0.0)
        n = 60
        X = torch.randn(n, 5)
        y_true = torch.zeros(n)
        y_true[:20] = 1.0
        y_true[30:50] = 1.0
        mask = torch.zeros(n, dtype=torch.bool)
        mask[:30] = True

        cfg = ASBRConfig(fairness_metric="eo", lambd=1.0)
        loss_fn = ASBRLRTLoss(cfg)
        y_pred = model(X)
        total_loss, _ = loss_fn(y_pred, y_true, {"group": mask})
        total_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_sp_mode_ignores_y_true_for_fairness(self, simple_scenario):
        """Default SP mode should produce the same result regardless of y_true passed."""
        y_pred, y_true, mask_S, _ = simple_scenario
        cfg = ASBRConfig(fairness_metric="sp", lambd=1.0)
        loss_fn = ASBRLRTLoss(cfg)

        _, diag_with = loss_fn.compute_fairness_loss(
            y_pred, {"g": mask_S}, y_true=y_true
        )
        _, diag_without = loss_fn.compute_fairness_loss(
            y_pred, {"g": mask_S}, y_true=None
        )
        assert diag_with["g"].lambda_stat == pytest.approx(
            diag_without["g"].lambda_stat, rel=1e-5
        )
