"""Tests for the SAFT (Size-Adaptive Fairness Testing) assessment module."""

import numpy as np
import pytest

from asbr.fairness.saft import (
    SAFTResult,
    SAFTSummary,
    EOSAFTResult,
    EOSAFTSummary,
    apply_multiplicity_correction,
    apply_multiplicity_correction_eo,
    compute_hard_counts,
    compute_eo_counts,
    compute_mspe,
    compute_msee,
    _reject_h0,
    _wald_test,
    _bayesian_mc_test,
    saft_test_subgroup,
    saft_eo_test_subgroup,
    saft_audit,
    saft_eo_audit,
    saft_summary,
    saft_eo_summary,
)


# ---------------------------------------------------------------------------
# Test hard counts
# ---------------------------------------------------------------------------

class TestComputeHardCounts:
    def test_basic_counts(self):
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        y_true = np.array([1, 0, 1, 0, 1, 0])  # not used for SP
        mask_S = np.array([True, True, True, False, False, False])

        counts = compute_hard_counts(y_pred, y_true, mask_S)
        # S: y_pred = [1, 1, 0] -> c1S=2, c0S=1
        # Sbar: y_pred = [0, 1, 0] -> c1Sbar=1, c0Sbar=2
        assert counts[0] == 2  # c1S
        assert counts[1] == 1  # c0S
        assert counts[2] == 1  # c1Sbar
        assert counts[3] == 2  # c0Sbar

    def test_all_positive(self):
        y_pred = np.ones(10, dtype=int)
        y_true = np.zeros(10, dtype=int)
        mask_S = np.array([True] * 4 + [False] * 6)
        counts = compute_hard_counts(y_pred, y_true, mask_S)
        assert counts[0] == 4   # c1S
        assert counts[1] == 0   # c0S
        assert counts[2] == 6   # c1Sbar
        assert counts[3] == 0   # c0Sbar

    def test_sum_equals_n(self):
        rng = np.random.default_rng(0)
        n = 200
        y_pred = rng.integers(0, 2, size=n)
        y_true = rng.integers(0, 2, size=n)
        mask_S = rng.random(n) > 0.5
        counts = compute_hard_counts(y_pred, y_true, mask_S)
        assert counts.sum() == n


# ---------------------------------------------------------------------------
# Test Wald test path
# ---------------------------------------------------------------------------

class TestWaldTest:
    def test_no_disparity(self):
        """When acceptance rates are equal, SP ≈ 0 and p-value should be large."""
        # Equal rates: 50% in both groups
        # S: 50 positive, 50 negative; Sbar: 50 positive, 50 negative
        n = 200
        p_hat = np.array([50, 50, 50, 50]) / n
        sp, ci_lo, ci_hi, p_val = _wald_test(p_hat, n, alpha=0.05)
        assert abs(sp) < 1e-10
        assert ci_lo < 0 < ci_hi
        assert p_val > 0.9

    def test_large_disparity(self):
        """When there is a large acceptance-rate gap, p-value should be small."""
        n = 1000
        # S: 90% accept; Sbar: 10% accept
        p_hat = np.array([90, 10, 50, 450]) / n
        sp, ci_lo, ci_hi, p_val = _wald_test(p_hat, n, alpha=0.05)
        assert abs(sp) > 0.5
        assert p_val < 0.001

    def test_ci_width_shrinks_with_n(self):
        """CI should narrow as sample size grows."""
        p_hat = np.array([0.2, 0.3, 0.15, 0.35])
        _, lo1, hi1, _ = _wald_test(p_hat, n=100, alpha=0.05)
        _, lo2, hi2, _ = _wald_test(p_hat, n=10000, alpha=0.05)
        assert (hi2 - lo2) < (hi1 - lo1)


# ---------------------------------------------------------------------------
# Test Bayesian MC test path
# ---------------------------------------------------------------------------

class TestBayesianMCTest:
    def test_no_disparity_does_not_reject(self):
        """With equal rates in small sample, CI should contain 0."""
        counts = np.array([5, 5, 5, 5])
        rng = np.random.default_rng(42)
        sp, ci_lo, ci_hi, p_val = _bayesian_mc_test(
            counts, alpha=0.05, K=20000, rng=rng,
        )
        assert ci_lo < 0 < ci_hi
        assert p_val > 0.05

    def test_large_disparity_rejects(self):
        """With strongly asymmetric counts, should reject."""
        counts = np.array([9, 1, 1, 9])  # S: 90% accept, Sbar: 10%
        rng = np.random.default_rng(42)
        sp, ci_lo, ci_hi, p_val = _bayesian_mc_test(
            counts, alpha=0.05, K=20000, rng=rng,
        )
        assert abs(sp) > 0.3
        assert p_val < 0.05

    def test_reproducibility(self):
        """Same seed should give same result."""
        counts = np.array([8, 2, 3, 7])
        r1 = _bayesian_mc_test(counts, rng=np.random.default_rng(123))
        r2 = _bayesian_mc_test(counts, rng=np.random.default_rng(123))
        assert r1 == r2


# ---------------------------------------------------------------------------
# Test switching logic
# ---------------------------------------------------------------------------

class TestSAFTSwitch:
    def test_large_subgroup_uses_wald(self):
        """Subgroup with min cell ≥ n_tilde and nonzero SP should use Wald."""
        # S: 70% accept (140/200), Sbar: 50% accept (100/200)
        # All cells >= 30: c1S=140, c0S=60, c1Sbar=100, c0Sbar=100
        y_pred_S = np.array([1] * 140 + [0] * 60)
        y_pred_Sbar = np.array([1] * 100 + [0] * 100)
        y_pred = np.concatenate([y_pred_S, y_pred_Sbar])
        y_true = np.zeros(400, dtype=int)
        mask_S = np.array([True] * 200 + [False] * 200)

        result = saft_test_subgroup(
            y_pred, y_true, mask_S, "test_group",
            alpha=0.05, n_tilde=30,
        )
        assert result.test_type == "wald"

    def test_small_subgroup_uses_bayesian(self):
        """Subgroup with min cell < n_tilde should use Bayesian."""
        # Small S group: 3 positive, 2 negative; large Sbar
        y_pred = np.array([1, 1, 1, 0, 0] + [1] * 50 + [0] * 50)
        y_true = np.zeros(len(y_pred), dtype=int)
        mask_S = np.array([True] * 5 + [False] * 100)

        result = saft_test_subgroup(
            y_pred, y_true, mask_S, "small_group",
            alpha=0.05, n_tilde=30, rng=np.random.default_rng(0),
        )
        assert result.test_type == "bayesian"

    def test_result_fields(self):
        """Check all SAFTResult fields are populated."""
        y_pred = np.array([1] * 50 + [0] * 50 + [1] * 50 + [0] * 50)
        y_true = np.zeros(200, dtype=int)
        mask_S = np.array([True] * 100 + [False] * 100)

        result = saft_test_subgroup(
            y_pred, y_true, mask_S, "group_A",
            alpha=0.05, n_tilde=30,
        )
        assert result.subgroup == "group_A"
        assert result.n_S == 100
        assert isinstance(result.p_value, float)
        assert isinstance(result.reject_h0, bool)
        assert result.test_type in ("wald", "bayesian", "exact_zero")

    def test_constant_predictions_no_rejection(self):
        """Constant predictions (all 0s) should never reject — SP is exactly 0."""
        n = 200
        y_pred = np.zeros(n, dtype=int)
        y_true = np.zeros(n, dtype=int)
        mask_S = np.array([True] * 20 + [False] * 180)

        result = saft_test_subgroup(
            y_pred, y_true, mask_S, "const_group",
            alpha=0.05, n_tilde=30, rng=np.random.default_rng(0),
        )
        assert result.test_type == "exact_zero"
        assert result.reject_h0 is False
        assert result.sp_estimate == 0.0
        assert result.p_value == 1.0

    def test_constant_predictions_all_ones_no_rejection(self):
        """All-1 predictions also have zero SP."""
        n = 300
        y_pred = np.ones(n, dtype=int)
        y_true = np.zeros(n, dtype=int)
        mask_S = np.array([True] * 10 + [False] * 290)

        result = saft_test_subgroup(
            y_pred, y_true, mask_S, "all_ones",
            alpha=0.05, n_tilde=30, rng=np.random.default_rng(0),
        )
        assert result.test_type == "exact_zero"
        assert result.reject_h0 is False

    def test_equal_rates_different_sizes_no_rejection(self):
        """If acceptance rates are identical across S and Sbar, SP=0 exactly."""
        # S: 50% accept (5/10), Sbar: 50% accept (45/90)
        y_pred_S = np.array([1]*5 + [0]*5)
        y_pred_Sbar = np.array([1]*45 + [0]*45)
        y_pred = np.concatenate([y_pred_S, y_pred_Sbar])
        y_true = np.zeros(100, dtype=int)
        mask_S = np.array([True]*10 + [False]*90)

        result = saft_test_subgroup(
            y_pred, y_true, mask_S, "equal_rates",
            alpha=0.05, n_tilde=30, rng=np.random.default_rng(0),
        )
        assert result.test_type == "exact_zero"
        assert result.reject_h0 is False

    def test_nonzero_sp_still_tested(self):
        """When observed SP != 0, the test should still proceed normally."""
        # S: 80% accept, Sbar: 50% accept
        y_pred_S = np.array([1]*8 + [0]*2)
        y_pred_Sbar = np.array([1]*45 + [0]*45)
        y_pred = np.concatenate([y_pred_S, y_pred_Sbar])
        y_true = np.zeros(100, dtype=int)
        mask_S = np.array([True]*10 + [False]*90)

        result = saft_test_subgroup(
            y_pred, y_true, mask_S, "nonzero_sp",
            alpha=0.05, n_tilde=30, rng=np.random.default_rng(0),
        )
        assert result.test_type in ("wald", "bayesian")
        assert result.sp_estimate != 0.0

    def test_constant_predictions_audit_zero_violations(self):
        """Full audit with constant predictions should find zero violations."""
        n = 500
        y_pred = np.zeros(n, dtype=int)
        y_true = np.zeros(n, dtype=int)
        masks = {
            "small": np.arange(n) < 15,
            "medium": np.arange(n) < 80,
            "large": np.arange(n) < 250,
        }
        results = saft_audit(y_pred, y_true, masks, seed=42)
        summ = saft_summary(results)
        assert summ.n_violations == 0
        assert summ.violation_rate == 0.0
        assert all(r.test_type == "exact_zero" for r in results)


# ---------------------------------------------------------------------------
# Test audit
# ---------------------------------------------------------------------------

class TestSAFTAudit:
    def test_audit_runs_on_multiple_subgroups(self):
        n = 200
        y_pred = np.random.default_rng(0).integers(0, 2, size=n)
        y_true = np.random.default_rng(1).integers(0, 2, size=n)
        masks = {
            "A": np.arange(n) < 100,
            "B": np.arange(n) >= 50,
        }
        results = saft_audit(y_pred, y_true, masks, seed=42)
        assert len(results) == 2
        assert all(isinstance(r, SAFTResult) for r in results)
        names = [r.subgroup for r in results]
        assert "A" in names and "B" in names


# ---------------------------------------------------------------------------
# Test summary aggregation
# ---------------------------------------------------------------------------

class TestSAFTSummary:
    def _make_result(self, name, n_S, reject, sp=0.1):
        return SAFTResult(
            subgroup=name, n_S=n_S, min_cell_count=n_S // 4,
            sp_estimate=sp, ci_lower=sp - 0.05, ci_upper=sp + 0.05,
            p_value=0.001 if reject else 0.5,
            reject_h0=reject, test_type="wald",
        )

    def test_violation_rate(self):
        results = [
            self._make_result("A", 100, True),
            self._make_result("B", 200, False),
            self._make_result("C", 50, True),
            self._make_result("D", 8, False),
        ]
        s = saft_summary(results)
        assert s.n_subgroups == 4
        assert s.n_violations == 2
        assert abs(s.violation_rate - 0.5) < 1e-10

    def test_size_buckets(self):
        results = [
            self._make_result("tiny", 5, True),
            self._make_result("small", 20, False),
            self._make_result("med", 50, True),
            self._make_result("large", 200, False),
        ]
        s = saft_summary(results)
        assert s.violations_by_size_bucket["[2, 10)"]["n_violations"] == 1
        assert s.violations_by_size_bucket["[10, 30)"]["n_violations"] == 0
        assert s.violations_by_size_bucket["[30, 100)"]["n_violations"] == 1
        assert s.violations_by_size_bucket["[100, inf)"]["n_violations"] == 0

    def test_mean_sp_among_violations(self):
        results = [
            self._make_result("A", 100, True, sp=0.2),
            self._make_result("B", 100, True, sp=0.4),
            self._make_result("C", 100, False, sp=0.01),
        ]
        s = saft_summary(results)
        assert abs(s.mean_sp_among_violations - 0.3) < 1e-10

    def test_no_violations(self):
        results = [
            self._make_result("A", 100, False),
            self._make_result("B", 100, False),
        ]
        s = saft_summary(results)
        assert s.n_violations == 0
        assert s.violation_rate == 0.0
        assert s.mean_sp_among_violations == 0.0


# ---------------------------------------------------------------------------
# Test concordance: for large samples, Bayesian and Wald should agree
# ---------------------------------------------------------------------------

class TestConcordance:
    def test_large_sample_agreement(self):
        """With large counts, Bayesian MC and Wald should agree on rejection."""
        # Large disparity, large sample
        counts_reject = np.array([180, 20, 20, 180])
        rng = np.random.default_rng(42)

        p_hat = counts_reject / counts_reject.sum()
        _, _, _, p_wald = _wald_test(p_hat, int(counts_reject.sum()))
        _, _, _, p_bayes = _bayesian_mc_test(counts_reject, K=20000, rng=rng)

        # Both should reject
        assert p_wald < 0.05
        assert p_bayes < 0.05

    def test_large_sample_no_disparity_agreement(self):
        """Both methods should fail to reject when SP ≈ 0."""
        counts_null = np.array([100, 100, 100, 100])
        rng = np.random.default_rng(42)

        p_hat = counts_null / counts_null.sum()
        _, _, _, p_wald = _wald_test(p_hat, int(counts_null.sum()))
        _, _, _, p_bayes = _bayesian_mc_test(counts_null, K=20000, rng=rng)

        assert p_wald > 0.05
        assert p_bayes > 0.05


# ---------------------------------------------------------------------------
# Test _reject_h0 helper
# ---------------------------------------------------------------------------

class TestRejectH0:
    def test_p_value_method_rejects(self):
        assert _reject_h0(p_value=0.01, ci_lower=0.05, ci_upper=0.3, alpha=0.05, method="p_value") is True

    def test_p_value_method_accepts(self):
        assert _reject_h0(p_value=0.10, ci_lower=-0.1, ci_upper=0.3, alpha=0.05, method="p_value") is False

    def test_ci_method_rejects_positive(self):
        """CI entirely above zero → reject."""
        assert _reject_h0(p_value=0.99, ci_lower=0.01, ci_upper=0.3, alpha=0.05, method="ci") is True

    def test_ci_method_rejects_negative(self):
        """CI entirely below zero → reject."""
        assert _reject_h0(p_value=0.99, ci_lower=-0.3, ci_upper=-0.01, alpha=0.05, method="ci") is True

    def test_ci_method_accepts(self):
        """CI straddles zero → do not reject."""
        assert _reject_h0(p_value=0.001, ci_lower=-0.1, ci_upper=0.1, alpha=0.05, method="ci") is False

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown rejection method"):
            _reject_h0(0.01, -0.1, 0.1, 0.05, method="invalid")


# ---------------------------------------------------------------------------
# Test equivalence of p-value and CI rejection methods
# ---------------------------------------------------------------------------

class TestRejectionMethodEquivalence:
    """The p-value and CI-containment rejection criteria are theoretically
    equivalent for continuous distributions.  For the Wald path this is exact;
    for the Bayesian MC path it holds up to MC sampling noise.
    """

    def test_wald_equivalence_clear_reject(self):
        """Large disparity (Wald) — both methods should reject."""
        n = 1000
        p_hat = np.array([90, 10, 50, 450]) / n
        sp, ci_lo, ci_hi, p_val = _wald_test(p_hat, n, alpha=0.05)

        reject_pv = _reject_h0(p_val, ci_lo, ci_hi, 0.05, "p_value")
        reject_ci = _reject_h0(p_val, ci_lo, ci_hi, 0.05, "ci")
        assert reject_pv == reject_ci == True  # noqa: E712

    def test_wald_equivalence_clear_accept(self):
        """No disparity (Wald) — both methods should accept."""
        n = 200
        p_hat = np.array([50, 50, 50, 50]) / n
        sp, ci_lo, ci_hi, p_val = _wald_test(p_hat, n, alpha=0.05)

        reject_pv = _reject_h0(p_val, ci_lo, ci_hi, 0.05, "p_value")
        reject_ci = _reject_h0(p_val, ci_lo, ci_hi, 0.05, "ci")
        assert reject_pv == reject_ci == False  # noqa: E712

    def test_wald_equivalence_sweep(self):
        """Sweep over many Wald scenarios — both methods should always agree.

        This is exact because the Wald CI is SP ± z_{α/2}·SE, and the
        p-value test checks |SP|/SE > z_{α/2}, which is algebraically
        identical to 0 ∉ [SP - z_{α/2}·SE, SP + z_{α/2}·SE].
        """
        rng = np.random.default_rng(99)
        alpha = 0.05
        disagreements = 0

        for _ in range(200):
            # Generate random 4-cell counts with min cell ≥ 30
            counts = rng.integers(30, 300, size=4)
            n = int(counts.sum())
            p_hat = counts / n

            sp, ci_lo, ci_hi, p_val = _wald_test(p_hat, n, alpha=alpha)
            reject_pv = _reject_h0(p_val, ci_lo, ci_hi, alpha, "p_value")
            reject_ci = _reject_h0(p_val, ci_lo, ci_hi, alpha, "ci")

            if reject_pv != reject_ci:
                disagreements += 1

        assert disagreements == 0, (
            f"Wald: p-value and CI methods disagreed on {disagreements}/200 cases"
        )

    def test_bayesian_equivalence_clear_reject(self):
        """Large disparity (Bayesian) — both methods should reject."""
        counts = np.array([9, 1, 1, 9])
        rng = np.random.default_rng(42)
        sp, ci_lo, ci_hi, p_val = _bayesian_mc_test(counts, alpha=0.05, K=50000, rng=rng)

        reject_pv = _reject_h0(p_val, ci_lo, ci_hi, 0.05, "p_value")
        reject_ci = _reject_h0(p_val, ci_lo, ci_hi, 0.05, "ci")
        assert reject_pv == reject_ci == True  # noqa: E712

    def test_bayesian_equivalence_clear_accept(self):
        """No disparity (Bayesian) — both methods should accept."""
        counts = np.array([5, 5, 5, 5])
        rng = np.random.default_rng(42)
        sp, ci_lo, ci_hi, p_val = _bayesian_mc_test(counts, alpha=0.05, K=50000, rng=rng)

        reject_pv = _reject_h0(p_val, ci_lo, ci_hi, 0.05, "p_value")
        reject_ci = _reject_h0(p_val, ci_lo, ci_hi, 0.05, "ci")
        assert reject_pv == reject_ci == False  # noqa: E712

    def test_bayesian_equivalence_sweep(self):
        """Sweep over many Bayesian scenarios — methods should nearly always agree.

        Unlike the Wald case, the Bayesian posterior tail probability and the
        percentile CI can disagree at the boundary due to finite MC samples.
        With K=50000 the disagreement rate should be negligible.
        """
        rng_gen = np.random.default_rng(77)
        alpha = 0.05
        disagreements = 0
        n_trials = 100

        for _ in range(n_trials):
            counts = rng_gen.integers(1, 20, size=4)
            mc_rng = np.random.default_rng(rng_gen.integers(0, 2**31))
            sp, ci_lo, ci_hi, p_val = _bayesian_mc_test(
                counts, alpha=alpha, K=50000, rng=mc_rng,
            )
            reject_pv = _reject_h0(p_val, ci_lo, ci_hi, alpha, "p_value")
            reject_ci = _reject_h0(p_val, ci_lo, ci_hi, alpha, "ci")

            if reject_pv != reject_ci:
                disagreements += 1

        # Allow a small number of boundary disagreements due to MC noise
        assert disagreements <= 3, (
            f"Bayesian: p-value and CI methods disagreed on "
            f"{disagreements}/{n_trials} cases (expected ≤ 3)"
        )

    def test_saft_audit_equivalence(self):
        """Full audit: p_value and CI methods produce the same rejections."""
        rng = np.random.default_rng(0)
        n = 500
        y_pred = rng.integers(0, 2, size=n)
        y_true = rng.integers(0, 2, size=n)

        # Create several subgroups of varying sizes (both Wald and Bayesian paths)
        masks = {
            "large_A": np.arange(n) < 200,
            "large_B": np.arange(n) >= 300,
            "small_C": np.arange(n) < 15,
            "medium_D": (np.arange(n) >= 100) & (np.arange(n) < 180),
        }

        results_pv = saft_audit(
            y_pred, y_true, masks, seed=42, mc_samples=50000,
            rejection_method="p_value",
        )
        results_ci = saft_audit(
            y_pred, y_true, masks, seed=42, mc_samples=50000,
            rejection_method="ci",
        )

        # SP estimates, CIs, and p-values should be identical (same seed)
        for r_pv, r_ci in zip(results_pv, results_ci):
            assert r_pv.subgroup == r_ci.subgroup
            assert r_pv.sp_estimate == pytest.approx(r_ci.sp_estimate)
            assert r_pv.ci_lower == pytest.approx(r_ci.ci_lower)
            assert r_pv.ci_upper == pytest.approx(r_ci.ci_upper)
            assert r_pv.p_value == pytest.approx(r_ci.p_value)
            assert r_pv.test_type == r_ci.test_type
            # Rejection decisions should agree
            assert r_pv.reject_h0 == r_ci.reject_h0, (
                f"Disagreement on subgroup {r_pv.subgroup}: "
                f"p_value={r_pv.p_value:.4e}, CI=[{r_pv.ci_lower:.4f}, {r_pv.ci_upper:.4f}], "
                f"reject_pv={r_pv.reject_h0}, reject_ci={r_ci.reject_h0}"
            )


# ---------------------------------------------------------------------------
# Test multiplicity correction
# ---------------------------------------------------------------------------

class TestMultiplicityCorrection:
    def _make_result(self, name, p_value, sp=0.1, reject=None):
        if reject is None:
            reject = p_value < 0.05
        return SAFTResult(
            subgroup=name, n_S=100, min_cell_count=40,
            sp_estimate=sp, ci_lower=sp - 0.05, ci_upper=sp + 0.05,
            p_value=p_value, reject_h0=reject, test_type="wald",
        )

    def test_bonferroni_reduces_rejections(self):
        """Bonferroni correction should reject fewer hypotheses."""
        results = [
            self._make_result("A", 0.001),  # survives
            self._make_result("B", 0.010),  # survives (0.01 < 0.05/4 = 0.0125)
            self._make_result("C", 0.030),  # does NOT survive (0.03 > 0.0125)
            self._make_result("D", 0.200),  # does NOT survive
        ]
        corrected = apply_multiplicity_correction(results, method="bonferroni", alpha=0.05)

        assert corrected[0].reject_h0 is True   # p=0.001 < 0.0125
        assert corrected[1].reject_h0 is True   # p=0.010 < 0.0125
        assert corrected[2].reject_h0 is False  # p=0.030 > 0.0125
        assert corrected[3].reject_h0 is False  # p=0.200 > 0.0125

    def test_bh_more_powerful_than_bonferroni(self):
        """BH should reject at least as many hypotheses as Bonferroni."""
        results = [
            self._make_result("A", 0.001),
            self._make_result("B", 0.010),
            self._make_result("C", 0.020),
            self._make_result("D", 0.030),
            self._make_result("E", 0.200),
        ]
        bonf = apply_multiplicity_correction(results, method="bonferroni", alpha=0.05)
        bh = apply_multiplicity_correction(results, method="bh", alpha=0.05)

        n_bonf = sum(r.reject_h0 for r in bonf)
        n_bh = sum(r.reject_h0 for r in bh)
        assert n_bh >= n_bonf

    def test_bh_step_up_procedure(self):
        """BH should reject the correct set based on step-up logic."""
        # 5 tests: sorted p-values are 0.005, 0.01, 0.03, 0.04, 0.5
        # BH thresholds at alpha=0.05: 0.01, 0.02, 0.03, 0.04, 0.05
        # p_(1) = 0.005 <= 0.01 ✓
        # p_(2) = 0.01  <= 0.02 ✓
        # p_(3) = 0.03  <= 0.03 ✓
        # p_(4) = 0.04  <= 0.04 ✓
        # p_(5) = 0.50  <= 0.05 ✗  => max_k = 4
        results = [
            self._make_result("A", 0.03),
            self._make_result("B", 0.005),
            self._make_result("C", 0.50),
            self._make_result("D", 0.01),
            self._make_result("E", 0.04),
        ]
        corrected = apply_multiplicity_correction(results, method="bh", alpha=0.05)

        assert corrected[0].reject_h0 is True   # A: p=0.03
        assert corrected[1].reject_h0 is True   # B: p=0.005
        assert corrected[2].reject_h0 is False  # C: p=0.50
        assert corrected[3].reject_h0 is True   # D: p=0.01
        assert corrected[4].reject_h0 is True   # E: p=0.04

    def test_preserves_other_fields(self):
        """Correction should only change reject_h0, not other fields."""
        results = [self._make_result("X", 0.03, sp=0.15)]
        corrected = apply_multiplicity_correction(results, method="bonferroni", alpha=0.05)

        assert corrected[0].subgroup == "X"
        assert corrected[0].p_value == 0.03
        assert corrected[0].sp_estimate == 0.15
        assert corrected[0].n_S == 100
        assert corrected[0].test_type == "wald"

    def test_empty_results(self):
        assert apply_multiplicity_correction([], method="bh") == []

    def test_all_null_no_rejections(self):
        """Under the null (all large p-values), nothing should be rejected."""
        results = [self._make_result(f"G{i}", 0.5 + 0.01 * i) for i in range(20)]
        for method in ("bonferroni", "bh"):
            corrected = apply_multiplicity_correction(results, method=method)
            assert sum(r.reject_h0 for r in corrected) == 0

    def test_invalid_method_raises(self):
        results = [self._make_result("X", 0.01)]
        with pytest.raises(ValueError, match="Unknown correction"):
            apply_multiplicity_correction(results, method="invalid")


# ---------------------------------------------------------------------------
# Test MSPE (Mean Significant Parity Excess)
# ---------------------------------------------------------------------------

def _make_saft_result(
    sp: float,
    ci_lower: float,
    ci_upper: float,
    reject: bool = False,
    name: str = "G",
) -> SAFTResult:
    """Helper: construct a SAFTResult with given SP and CI bounds."""
    p_value = 0.01 if reject else 0.5
    return SAFTResult(
        subgroup=name,
        n_S=100,
        min_cell_count=30,
        sp_estimate=sp,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        reject_h0=reject,
        test_type="wald",
    )


class TestMSPE:
    """Tests for compute_mspe() and its integration into saft_summary."""

    def test_zero_when_no_results(self):
        assert compute_mspe([]) == pytest.approx(0.0)

    def test_zero_when_ci_contains_zero(self):
        """CI straddles zero → excess = 0."""
        r = _make_saft_result(sp=0.10, ci_lower=-0.05, ci_upper=0.25)
        assert compute_mspe([r]) == pytest.approx(0.0)

    def test_positive_when_ci_excludes_zero(self):
        """CI entirely above zero → excess = ci_lower."""
        r = _make_saft_result(sp=0.15, ci_lower=0.03, ci_upper=0.27)
        assert compute_mspe([r]) == pytest.approx(0.03, abs=1e-9)

    def test_positive_sp_uses_ci_lower(self):
        """For positive SP, excess = max(0, ci_lower)."""
        r = _make_saft_result(sp=0.20, ci_lower=0.08, ci_upper=0.32)
        assert compute_mspe([r]) == pytest.approx(0.08, abs=1e-9)

    def test_negative_sp_uses_ci_upper(self):
        """For negative SP, excess = max(0, -ci_upper)."""
        r = _make_saft_result(sp=-0.20, ci_lower=-0.32, ci_upper=-0.08)
        assert compute_mspe([r]) == pytest.approx(0.08, abs=1e-9)

    def test_negative_sp_ci_contains_zero(self):
        """Negative SP but CI contains zero → excess = 0."""
        r = _make_saft_result(sp=-0.10, ci_lower=-0.25, ci_upper=0.05)
        assert compute_mspe([r]) == pytest.approx(0.0)

    def test_mean_over_groups(self):
        """MSPE is the mean over all groups, including non-violations."""
        r_viol = _make_saft_result(sp=0.20, ci_lower=0.06, ci_upper=0.34, reject=True)
        r_null = _make_saft_result(sp=0.05, ci_lower=-0.02, ci_upper=0.12, reject=False)
        # excess: 0.06 for r_viol, 0.0 for r_null
        mspe = compute_mspe([r_viol, r_null])
        assert mspe == pytest.approx(0.03, abs=1e-9)  # (0.06 + 0.0) / 2

    def test_exact_zero_contributes_nothing(self):
        """A group with sp=0, ci=[0,0] (exact_zero) contributes 0."""
        r = SAFTResult(
            subgroup="Z", n_S=50, min_cell_count=20,
            sp_estimate=0.0, ci_lower=0.0, ci_upper=0.0,
            p_value=1.0, reject_h0=False, test_type="exact_zero",
        )
        assert compute_mspe([r]) == pytest.approx(0.0)

    def test_mspe_less_than_mean_sp_for_borderline_violation(self):
        """For a barely-significant group, MSPE < |SP|.

        MSPE = |SP| - z_alpha*SE, while mean SP = |SP|.
        Borderline significant → small excess relative to SP magnitude.
        """
        # SP=0.10, CI barely excludes zero: [0.001, 0.199]
        r = _make_saft_result(sp=0.10, ci_lower=0.001, ci_upper=0.199, reject=True)
        mspe = compute_mspe([r])
        assert mspe < abs(r.sp_estimate)
        assert mspe == pytest.approx(0.001, abs=1e-9)

    def test_mspe_in_saft_summary(self):
        """saft_summary includes mspe field computed by compute_mspe."""
        r_viol = _make_saft_result(sp=0.20, ci_lower=0.05, ci_upper=0.35, reject=True)
        r_null = _make_saft_result(sp=0.03, ci_lower=-0.04, ci_upper=0.10, reject=False)
        summary = saft_summary([r_viol, r_null])
        assert hasattr(summary, "mspe")
        expected = compute_mspe([r_viol, r_null])
        assert summary.mspe == pytest.approx(expected, abs=1e-9)

    def test_mspe_zero_when_all_non_significant(self):
        """All groups non-significant → MSPE = 0."""
        results = [
            _make_saft_result(sp=0.08, ci_lower=-0.01, ci_upper=0.17),
            _make_saft_result(sp=-0.05, ci_lower=-0.14, ci_upper=0.04),
        ]
        assert compute_mspe(results) == pytest.approx(0.0)

    def test_mspe_end_to_end_from_audit(self):
        """MSPE computed from a real saft_audit run is non-negative and finite."""
        rng = np.random.default_rng(0)
        n = 400
        y_pred = rng.integers(0, 2, size=n)
        y_true = rng.integers(0, 2, size=n)
        mask = rng.random(n) > 0.5
        masks = {"group_A": mask, "group_B": ~mask}

        results = saft_audit(y_pred, y_true, masks, alpha=0.05)
        summary = saft_summary(results)

        assert summary.mspe >= 0.0
        assert np.isfinite(summary.mspe)


# ---------------------------------------------------------------------------
# EO counts
# ---------------------------------------------------------------------------

class TestComputeEOCounts:
    def test_basic_counts(self):
        # S: y_true=1 at indices 0,1 → y_pred=[1,0] → TP_S=1, FN_S=1
        # Sbar: y_true=1 at index 4 → y_pred=[1] → TP_Sbar=1, FN_Sbar=0
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        y_true = np.array([1, 1, 0, 0, 1, 0])
        mask_S = np.array([True, True, True, False, False, False])
        counts = compute_eo_counts(y_pred, y_true, mask_S)
        assert counts is not None
        assert counts[0] == 1  # TP_S
        assert counts[1] == 1  # FN_S
        assert counts[2] == 1  # TP_Sbar
        assert counts[3] == 0  # FN_Sbar

    def test_returns_none_no_positives_in_S(self):
        y_pred = np.array([1, 0, 1, 1, 0])
        y_true = np.array([0, 0, 0, 1, 1])  # no y_true==1 in mask_S
        mask_S = np.array([True, True, True, False, False])
        assert compute_eo_counts(y_pred, y_true, mask_S) is None

    def test_returns_none_no_positives_in_sbar(self):
        y_pred = np.array([1, 0, 1, 1, 0])
        y_true = np.array([1, 1, 1, 0, 0])  # no y_true==1 in ~mask_S
        mask_S = np.array([True, True, True, False, False])
        assert compute_eo_counts(y_pred, y_true, mask_S) is None

    def test_sum_equals_n_positives(self):
        rng = np.random.default_rng(7)
        n = 300
        y_pred = rng.integers(0, 2, size=n)
        y_true = rng.integers(0, 2, size=n)
        mask_S = rng.random(n) > 0.5
        counts = compute_eo_counts(y_pred, y_true, mask_S)
        if counts is not None:
            assert counts.sum() == int((y_true == 1).sum())

    def test_ignores_negative_labels(self):
        """Adding more y_true==0 rows should not change any count cell."""
        y_pred = np.array([1, 0, 1, 1, 0, 0, 0, 0])
        y_true = np.array([1, 1, 0, 1, 1, 0, 0, 0])
        mask_S = np.array([True, True, True, False, False, True, True, True])
        counts_full = compute_eo_counts(y_pred, y_true, mask_S)

        # same data without the trailing negatives
        y_pred2 = np.array([1, 0, 1, 1, 0])
        y_true2 = np.array([1, 1, 0, 1, 1])
        mask_S2 = np.array([True, True, True, False, False])
        counts_trimmed = compute_eo_counts(y_pred2, y_true2, mask_S2)

        assert counts_full is not None
        assert counts_trimmed is not None
        np.testing.assert_array_equal(counts_full, counts_trimmed)


# ---------------------------------------------------------------------------
# EO SAFT subgroup test
# ---------------------------------------------------------------------------

class TestSAFTEOSubgroup:
    def test_large_subgroup_uses_wald(self):
        """Subgroup where all EO cells ≥ n_tilde → Wald path."""
        # S: 200 positives, TPR_S=0.7 → TP_S=140, FN_S=60
        # Sbar: 200 positives, TPR_Sbar=0.5 → TP_Sbar=100, FN_Sbar=100
        rng = np.random.default_rng(0)
        y_true_S    = np.ones(200, dtype=int)
        y_pred_S    = np.array([1]*140 + [0]*60)
        y_true_Sbar = np.ones(200, dtype=int)
        y_pred_Sbar = np.array([1]*100 + [0]*100)
        # pad with negatives so mask positions are clear
        y_pred = np.concatenate([y_pred_S, y_pred_Sbar])
        y_true = np.concatenate([y_true_S, y_true_Sbar])
        mask_S = np.array([True]*200 + [False]*200)

        result = saft_eo_test_subgroup(
            y_pred, y_true, mask_S, "large_group", alpha=0.05, n_tilde=30,
        )
        assert result is not None
        assert result.test_type == "wald"

    def test_small_subgroup_uses_bayesian(self):
        """Subgroup where min EO cell < n_tilde → Bayesian path."""
        # S: 5 positives, Sbar: 100 positives
        y_pred_S    = np.array([1, 0, 1, 0, 1])
        y_true_S    = np.ones(5, dtype=int)
        y_pred_Sbar = np.array([1]*50 + [0]*50)
        y_true_Sbar = np.ones(100, dtype=int)
        y_pred = np.concatenate([y_pred_S, y_pred_Sbar])
        y_true = np.concatenate([y_true_S, y_true_Sbar])
        mask_S = np.array([True]*5 + [False]*100)

        result = saft_eo_test_subgroup(
            y_pred, y_true, mask_S, "small_group",
            alpha=0.05, n_tilde=30, rng=np.random.default_rng(0),
        )
        assert result is not None
        assert result.test_type == "bayesian"

    def test_returns_none_when_eo_undefined(self):
        """No positive labels in S side → None."""
        y_pred = np.array([1, 0, 1, 1, 1])
        y_true = np.array([0, 0, 0, 1, 1])  # no positives in S
        mask_S = np.array([True, True, True, False, False])
        assert saft_eo_test_subgroup(y_pred, y_true, mask_S, "undef") is None

    def test_exact_zero_when_tpr_equal(self):
        """TPR_S == TPR_Sbar → exact_zero, no rejection."""
        # S: 10 positives, TPR=0.5; Sbar: 10 positives, TPR=0.5
        y_pred_S    = np.array([1]*5 + [0]*5)
        y_pred_Sbar = np.array([1]*5 + [0]*5)
        y_true = np.ones(20, dtype=int)
        y_pred = np.concatenate([y_pred_S, y_pred_Sbar])
        mask_S = np.array([True]*10 + [False]*10)

        result = saft_eo_test_subgroup(y_pred, y_true, mask_S, "equal_tpr")
        assert result is not None
        assert result.test_type == "exact_zero"
        assert result.reject_h0 is False
        assert result.eo_estimate == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)

    def test_result_fields_populated(self):
        """All EOSAFTResult fields should be populated with correct types."""
        y_pred = np.array([1]*70 + [0]*30 + [1]*50 + [0]*50)
        y_true = np.ones(200, dtype=int)
        mask_S = np.array([True]*100 + [False]*100)

        result = saft_eo_test_subgroup(y_pred, y_true, mask_S, "group_A")
        assert result is not None
        assert result.subgroup == "group_A"
        assert isinstance(result.n_S_pos, int)
        assert isinstance(result.p_value, float)
        assert isinstance(result.reject_h0, bool)
        assert result.test_type in ("wald", "bayesian", "exact_zero")

    def test_large_disparity_rejects(self):
        """Large TPR gap with enough samples → rejection."""
        # S: 200 positives, TPR=0.9; Sbar: 200 positives, TPR=0.1
        y_pred = np.array([1]*180 + [0]*20 + [1]*20 + [0]*180)
        y_true = np.ones(400, dtype=int)
        mask_S = np.array([True]*200 + [False]*200)

        result = saft_eo_test_subgroup(
            y_pred, y_true, mask_S, "large_disp", alpha=0.05, n_tilde=30,
        )
        assert result is not None
        assert result.reject_h0 is True


# ---------------------------------------------------------------------------
# EO SAFT audit
# ---------------------------------------------------------------------------

class TestSAFTEOAudit:
    def test_skips_undefined_subgroups(self):
        """Subgroups with no positives on one side are dropped."""
        # y_true: indices 0-2 are negative, 3-5 are positive
        # "valid":  S={0,1,2,3}, Sbar={4,5} — both sides have positives (index 3 in S)
        # "undef_S": S={0,1,2}, Sbar={3,4,5} — S has no positives → undefined
        y_pred  = np.array([1, 0, 1, 1, 0, 1])
        y_true  = np.array([0, 0, 0, 1, 1, 1])
        masks = {
            "valid":   np.array([True,  True,  True,  True,  False, False]),
            "undef_S": np.array([True,  True,  True,  False, False, False]),
        }
        results = saft_eo_audit(y_pred, y_true, masks)
        assert len(results) == 1
        assert results[0].subgroup == "valid"

    def test_returns_eo_saft_results(self):
        """All returned objects are EOSAFTResult instances."""
        rng = np.random.default_rng(5)
        n = 200
        y_pred = rng.integers(0, 2, size=n)
        y_true = rng.integers(0, 2, size=n)
        masks = {"A": np.arange(n) < 100, "B": np.arange(n) >= 100}
        results = saft_eo_audit(y_pred, y_true, masks)
        assert all(isinstance(r, EOSAFTResult) for r in results)

    def test_audit_multiple_subgroups(self):
        """Audit runs on 3 subgroups; returns at most 3 results."""
        rng = np.random.default_rng(9)
        n = 300
        y_pred = rng.integers(0, 2, size=n)
        y_true = rng.integers(0, 2, size=n)
        masks = {
            "A": np.arange(n) < 100,
            "B": (np.arange(n) >= 100) & (np.arange(n) < 200),
            "C": np.arange(n) >= 200,
        }
        results = saft_eo_audit(y_pred, y_true, masks)
        assert len(results) <= 3
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# MSEE
# ---------------------------------------------------------------------------

def _make_eo_result(
    eo: float,
    ci_lower: float,
    ci_upper: float,
    reject: bool = False,
    name: str = "G",
) -> EOSAFTResult:
    return EOSAFTResult(
        subgroup=name, n_S_pos=50, min_cell_count=20,
        eo_estimate=eo, ci_lower=ci_lower, ci_upper=ci_upper,
        p_value=0.01 if reject else 0.5,
        reject_h0=reject, test_type="wald",
    )


class TestMSEE:
    def test_zero_when_no_results(self):
        assert compute_msee([]) == pytest.approx(0.0)

    def test_zero_when_ci_contains_zero(self):
        r = _make_eo_result(eo=0.10, ci_lower=-0.05, ci_upper=0.25)
        assert compute_msee([r]) == pytest.approx(0.0)

    def test_positive_eo_uses_ci_lower(self):
        r = _make_eo_result(eo=0.20, ci_lower=0.08, ci_upper=0.32)
        assert compute_msee([r]) == pytest.approx(0.08, abs=1e-9)

    def test_negative_eo_uses_ci_upper(self):
        r = _make_eo_result(eo=-0.20, ci_lower=-0.32, ci_upper=-0.08)
        assert compute_msee([r]) == pytest.approx(0.08, abs=1e-9)

    def test_averaging_includes_non_violations(self):
        """Mean is over all results, not just violations."""
        r_viol = _make_eo_result(eo=0.3, ci_lower=0.12, ci_upper=0.48, reject=True)
        r_null1 = _make_eo_result(eo=0.05, ci_lower=-0.03, ci_upper=0.13)
        r_null2 = _make_eo_result(eo=0.05, ci_lower=-0.03, ci_upper=0.13)
        msee = compute_msee([r_viol, r_null1, r_null2])
        assert msee == pytest.approx(0.12 / 3, abs=1e-9)

    def test_msee_in_saft_eo_summary(self):
        r_viol = _make_eo_result(eo=0.20, ci_lower=0.05, ci_upper=0.35, reject=True)
        r_null = _make_eo_result(eo=0.03, ci_lower=-0.04, ci_upper=0.10)
        summary = saft_eo_summary([r_viol, r_null])
        assert hasattr(summary, "msee")
        assert summary.msee == pytest.approx(compute_msee([r_viol, r_null]), abs=1e-9)

    def test_msee_end_to_end_from_audit(self):
        """MSEE from a real audit run is non-negative and finite."""
        rng = np.random.default_rng(1)
        n = 400
        y_pred = rng.integers(0, 2, size=n)
        y_true = rng.integers(0, 2, size=n)
        mask = rng.random(n) > 0.5
        masks = {"group_A": mask, "group_B": ~mask}
        results = saft_eo_audit(y_pred, y_true, masks, alpha=0.05)
        summary = saft_eo_summary(results)
        assert summary.msee >= 0.0
        assert np.isfinite(summary.msee)


# ---------------------------------------------------------------------------
# EO multiplicity correction
# ---------------------------------------------------------------------------

class TestMultiplicityCorrectionEO:
    def _make_result(self, name, p_value, eo=0.1):
        return EOSAFTResult(
            subgroup=name, n_S_pos=50, min_cell_count=20,
            eo_estimate=eo, ci_lower=eo - 0.05, ci_upper=eo + 0.05,
            p_value=p_value, reject_h0=(p_value < 0.05), test_type="wald",
        )

    def test_bonferroni_eo(self):
        results = [
            self._make_result("A", 0.001),  # survives: 0.001 < 0.05/4
            self._make_result("B", 0.010),  # survives: 0.010 < 0.0125
            self._make_result("C", 0.030),  # fails: 0.030 > 0.0125
            self._make_result("D", 0.200),  # fails
        ]
        corrected = apply_multiplicity_correction_eo(results, method="bonferroni", alpha=0.05)
        assert corrected[0].reject_h0 is True
        assert corrected[1].reject_h0 is True
        assert corrected[2].reject_h0 is False
        assert corrected[3].reject_h0 is False

    def test_bh_eo_step_up(self):
        # same p-values as TestMultiplicityCorrection.test_bh_step_up_procedure
        results = [
            self._make_result("A", 0.03),
            self._make_result("B", 0.005),
            self._make_result("C", 0.50),
            self._make_result("D", 0.01),
            self._make_result("E", 0.04),
        ]
        corrected = apply_multiplicity_correction_eo(results, method="bh", alpha=0.05)
        assert corrected[0].reject_h0 is True   # A: p=0.03
        assert corrected[1].reject_h0 is True   # B: p=0.005
        assert corrected[2].reject_h0 is False  # C: p=0.50
        assert corrected[3].reject_h0 is True   # D: p=0.01
        assert corrected[4].reject_h0 is True   # E: p=0.04

    def test_empty_list(self):
        assert apply_multiplicity_correction_eo([], method="bh") == []

    def test_preserves_other_fields(self):
        results = [self._make_result("X", 0.03, eo=0.15)]
        corrected = apply_multiplicity_correction_eo(results, method="bonferroni")
        assert corrected[0].subgroup == "X"
        assert corrected[0].p_value == pytest.approx(0.03)
        assert corrected[0].eo_estimate == pytest.approx(0.15)
        assert corrected[0].n_S_pos == 50

    def test_invalid_method_raises(self):
        results = [self._make_result("X", 0.01)]
        with pytest.raises(ValueError, match="Unknown correction"):
            apply_multiplicity_correction_eo(results, method="invalid")


# ---------------------------------------------------------------------------
# EO summary
# ---------------------------------------------------------------------------

class TestSAFTEOSummary:
    def _make_result(self, name, n_S_pos, reject, eo=0.1):
        return EOSAFTResult(
            subgroup=name, n_S_pos=n_S_pos, min_cell_count=n_S_pos // 4,
            eo_estimate=eo, ci_lower=eo - 0.05, ci_upper=eo + 0.05,
            p_value=0.001 if reject else 0.5,
            reject_h0=reject, test_type="wald",
        )

    def test_violation_rate(self):
        results = [
            self._make_result("A", 100, True),
            self._make_result("B", 200, False),
            self._make_result("C", 50, True),
            self._make_result("D", 8, False),
        ]
        s = saft_eo_summary(results)
        assert s.n_subgroups == 4
        assert s.n_violations == 2
        assert s.violation_rate == pytest.approx(0.5)

    def test_size_buckets_use_n_s_pos(self):
        results = [
            self._make_result("tiny", 5, True),
            self._make_result("small", 20, False),
            self._make_result("med", 50, True),
            self._make_result("large", 200, False),
        ]
        s = saft_eo_summary(results)
        assert s.violations_by_size_bucket["[2, 10)"]["n_violations"] == 1
        assert s.violations_by_size_bucket["[10, 30)"]["n_violations"] == 0
        assert s.violations_by_size_bucket["[30, 100)"]["n_violations"] == 1
        assert s.violations_by_size_bucket["[100, inf)"]["n_violations"] == 0

    def test_mean_eo_among_violations(self):
        results = [
            self._make_result("A", 100, True,  eo=0.2),
            self._make_result("B", 100, True,  eo=0.4),
            self._make_result("C", 100, False, eo=0.01),
        ]
        s = saft_eo_summary(results)
        assert s.mean_eo_among_violations == pytest.approx(0.3)

    def test_no_violations(self):
        # CI straddles zero → excess = 0 → msee = 0
        r1 = EOSAFTResult(
            subgroup="A", n_S_pos=100, min_cell_count=25,
            eo_estimate=0.05, ci_lower=-0.03, ci_upper=0.13,
            p_value=0.5, reject_h0=False, test_type="wald",
        )
        r2 = EOSAFTResult(
            subgroup="B", n_S_pos=100, min_cell_count=25,
            eo_estimate=-0.04, ci_lower=-0.12, ci_upper=0.04,
            p_value=0.5, reject_h0=False, test_type="wald",
        )
        s = saft_eo_summary([r1, r2])
        assert s.n_violations == 0
        assert s.mean_eo_among_violations == pytest.approx(0.0)
        assert s.msee == pytest.approx(0.0)
