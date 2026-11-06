"""Tests for data pipeline: intersections and noise injection."""

import numpy as np
import pandas as pd
import pytest
import torch

from asbr.data.intersections import build_subgroup_masks, masks_to_torch, subgroup_summary
from asbr.data.noise_injection import inject_noise_groups


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_attributes():
    """Simple protected attribute DataFrame for testing."""
    return pd.DataFrame({
        "race": ["A", "A", "A", "B", "B", "B", "A", "B", "A", "B"],
        "sex": ["M", "F", "M", "F", "M", "F", "F", "M", "M", "F"],
    })


@pytest.fixture
def three_attr():
    """Three protected attributes for level-3 testing."""
    n = 24
    return pd.DataFrame({
        "race": (["A"] * 12) + (["B"] * 12),
        "sex": (["M", "F"] * 6) + (["M", "F"] * 6),
        "age": (["young", "young", "old", "old"] * 3) + (["young", "young", "old", "old"] * 3),
    })


# ---------------------------------------------------------------------------
# Intersections
# ---------------------------------------------------------------------------

class TestBuildSubgroupMasks:
    def test_level1_creates_single_attr_groups(self, sample_attributes):
        masks = build_subgroup_masks(sample_attributes, level=1)
        # Should have groups for race=A, race=B, sex=M, sex=F
        assert len(masks) == 4
        assert any("race=A" in name for name in masks)
        assert any("sex=M" in name for name in masks)

    def test_level2_creates_pairwise_groups(self, sample_attributes):
        masks = build_subgroup_masks(sample_attributes, level=2)
        # Level 1 (4 groups) + Level 2 (pairwise combos)
        pairwise = [n for n in masks if "&" in n]
        assert len(pairwise) > 0
        assert any("race=A&sex=M" in name for name in masks)

    def test_level3_creates_three_way_groups(self, three_attr):
        masks = build_subgroup_masks(three_attr, level=3)
        # Should have level-3 groups with two & separators
        three_way = [n for n in masks if n.count("&") == 2]
        assert len(three_way) > 0

    def test_masks_are_boolean_correct_length(self, sample_attributes):
        masks = build_subgroup_masks(sample_attributes, level=1)
        for name, mask in masks.items():
            assert mask.dtype == bool
            assert len(mask) == len(sample_attributes)

    def test_masks_cover_all_samples_at_level1(self, sample_attributes):
        masks = build_subgroup_masks(sample_attributes, level=1)
        # For each attribute, every sample belongs to exactly one value
        race_masks = {n: m for n, m in masks.items() if n.startswith("race=")}
        combined = np.zeros(len(sample_attributes), dtype=bool)
        for m in race_masks.values():
            combined |= m
        assert combined.all()

    def test_min_size_filters_small_groups(self, sample_attributes):
        masks_all = build_subgroup_masks(sample_attributes, level=2, min_size=1)
        masks_filtered = build_subgroup_masks(sample_attributes, level=2, min_size=4)
        assert len(masks_filtered) <= len(masks_all)

    def test_empty_dataframe(self):
        A = pd.DataFrame({"race": pd.Series([], dtype=str)})
        masks = build_subgroup_masks(A, level=1)
        assert len(masks) == 0


class TestMasksToTorch:
    def test_conversion_preserves_values(self, sample_attributes):
        masks_np = build_subgroup_masks(sample_attributes, level=1)
        masks_torch = masks_to_torch(masks_np)
        for name in masks_np:
            assert torch.equal(
                masks_torch[name],
                torch.tensor(masks_np[name], dtype=torch.bool),
            )

    def test_device_placement(self, sample_attributes):
        masks_np = build_subgroup_masks(sample_attributes, level=1)
        masks_torch = masks_to_torch(masks_np, device="cpu")
        for t in masks_torch.values():
            assert t.device == torch.device("cpu")


class TestSubgroupSummary:
    def test_summary_has_correct_columns(self, sample_attributes):
        masks = build_subgroup_masks(sample_attributes, level=1)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        summary = subgroup_summary(masks, y)
        assert "subgroup" in summary.columns
        assert "size" in summary.columns
        assert "fraction" in summary.columns
        assert "pos_rate" in summary.columns

    def test_summary_without_labels(self, sample_attributes):
        masks = build_subgroup_masks(sample_attributes, level=1)
        summary = subgroup_summary(masks)
        assert "pos_rate" not in summary.columns
        assert len(summary) == len(masks)

    def test_fractions_sum_for_single_attr(self, sample_attributes):
        masks = build_subgroup_masks(sample_attributes, level=1)
        race_masks = {n: m for n, m in masks.items() if n.startswith("race=")}
        summary = subgroup_summary(race_masks)
        assert abs(summary["fraction"].sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Noise Injection
# ---------------------------------------------------------------------------

class TestNoiseInjection:
    def test_correct_number_of_groups(self):
        masks = inject_noise_groups(n=100, n_groups=5, sizes=(10, 20))
        assert len(masks) == 10  # 5 groups * 2 sizes

    def test_correct_group_sizes(self):
        masks = inject_noise_groups(n=100, n_groups=3, sizes=(15,))
        for name, mask in masks.items():
            assert mask.sum() == 15

    def test_names_encode_size(self):
        masks = inject_noise_groups(n=100, n_groups=2, sizes=(10, 25))
        names = list(masks.keys())
        assert any("noise_10_" in n for n in names)
        assert any("noise_25_" in n for n in names)

    def test_reproducibility(self):
        m1 = inject_noise_groups(n=100, n_groups=5, sizes=(10,), seed=42)
        m2 = inject_noise_groups(n=100, n_groups=5, sizes=(10,), seed=42)
        for name in m1:
            assert np.array_equal(m1[name], m2[name])

    def test_different_seeds_differ(self):
        m1 = inject_noise_groups(n=1000, n_groups=5, sizes=(50,), seed=1)
        m2 = inject_noise_groups(n=1000, n_groups=5, sizes=(50,), seed=2)
        # At least one group should differ
        any_diff = any(
            not np.array_equal(m1[name], m2[name])
            for name in m1 if name in m2
        )
        assert any_diff

    def test_skips_sizes_too_large(self):
        masks = inject_noise_groups(n=10, n_groups=3, sizes=(5, 15))
        # Size 15 >= n=10, should be skipped
        assert all("noise_15" not in name for name in masks)
        assert len(masks) == 3  # Only size=5 groups
