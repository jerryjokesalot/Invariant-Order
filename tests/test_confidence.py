"""Tests for statistical comparison with confidence bounds."""

import numpy as np
import pytest
from invariant_order import compare


class TestCompare:
    def test_detects_structural_change(self):
        """Detect change in variability structure, not just scale."""
        np.random.seed(42)
        before = np.random.exponential(1.0, 500)
        # Inject structural change: multiply by random modulation
        after = np.random.exponential(1.0, 500)
        after *= np.maximum(1 + 0.8 * np.random.randn(500), 0.1)
        result = compare(before, after, nuisance="multiplicative", order=1)
        assert result.p_value < 0.05

    def test_no_false_detection_on_scale_change(self):
        """Scale change should NOT trigger — that's the nuisance."""
        np.random.seed(42)
        before = np.random.exponential(1.0, 1000)
        after = np.random.exponential(5.0, 1000)  # 5x scale is nuisance
        result = compare(before, after, nuisance="multiplicative", order=1,
                         n_permutations=2000)
        assert result.p_value > 0.01

    def test_no_false_detection_same_distribution(self):
        np.random.seed(123)
        before = np.random.exponential(1.0, 1000)
        after = np.random.exponential(1.0, 1000)
        result = compare(before, after, nuisance="multiplicative", order=1,
                         n_permutations=2000)
        assert result.p_value > 0.01

    def test_drift_does_not_trigger(self):
        np.random.seed(42)
        signal = np.random.exponential(1.0, 2000)
        drift = np.linspace(1, 3, 2000)
        drifted = signal * drift
        result = compare(drifted[:1000], drifted[1000:], nuisance="multiplicative",
                         order=1, n_permutations=2000)
        assert result.p_value > 0.01

    def test_change_through_drift_detected(self):
        np.random.seed(42)
        signal = np.random.exponential(1.0, 1000)
        # Real structural change: variability modulation
        signal[500:] *= np.maximum(1 + 0.8 * np.random.randn(500), 0.1)
        drift = np.linspace(1, 3, 1000)
        drifted = signal * drift
        result = compare(drifted[:500], drifted[500:], nuisance="multiplicative", order=1)
        assert result.p_value < 0.05

    def test_additive_comparison(self):
        np.random.seed(42)
        before = np.random.randn(500) * 1.0
        after = np.random.randn(500) * 3.0  # variance change in linear space
        result = compare(before, after, nuisance="additive", order=1)
        assert result.p_value < 0.05

    def test_mean_abs_stat(self):
        np.random.seed(42)
        before = np.random.exponential(1.0, 500)
        after = np.random.exponential(1.0, 500)
        after *= np.maximum(1 + 0.8 * np.random.randn(500), 0.1)
        result = compare(before, after, nuisance="multiplicative", order=1, stat="mean_abs")
        assert result.p_value < 0.05

    def test_result_fields(self):
        np.random.seed(42)
        before = np.random.exponential(1.0, 200)
        after = np.random.exponential(1.0, 200)
        result = compare(before, after, n_permutations=100)
        assert hasattr(result, "effect_size")
        assert hasattr(result, "p_value")
        assert hasattr(result, "stat_before")
        assert hasattr(result, "stat_after")
        assert result.n_permutations == 100
