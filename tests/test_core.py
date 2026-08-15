"""Tests for core invariant-order computation."""

import numpy as np
import pytest
from invariant_order.core import (
    transform,
    _finite_diff,
    coefficient_moments,
    invariant_order,
    frequency_response,
)


class TestFiniteDiff:
    def test_first_difference(self):
        seq = np.array([1.0, 3.0, 6.0, 10.0])
        result = _finite_diff(seq, 1)
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_second_difference(self):
        seq = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        result = _finite_diff(seq, 2)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

    def test_constant_annihilated_by_first_diff(self):
        seq = np.ones(100) * 42.0
        result = _finite_diff(seq, 1)
        np.testing.assert_allclose(result, 0, atol=1e-14)

    def test_linear_annihilated_by_second_diff(self):
        seq = np.arange(100, dtype=float) * 3.7
        result = _finite_diff(seq, 2)
        np.testing.assert_allclose(result, 0, atol=1e-10)

    def test_quadratic_annihilated_by_third_diff(self):
        n = np.arange(100, dtype=float)
        seq = 2.5 * n ** 2
        result = _finite_diff(seq, 3)
        np.testing.assert_allclose(result, 0, atol=1e-7)


class TestTransform:
    def test_multiplicative_requires_positive(self):
        with pytest.raises(ValueError, match="positive"):
            transform(np.array([1.0, -1.0, 2.0]), nuisance="multiplicative")

    def test_multiplicative_basic(self):
        signal = np.array([1.0, 2.0, 4.0, 8.0])
        result = transform(signal, nuisance="multiplicative", order=1)
        expected = np.diff(np.log(signal))
        np.testing.assert_allclose(result, expected)

    def test_additive_basic(self):
        signal = np.array([1.0, 3.0, 6.0, 10.0])
        result = transform(signal, nuisance="additive", order=1)
        np.testing.assert_allclose(result, [2.0, 3.0, 4.0])

    def test_multiplicative_scale_invariance(self):
        np.random.seed(42)
        signal = np.random.exponential(1.0, 100)
        t1 = transform(signal, nuisance="multiplicative", order=1)
        t2 = transform(signal * 7.3, nuisance="multiplicative", order=1)
        np.testing.assert_allclose(t1, t2, atol=1e-12)

    def test_additive_shift_invariance(self):
        np.random.seed(42)
        signal = np.random.randn(100)
        t1 = transform(signal, nuisance="additive", order=1)
        t2 = transform(signal + 999.0, nuisance="additive", order=1)
        np.testing.assert_allclose(t1, t2, atol=1e-12)

    def test_multiplicative_drift_blindness(self):
        """D^1 should be blind to slow multiplicative drift."""
        np.random.seed(42)
        signal = np.random.exponential(1.0, 1000)
        drift = 1 + 0.5 * np.linspace(0, 1, 1000)
        t_clean = transform(signal, nuisance="multiplicative", order=1)
        t_drifted = transform(signal * drift, nuisance="multiplicative", order=1)
        # Std should be nearly identical (drift adds a smooth trend to log, killed by diff)
        ratio = np.std(t_drifted) / np.std(t_clean)
        assert 0.95 < ratio < 1.05

    def test_order_too_large(self):
        with pytest.raises(ValueError, match="exceed"):
            transform(np.array([1.0, 2.0]), order=3)

    def test_invalid_nuisance(self):
        with pytest.raises(ValueError, match="nuisance"):
            transform(np.array([1.0, 2.0, 3.0]), nuisance="bogus")


class TestCoefficientMoments:
    def test_gap_ratio_exponents(self):
        # s_1/s_2 has exponents [1, -1]
        moments = coefficient_moments([1, -1])
        assert abs(moments[0]) < 1e-14  # mu_0 = 0 (scale invariant)
        assert abs(moments[1] - (-1)) < 1e-14  # mu_1 = -1

    def test_second_difference_exponents(self):
        # [1, -2, 1] → mu_0=0, mu_1=0, mu_2=2
        moments = coefficient_moments([1, -2, 1])
        assert abs(moments[0]) < 1e-14
        assert abs(moments[1]) < 1e-14
        assert abs(moments[2] - 2) < 1e-14


class TestInvariantOrder:
    def test_first_order(self):
        assert invariant_order([1, -1]) == 1

    def test_second_order(self):
        assert invariant_order([1, -2, 1]) == 2

    def test_third_order(self):
        assert invariant_order([-1, 3, -3, 1]) == 3

    def test_random_degree_zero_is_first_order(self):
        np.random.seed(42)
        a = np.random.randn(5)
        a[-1] = -np.sum(a[:-1])  # force degree-0
        assert invariant_order(a) == 1


class TestFrequencyResponse:
    def test_shape(self):
        omegas, mags = frequency_response(2)
        assert len(omegas) == 200
        assert len(mags) == 200

    def test_values_order_1(self):
        omegas = np.array([0.1, 0.5, 1.0])
        _, mags = frequency_response(1, omegas)
        expected = 2 * np.sin(omegas / 2)
        np.testing.assert_allclose(mags, expected)

    def test_suppression_at_low_freq(self):
        _, mags_m1 = frequency_response(1, np.array([0.01]))
        _, mags_m3 = frequency_response(3, np.array([0.01]))
        # Higher order = more suppression at low frequency
        assert mags_m3[0] < mags_m1[0]
