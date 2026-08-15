"""Tests for batch change-point detection."""

import numpy as np
import pytest
from invariant_order import Detector, scan


class TestDetector:
    def test_detects_variance_change(self):
        np.random.seed(42)
        signal = np.random.exponential(1.0, 2000)
        signal[1000:] *= np.maximum(1 + 0.8 * np.random.randn(1000), 0.1)
        result = Detector(nuisance="multiplicative", order=1, threshold=2.0).scan(signal)
        assert len(result.change_points) >= 1
        closest = min(result.change_points, key=lambda c: abs(c.index - 1000))
        assert abs(closest.index - 1000) < 300

    def test_no_false_alarm_on_stable_signal(self):
        np.random.seed(42)
        signal = np.random.exponential(1.0, 2000)
        result = Detector(nuisance="multiplicative", order=1).scan(signal)
        assert len(result.change_points) == 0

    def test_blind_to_multiplicative_drift(self):
        np.random.seed(42)
        signal = np.random.exponential(1.0, 2000)
        drift = np.linspace(1, 3, 2000)
        result = Detector(nuisance="multiplicative", order=1).scan(signal * drift)
        assert len(result.change_points) == 0

    def test_detects_change_through_drift(self):
        np.random.seed(42)
        signal = np.random.exponential(1.0, 2000)
        drift = np.linspace(1, 3, 2000)
        signal[1000:] *= np.maximum(1 + 0.8 * np.random.randn(1000), 0.1)
        result = Detector(nuisance="multiplicative", order=1, threshold=2.0).scan(signal * drift)
        assert len(result.change_points) >= 1

    def test_additive_detects_variance_change(self):
        np.random.seed(42)
        signal = np.random.randn(2000)
        signal[1000:] *= 3.0
        result = Detector(nuisance="additive", order=1, threshold=2.0).scan(signal)
        assert len(result.change_points) >= 1

    def test_additive_blind_to_linear_drift(self):
        np.random.seed(42)
        signal = np.random.randn(2000)
        drift = np.linspace(0, 100, 2000)
        result = Detector(nuisance="additive", order=1).scan(signal + drift)
        assert len(result.change_points) == 0

    def test_scan_result_has_transformed_signal(self):
        signal = np.random.exponential(1.0, 100)
        result = Detector(nuisance="multiplicative", order=2).scan(signal)
        assert result.transformed is not None
        assert len(result.transformed) == 98

    def test_scan_convenience_function(self):
        np.random.seed(42)
        signal = np.random.exponential(1.0, 2000)
        signal[1000:] *= np.maximum(1 + 0.8 * np.random.randn(1000), 0.1)
        result = scan(signal, nuisance="multiplicative", order=1, threshold=2.0)
        assert len(result.change_points) >= 1


class TestDetectorHigherOrder:
    def test_order_2_detects_curvature_change(self):
        np.random.seed(42)
        n = 4000
        signal = np.random.exponential(1.0, n)
        # Strong variability ramp in second half
        for i in range(2000, n):
            signal[i] *= np.maximum(1 + 0.5 * np.random.randn() * (i - 2000) / 1000, 0.1)
        result = Detector(nuisance="multiplicative", order=2, threshold=2.0).scan(signal)
        assert len(result.change_points) >= 1
