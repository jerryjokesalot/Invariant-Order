"""Tests for streaming change detection."""

import numpy as np
import pytest
from invariant_order import StreamDetector


class TestStreamDetector:
    def test_no_alert_during_baseline(self):
        detector = StreamDetector(nuisance="multiplicative", order=1, baseline_window=100)
        np.random.seed(42)
        for _ in range(100):
            alert = detector.push(np.random.exponential(1.0))
        assert alert is None

    def test_detects_regime_change(self):
        detector = StreamDetector(
            nuisance="multiplicative", order=1,
            baseline_window=200, threshold=2.0, variance_window=50,
        )
        np.random.seed(42)
        alerts = []
        for _ in range(300):
            alert = detector.push(np.random.exponential(1.0))
            if alert:
                alerts.append(alert)
        baseline_alerts = len(alerts)

        # Structural change: variability modulation
        for _ in range(300):
            val = np.random.exponential(1.0) * max(1 + 0.8 * np.random.randn(), 0.1)
            alert = detector.push(val)
            if alert:
                alerts.append(alert)

        assert len(alerts) > baseline_alerts

    def test_blind_to_scale_drift(self):
        detector = StreamDetector(
            nuisance="multiplicative", order=1,
            baseline_window=200, variance_window=50,
        )
        np.random.seed(42)
        alerts = []
        for i in range(600):
            scale = 1 + i / 600
            alert = detector.push(np.random.exponential(scale))
            if alert:
                alerts.append(alert)
        assert len(alerts) == 0

    def test_additive_stream(self):
        detector = StreamDetector(
            nuisance="additive", order=1,
            baseline_window=200, threshold=2.0, variance_window=50,
        )
        np.random.seed(42)
        alerts = []
        for _ in range(300):
            detector.push(np.random.randn())
        for _ in range(300):
            alert = detector.push(np.random.randn() * 3)
            if alert:
                alerts.append(alert)
        assert len(alerts) >= 1

    def test_reset_baseline(self):
        detector = StreamDetector(
            nuisance="multiplicative", order=1,
            baseline_window=100,
        )
        np.random.seed(42)
        for _ in range(150):
            detector.push(np.random.exponential(1.0))
        old_var = detector._baseline_var
        for _ in range(100):
            detector.push(np.random.exponential(1.0) * max(1 + np.random.randn(), 0.1))
        detector.reset_baseline()
        assert detector._baseline_var != old_var
