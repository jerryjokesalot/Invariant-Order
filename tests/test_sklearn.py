"""Tests for sklearn-compatible InvariantScaler."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from invariant_order.sklearn import InvariantScaler


class TestInvariantScaler:
    def test_transform_shape(self):
        X = np.random.exponential(1.0, (10, 100))
        scaler = InvariantScaler()
        out = scaler.transform(X)
        assert out.shape == (10, 6)

    def test_fit_returns_self(self):
        scaler = InvariantScaler()
        result = scaler.fit(np.random.randn(5, 50))
        assert result is scaler

    def test_multiplicative_drift_invariance(self):
        rng = np.random.RandomState(42)
        X = rng.exponential(1.0, (20, 200))
        scaler = InvariantScaler(nuisance="multiplicative", order=1)

        feats_clean = scaler.transform(X)
        feats_drifted = scaler.transform(X * 10.0)

        np.testing.assert_allclose(feats_clean, feats_drifted, rtol=1e-5)

    def test_additive_shift_invariance(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 200)
        scaler = InvariantScaler(nuisance="additive", order=1)

        feats_clean = scaler.transform(X)
        feats_shifted = scaler.transform(X + 1000.0)

        np.testing.assert_allclose(feats_clean, feats_shifted, rtol=1e-10)

    def test_pipeline_integration(self):
        rng = np.random.RandomState(42)
        X = rng.exponential(1.0, (40, 200))
        y = np.array([0]*20 + [1]*20)
        X[20:] *= np.maximum(1 + 1.5 * rng.randn(20, 200), 0.1)

        pipe = Pipeline([
            ("invariant", InvariantScaler()),
            ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
        ])
        pipe.fit(X, y)

        # Test on drifted data
        X_test = rng.exponential(1.0, (20, 200))
        X_test[10:] *= np.maximum(1 + 1.5 * rng.randn(10, 200), 0.1)
        y_test = np.array([0]*10 + [1]*10)
        X_drifted = X_test * 50.0

        acc = pipe.score(X_drifted, y_test)
        assert acc >= 0.8

    def test_pipeline_with_logistic_regression(self):
        rng = np.random.RandomState(42)
        X = rng.exponential(1.0, (40, 200))
        y = np.array([0]*20 + [1]*20)
        X[20:] *= np.maximum(1 + 1.5 * rng.randn(20, 200), 0.1)

        pipe = Pipeline([
            ("invariant", InvariantScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=1000)),
        ])
        pipe.fit(X, y)

        X_test = rng.exponential(1.0, (20, 200))
        X_test[10:] *= np.maximum(1 + 1.5 * rng.randn(10, 200), 0.1)
        y_test = np.array([0]*10 + [1]*10)
        X_drifted = X_test * 50.0

        acc = pipe.score(X_drifted, y_test)
        assert acc >= 0.8

    def test_custom_features(self):
        X = np.random.exponential(1.0, (5, 100))
        scaler = InvariantScaler(features=["var", "mean_abs"])
        out = scaler.transform(X)
        assert out.shape == (5, 2)

    def test_feature_names_out(self):
        scaler = InvariantScaler(nuisance="multiplicative", order=2)
        names = scaler.get_feature_names_out()
        assert len(names) == 6
        assert names[0] == "inv_m2_var"

    def test_1d_input(self):
        X = np.random.exponential(1.0, 100)
        scaler = InvariantScaler()
        out = scaler.transform(X)
        assert out.shape == (1, 6)

    def test_higher_order_drift(self):
        """Order 2 should be invariant to quadratic drift."""
        rng = np.random.RandomState(42)
        X = rng.exponential(1.0, (10, 300))
        quadratic_drift = (1 + np.linspace(0, 1, 300)**2)
        X_drifted = X * quadratic_drift

        scaler = InvariantScaler(nuisance="multiplicative", order=2)
        feats_clean = scaler.transform(X)
        feats_drifted = scaler.transform(X_drifted)

        np.testing.assert_allclose(feats_clean, feats_drifted, rtol=1e-4)
