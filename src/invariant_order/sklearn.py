"""Scikit-learn compatible transformer for invariant feature extraction."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .core import transform


class InvariantScaler(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that extracts drift-invariant features.

    Transforms each row (sample) of X by applying the invariant-order
    transform, then extracting statistical features from the result.
    Downstream models trained on these features are immune to
    multiplicative or additive drift.

    Parameters
    ----------
    nuisance : str, default="multiplicative"
        Type of nuisance to suppress: "multiplicative" or "additive".
    order : int, default=1
        Differential order. Higher order suppresses higher-degree drift.
    features : list of str, optional
        Which summary statistics to extract from each transformed segment.
        Default: ["var", "mean_abs", "max_abs", "p95", "moment2", "exceedance"].
    """

    FEATURE_NAMES = ["var", "mean_abs", "max_abs", "p95", "moment2", "exceedance"]

    def __init__(self, nuisance="multiplicative", order=1, features=None):
        self.nuisance = nuisance
        self.order = order
        self.features = features or self.FEATURE_NAMES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._extract(row) for row in X])

    def _extract(self, signal):
        if self.nuisance == "multiplicative":
            signal = np.abs(signal) + 1e-10
        t = transform(signal, self.nuisance, self.order)
        return self._summarize(t)

    def _summarize(self, t):
        std = np.std(t)
        extractors = {
            "var": lambda: np.var(t),
            "mean_abs": lambda: np.mean(np.abs(t)),
            "max_abs": lambda: np.max(np.abs(t)),
            "p95": lambda: np.percentile(np.abs(t), 95),
            "moment2": lambda: np.mean(t ** 2),
            "exceedance": lambda: np.sum(np.abs(t) > 2 * std) / len(t) if std > 0 else 0.0,
        }
        return np.array([extractors[f]() for f in self.features])

    def get_feature_names_out(self, input_features=None):
        prefix = f"inv_{self.nuisance[0]}{self.order}"
        return np.array([f"{prefix}_{f}" for f in self.features])
