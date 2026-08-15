"""Batch change-point detection on invariant-transformed signals."""

from dataclasses import dataclass, field
import numpy as np
from .core import transform


@dataclass
class ChangePoint:
    index: int
    score: float
    baseline_std: float


@dataclass
class ScanResult:
    change_points: list = field(default_factory=list)
    transformed: np.ndarray = field(default=None)
    baseline_mean: float = 0.0
    baseline_std: float = 0.0


class Detector:
    """Batch change-point detector using invariant-order transformation.

    Applies the invariant transform, then compares local variance in sliding
    windows against a baseline. Alerts when the variance ratio exceeds threshold.
    """

    def __init__(self, nuisance="multiplicative", order=1, baseline_fraction=0.25,
                 threshold=3.0, window_size=None):
        """
        Args:
            nuisance: 'multiplicative' or 'additive'.
            order: Differential order m.
            baseline_fraction: Fraction of signal used for baseline.
            threshold: Variance ratio threshold. Alert when local_var > threshold * baseline_var.
            window_size: Size of variance-estimation windows. Default: auto.
        """
        self.nuisance = nuisance
        self.order = order
        self.baseline_fraction = baseline_fraction
        self.threshold = threshold
        self.window_size = window_size

    def scan(self, signal):
        """Scan a signal for structural change points."""
        signal = np.asarray(signal, dtype=float)
        transformed = transform(signal, self.nuisance, self.order)

        n = len(transformed)
        w = self.window_size or max(20, n // 20)
        n_windows = n // w
        if n_windows < 4:
            return ScanResult(transformed=transformed)

        window_vars = np.array([
            np.var(transformed[i * w:(i + 1) * w], ddof=1)
            for i in range(n_windows)
        ])

        baseline_n = max(2, int(n_windows * self.baseline_fraction))
        baseline_var = np.median(window_vars[:baseline_n])

        if baseline_var < 1e-15:
            return ScanResult(transformed=transformed, baseline_std=0.0)

        change_points = []
        for i in range(baseline_n, n_windows):
            ratio = window_vars[i] / baseline_var
            if ratio > self.threshold:
                change_points.append(ChangePoint(
                    index=i * w + self.order,
                    score=ratio,
                    baseline_std=np.sqrt(baseline_var),
                ))

        return ScanResult(
            change_points=change_points,
            transformed=transformed,
            baseline_mean=np.mean(np.abs(transformed[:baseline_n * w])),
            baseline_std=np.sqrt(baseline_var),
        )
