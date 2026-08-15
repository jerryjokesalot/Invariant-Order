"""Streaming change detector with rolling window."""

from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .core import _diff_coeffs


@dataclass
class Alert:
    score: float
    baseline_mean: float
    baseline_std: float
    values_seen: int


class StreamDetector:
    """Real-time change detector for streaming data.

    Maintains a rolling buffer, computes the invariant transform incrementally,
    and compares rolling variance against baseline variance.
    """

    def __init__(self, nuisance="multiplicative", order=1, baseline_window=500,
                 threshold=3.0, variance_window=50):
        """
        Args:
            nuisance: 'multiplicative' or 'additive'.
            order: Differential order m.
            baseline_window: Number of transformed values for baseline estimation.
            threshold: Variance ratio threshold.
            variance_window: Window for rolling variance estimation.
        """
        self.nuisance = nuisance
        self.order = order
        self.baseline_window = baseline_window
        self.threshold = threshold
        self.variance_window = variance_window
        self._coeffs = _diff_coeffs(order)

        self._buffer = deque(maxlen=order + 1)
        self._transform_history = deque(maxlen=baseline_window)
        self._recent_transforms = deque(maxlen=variance_window)
        self._baseline_set = False
        self._baseline_var = 1.0
        self._baseline_mean = 0.0
        self._baseline_std = 1.0
        self._count = 0
        self._last_alert = -baseline_window  # cooldown

    def push(self, value: float) -> Optional[Alert]:
        """Process a new value from the stream.

        Returns:
            Alert if a structural change is detected, None otherwise.
        """
        self._count += 1

        if self.nuisance == "multiplicative":
            if value <= 0:
                return None
            self._buffer.append(np.log(value))
        else:
            self._buffer.append(float(value))

        if len(self._buffer) < self.order + 1:
            return None

        buf = list(self._buffer)
        t_val = sum(self._coeffs[j] * buf[j] for j in range(self.order + 1))
        self._transform_history.append(t_val)
        self._recent_transforms.append(t_val)

        # Build baseline
        if not self._baseline_set:
            if len(self._transform_history) < self.baseline_window:
                return None
            hist = np.array(self._transform_history)
            self._baseline_var = np.var(hist, ddof=1)
            self._baseline_mean = np.mean(np.abs(hist))
            self._baseline_std = np.sqrt(self._baseline_var)
            self._baseline_set = True
            if self._baseline_var < 1e-15:
                self._baseline_var = 1.0
                self._baseline_std = 1.0
            return None

        # Check rolling variance vs baseline
        if len(self._recent_transforms) < self.variance_window:
            return None

        recent = np.array(self._recent_transforms)
        local_var = np.var(recent, ddof=1)
        ratio = local_var / self._baseline_var

        # Cooldown: don't alert again within baseline_window of last alert
        if ratio > self.threshold and (self._count - self._last_alert) > self.variance_window:
            self._last_alert = self._count
            return Alert(
                score=ratio,
                baseline_mean=self._baseline_mean,
                baseline_std=self._baseline_std,
                values_seen=self._count,
            )

        return None

    def reset_baseline(self):
        """Reset baseline statistics from current history."""
        if len(self._transform_history) > 10:
            hist = np.array(self._transform_history)
            self._baseline_var = max(np.var(hist, ddof=1), 1e-15)
            self._baseline_mean = np.mean(np.abs(hist))
            self._baseline_std = np.sqrt(self._baseline_var)
            self._baseline_set = True
            self._last_alert = -self.baseline_window
