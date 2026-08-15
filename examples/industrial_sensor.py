#!/usr/bin/env python3
"""Industrial sensor example: detect structural change through baseline drift.

A machine vibration sensor drifts slowly over time (nuisance).
At some point, a bearing starts failing (structural change).
Can we detect the failure while ignoring the drift?
"""

import numpy as np
import invariant_order as io

np.random.seed(42)

# Simulate 5000 vibration interval readings
N = 5000
intervals = np.random.exponential(1.0, N)

# Nuisance: sensor baseline drifts by 50% over the measurement period
drift = 1 + 0.5 * np.linspace(0, 1, N)
drifted = intervals * drift

# Real event: bearing failure at sample 3000 causes variability to change
signal = drifted.copy()
signal[3000:] *= np.maximum(1 + 2 * np.random.randn(2000), 0.1)

# --- Traditional approach: raw signal comparison ---
raw_before = np.mean(signal[:2500])
raw_after = np.mean(signal[2500:])
print(f"Raw signal ratio (after/before): {raw_after/raw_before:.3f}")
print(f"  But is that drift or failure? Can't tell.\n")

# --- Invariant Order approach ---

# Batch scan (threshold=2.0 for sensitive detection)
result = io.scan(signal, nuisance="multiplicative", order=1, threshold=2.0)
print(f"Change points detected: {len(result.change_points)}")
for cp in result.change_points:
    print(f"  Index {cp.index}: score={cp.score:.1f}")

# Confirm no false alarm on drift alone
result_drift = io.scan(drifted, nuisance="multiplicative", order=1)
print(f"\nFalse alarms on drift-only signal: {len(result_drift.change_points)}")

# Statistical comparison
comparison = io.compare(
    signal[:3000], signal[3000:],
    nuisance="multiplicative", order=1,
)
print(f"\nStatistical comparison:")
print(f"  Effect size: {comparison.effect_size:.2f}")
print(f"  p-value:     {comparison.p_value:.4f}")

# Streaming detection
print(f"\nStreaming detection:")
stream = io.StreamDetector(nuisance="multiplicative", order=1, baseline_window=500)
for i, value in enumerate(signal):
    alert = stream.push(value)
    if alert:
        print(f"  ALERT at sample {i}: score={alert.score:.1f}")
        break
