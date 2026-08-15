# Invariant Order

Distribution-independent change detection through algebraic invariance. Detect structural changes in signals while ignoring drift, scaling, and other nuisance transformations.

## The Problem

A sensor drifts over time. A machine starts failing. Traditional detectors (CUSUM, control charts) fire on both — they can't tell nuisance from signal. You end up tuning thresholds, detrending by hand, or drowning in false alarms.

## The Idea

The **Invariant Differential Order Theorem** says: for monomial scale-invariant observables, finite differences of a specific order annihilate nuisance transformations *exactly* — not approximately, not asymptotically, but algebraically. What survives is structural change only.

This isn't statistics. It's algebra. It works on any distribution.

## Install

```bash
pip install invariant-order
```

Or from source:

```bash
git clone https://github.com/jerryjokesalot/Invariant-Order.git
cd Invariant-Order
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import invariant_order as io

# A drifting signal with a structural change at sample 3000
signal = np.random.exponential(1.0, 5000)
signal *= 1 + 0.5 * np.linspace(0, 1, 5000)   # drift (nuisance)
signal[3000:] *= np.maximum(1 + 2*np.random.randn(2000), 0.1)  # failure (real)

# Batch scan — finds the change, ignores the drift
result = io.scan(signal, nuisance="multiplicative", order=1, threshold=2.0)
for cp in result.change_points:
    print(f"Change at {cp.index}, score={cp.score:.1f}")

# Confirm: no false alarms on drift alone
clean = np.random.exponential(1.0, 5000) * (1 + 0.5 * np.linspace(0, 1, 5000))
assert len(io.scan(clean).change_points) == 0
```

## Three Ways to Detect

**Batch scan** — scan a full signal for change points:

```python
result = io.scan(signal, nuisance="multiplicative", order=1)
```

**Compare** — test whether two segments differ structurally (permutation test):

```python
result = io.compare(before, after, nuisance="multiplicative", order=1)
print(f"p={result.p_value:.4f}, effect={result.effect_size:.2f}")
```

**Stream** — real-time detection, one sample at a time:

```python
stream = io.StreamDetector(nuisance="multiplicative", order=1)
for value in live_data:
    alert = stream.push(value)
    if alert:
        print(f"Structural change detected (score={alert.score:.1f})")
```

## ML Preprocessing: Drift-Immune Features

The `InvariantScaler` drops into any scikit-learn pipeline and makes your model immune to sensor drift — with a mathematical guarantee:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from invariant_order.sklearn import InvariantScaler

pipe = Pipeline([
    ("invariant", InvariantScaler(nuisance="multiplicative", order=1)),
    ("clf", RandomForestClassifier()),
])
pipe.fit(X_train, y_train)
pipe.predict(X_drifted)  # works at 2x, 10x, 50x drift — same accuracy
```

Tested on NASA IMS bearing data: raw features drop to 33% accuracy at 2x drift. Invariant features hold at 98.5% at 10x drift. See [`experiments/invariant_features_ml.py`](experiments/invariant_features_ml.py).

## Nuisance Types

| Nuisance | Transform | What it ignores |
|---|---|---|
| `multiplicative` | log → finite diff | Scaling, multiplicative drift |
| `additive` | finite diff | Shifts, additive drift |

Higher `order` suppresses higher-degree polynomial drift. Order 1 handles linear drift, order 2 handles quadratic, etc.

## API Reference

| Function / Class | Purpose |
|---|---|
| `io.scan(signal, ...)` | Batch change-point detection |
| `io.compare(before, after, ...)` | Permutation-based two-sample test |
| `io.StreamDetector(...)` | Real-time streaming detection |
| `io.transform(signal, nuisance, order)` | Raw invariant transform |
| `io.coefficient_moments(exponents)` | Compute coefficient moments μ_k |
| `io.invariant_order(exponents)` | Determine invariant differential order |
| `io.frequency_response(order, omegas)` | Frequency response [2sin(ω/2)]^m |
| `InvariantScaler(...)` | Sklearn-compatible drift-immune feature transformer |

## How It Works

1. **Transform**: Map the signal into a space where nuisance transformations become additive (log for multiplicative nuisance), then apply m-th order finite differences
2. **Annihilation**: Polynomial drift of degree < m is killed exactly (Vandermonde moment conditions)
3. **Detection**: Compare local variance of the transformed signal against a baseline — structural changes survive the transform, nuisance doesn't

The key insight: this is a **theorem**, not a heuristic. The annihilation is exact for any sample size, any distribution, any drift magnitude.

## Examples

See [`examples/industrial_sensor.py`](examples/industrial_sensor.py) for a complete walkthrough: simulated bearing failure detected through sensor drift.

## Requirements

- Python >= 3.9
- NumPy >= 1.20
- SciPy >= 1.7 (optional, for `compare()`)
- scikit-learn >= 1.0 (optional, for `InvariantScaler`; install with `pip install invariant-order[ml]`)

## License

MIT
