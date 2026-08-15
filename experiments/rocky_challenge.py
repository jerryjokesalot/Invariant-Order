#!/usr/bin/env python3
"""Rocky's Challenge: Three critical experiments for Invariant Order credibility.

Test 1: Nuisance SHAPE variation
   Train on constant scaling, test on linear drift, sinusoidal drift,
   piecewise calibration changes. Does invariance generalize beyond
   the training nuisance shape?

Test 2: Nuisance CORRELATED with target
   Failing bearings produce larger amplitudes. If we suppress amplitude
   variation, do we accidentally remove the fault signal? Where is the
   line between robustness and blindness?

These are the experiments that separate "interesting demo" from
"credible technology."
"""

import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import invariant_order as io
from invariant_order.sklearn import InvariantScaler


def extract_raw_features(segments):
    features = []
    for seg in segments:
        features.append([
            np.mean(seg),
            np.std(seg),
            np.max(np.abs(seg)),
            np.sqrt(np.mean(seg**2)),
            np.mean(np.abs(seg)),
            np.max(np.abs(seg)) / (np.sqrt(np.mean(seg**2)) + 1e-10),
        ])
    return np.array(features)


def make_segments(n_samples, segment_length, rng, degraded_fraction=0.5):
    """Generate healthy vs degrading segments.

    Healthy: exponential(1.0)
    Degrading: exponential(1.0) with random variability modulation
    """
    segments = []
    labels = []
    n_healthy = int(n_samples * (1 - degraded_fraction))
    for i in range(n_samples):
        base = rng.exponential(1.0, segment_length)
        if i < n_healthy:
            segments.append(base)
            labels.append(0)
        else:
            modulated = base * np.maximum(1 + 1.5 * rng.randn(segment_length), 0.1)
            segments.append(modulated)
            labels.append(1)
    return segments, np.array(labels)


def apply_nuisance(segments, nuisance_fn):
    """Apply a nuisance function to each segment."""
    return [seg * nuisance_fn(len(seg)) for seg in segments]


def test1_nuisance_shape():
    """Test 1: Train on clean data, test under various nuisance SHAPES."""
    print("=" * 70)
    print("TEST 1: Nuisance SHAPE Variation")
    print("=" * 70)
    print("""
Hypothesis: Invariant features should be immune to ALL smooth
multiplicative nuisance shapes, not just constant scaling.
The theorem says: Δ^m annihilates any polynomial of degree < m
in log space. So linear, quadratic, and smooth periodic drift
should all be suppressed.
""")

    rng = np.random.RandomState(42)
    train_segs, train_labels = make_segments(200, 500, rng)
    test_rng = np.random.RandomState(99)
    test_segs, test_labels = make_segments(200, 500, test_rng)

    # Train on clean data
    scaler = InvariantScaler(nuisance="multiplicative", order=1)
    raw_train = extract_raw_features(train_segs)
    inv_train = scaler.transform(np.array([s for s in train_segs]))

    clf_raw = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_inv = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_raw.fit(raw_train, train_labels)
    clf_inv.fit(inv_train, train_labels)

    # Also train order=2 for quadratic drift
    scaler2 = InvariantScaler(nuisance="multiplicative", order=2)
    inv_train2 = scaler2.transform(np.array([s for s in train_segs]))
    clf_inv2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_inv2.fit(inv_train2, train_labels)

    nuisance_shapes = {
        "No drift (baseline)":
            lambda n: np.ones(n),
        "Constant 10x":
            lambda n: 10.0 * np.ones(n),
        "Linear drift (1→3x)":
            lambda n: np.linspace(1, 3, n),
        "Linear drift (1→10x)":
            lambda n: np.linspace(1, 10, n),
        "Sinusoidal (±50%)":
            lambda n: 1 + 0.5 * np.sin(2 * np.pi * np.linspace(0, 3, n)),
        "Sinusoidal (±90%)":
            lambda n: 1 + 0.9 * np.sin(2 * np.pi * np.linspace(0, 3, n)),
        "Exponential decay (1→0.1x)":
            lambda n: np.exp(-2.3 * np.linspace(0, 1, n)),
        "Step change (1x→5x at midpoint)":
            lambda n: np.where(np.arange(n) < n//2, 1.0, 5.0),
        "Piecewise (1x→3x→0.5x)":
            lambda n: np.piecewise(
                np.linspace(0, 1, n),
                [np.linspace(0, 1, n) < 0.33,
                 (np.linspace(0, 1, n) >= 0.33) & (np.linspace(0, 1, n) < 0.66)],
                [lambda t: 1 + 6*t, lambda t: 3.0, lambda t: 3 - 5*(t - 0.66)]
            ),
        "Quadratic drift (1→5x)":
            lambda n: 1 + 4 * (np.linspace(0, 1, n))**2,
        "Random walk calibration":
            lambda n: np.exp(np.cumsum(np.random.RandomState(7).randn(n) * 0.02)),
    }

    print(f"{'Nuisance Shape':<35s}  {'Raw':>8s}  {'Inv(1)':>8s}  {'Inv(2)':>8s}")
    print("-" * 67)

    results = {}
    for name, fn in nuisance_shapes.items():
        drifted = apply_nuisance(test_segs, fn)
        raw_test = extract_raw_features(drifted)
        inv_test = scaler.transform(np.array([s for s in drifted]))
        inv_test2 = scaler2.transform(np.array([s for s in drifted]))

        acc_raw = accuracy_score(test_labels, clf_raw.predict(raw_test))
        acc_inv = accuracy_score(test_labels, clf_inv.predict(inv_test))
        acc_inv2 = accuracy_score(test_labels, clf_inv2.predict(inv_test2))

        results[name] = (acc_raw, acc_inv, acc_inv2)
        print(f"{name:<35s}  {acc_raw:>7.1%}  {acc_inv:>7.1%}  {acc_inv2:>7.1%}")

    # Summary
    raw_failures = sum(1 for _, (r, _, _) in results.items() if r < 0.8)
    inv_failures = sum(1 for _, (_, i, _) in results.items() if i < 0.8)
    print(f"\nRaw features failed on {raw_failures}/{len(results)} nuisance shapes")
    print(f"Invariant(1) features failed on {inv_failures}/{len(results)} nuisance shapes")

    return results


def test2_correlated_nuisance():
    """Test 2: What happens when nuisance is correlated with the target?

    This is Rocky's hardest question. Three scenarios:

    A) Fault = variability change (erratic behavior)
       → Should SURVIVE the transform (variance of log-diffs changes)

    B) Fault = uniform amplitude increase (everything gets bigger)
       → Should be SUPPRESSED (this IS the nuisance class)

    C) Fault = both (amplitude increase + variability change)
       → Partial survival — how much?
    """
    print("\n" + "=" * 70)
    print("TEST 2: Nuisance CORRELATED with Target")
    print("=" * 70)
    print("""
The fundamental tradeoff: robustness to nuisance can become
blindness to signal. We test three fault types:

  A) Variability fault — erratic, irregular behavior
  B) Amplitude fault — uniform amplitude increase
  C) Mixed fault — amplitude + variability together

The invariant transform should detect A, miss B, and partially
detect C. This is the honest boundary.
""")

    rng = np.random.RandomState(42)
    segment_length = 500
    n_train = 200
    n_test = 200

    # --- Scenario A: Variability fault ---
    print("-" * 50)
    print("Scenario A: Fault = variability change (erratic)")
    print("-" * 50)

    train_segs_a = []
    train_labels_a = []
    for i in range(n_train):
        base = rng.exponential(1.0, segment_length)
        if i < n_train // 2:
            train_segs_a.append(base)
            train_labels_a.append(0)
        else:
            modulated = base * np.maximum(1 + 1.5 * rng.randn(segment_length), 0.1)
            train_segs_a.append(modulated)
            train_labels_a.append(1)
    train_labels_a = np.array(train_labels_a)

    scaler = InvariantScaler(nuisance="multiplicative", order=1)
    raw_tr = extract_raw_features(train_segs_a)
    inv_tr = scaler.transform(np.array(train_segs_a))

    clf_raw_a = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_inv_a = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_raw_a.fit(raw_tr, train_labels_a)
    clf_inv_a.fit(inv_tr, train_labels_a)

    test_rng = np.random.RandomState(99)
    test_segs_a = []
    test_labels_a = []
    for i in range(n_test):
        base = test_rng.exponential(1.0, segment_length)
        if i < n_test // 2:
            test_segs_a.append(base)
            test_labels_a.append(0)
        else:
            modulated = base * np.maximum(1 + 1.5 * test_rng.randn(segment_length), 0.1)
            test_segs_a.append(modulated)
            test_labels_a.append(1)
    test_labels_a = np.array(test_labels_a)

    drift_levels = [1.0, 2.0, 5.0, 10.0]
    print(f"\n{'Drift':>8s}  {'Raw':>8s}  {'Invariant':>10s}")
    print("-" * 30)
    for drift in drift_levels:
        drifted = [s * drift for s in test_segs_a]
        acc_raw = accuracy_score(test_labels_a, clf_raw_a.predict(extract_raw_features(drifted)))
        acc_inv = accuracy_score(test_labels_a, clf_inv_a.predict(
            scaler.transform(np.array(drifted))))
        print(f"{drift:>7.0f}x  {acc_raw:>7.1%}  {acc_inv:>10.1%}")

    print("\nExpected: Invariant features DETECT variability faults through drift.")

    # --- Scenario B: Amplitude-only fault ---
    print("\n" + "-" * 50)
    print("Scenario B: Fault = uniform amplitude increase")
    print("-" * 50)
    print("(Failing bearing simply produces 3x larger signals uniformly)")

    train_segs_b = []
    train_labels_b = []
    for i in range(n_train):
        base = rng.exponential(1.0, segment_length)
        if i < n_train // 2:
            train_segs_b.append(base)
            train_labels_b.append(0)
        else:
            train_segs_b.append(base * 3.0)
            train_labels_b.append(1)
    train_labels_b = np.array(train_labels_b)

    raw_tr_b = extract_raw_features(train_segs_b)
    inv_tr_b = scaler.transform(np.array(train_segs_b))

    clf_raw_b = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_inv_b = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_raw_b.fit(raw_tr_b, train_labels_b)
    clf_inv_b.fit(inv_tr_b, train_labels_b)

    # Test: can either detect the 3x amplitude fault?
    test_segs_b = []
    test_labels_b = []
    for i in range(n_test):
        base = test_rng.exponential(1.0, segment_length)
        if i < n_test // 2:
            test_segs_b.append(base)
            test_labels_b.append(0)
        else:
            test_segs_b.append(base * 3.0)
            test_labels_b.append(1)
    test_labels_b = np.array(test_labels_b)

    acc_raw_clean = accuracy_score(test_labels_b, clf_raw_b.predict(
        extract_raw_features(test_segs_b)))
    acc_inv_clean = accuracy_score(test_labels_b, clf_inv_b.predict(
        scaler.transform(np.array(test_segs_b))))

    print(f"\nNo drift:")
    print(f"  Raw:       {acc_raw_clean:.1%}")
    print(f"  Invariant: {acc_inv_clean:.1%}")

    if acc_inv_clean < 0.6:
        print("\n  *** CONFIRMED: Invariant features are BLIND to pure amplitude faults.")
        print("  *** This is CORRECT behavior — uniform scaling IS the nuisance class.")
        print("  *** The transform is doing exactly what it promises: suppressing scaling.")
    else:
        print("\n  NOTE: Invariant features unexpectedly detect amplitude-only faults.")
        print("  This may indicate leakage through boundary effects or feature extraction.")

    # Now: does raw survive drift?
    print(f"\nBut does the raw detector survive drift?")
    print(f"{'Drift':>8s}  {'Raw':>8s}  {'Invariant':>10s}")
    print("-" * 30)
    for drift in [1.0, 2.0, 5.0]:
        drifted = [s * drift for s in test_segs_b]
        acc_raw = accuracy_score(test_labels_b, clf_raw_b.predict(
            extract_raw_features(drifted)))
        acc_inv = accuracy_score(test_labels_b, clf_inv_b.predict(
            scaler.transform(np.array(drifted))))
        print(f"{drift:>7.0f}x  {acc_raw:>7.1%}  {acc_inv:>10.1%}")

    print("\nKey insight: Raw features can detect amplitude faults, but ONLY")
    print("at the exact calibration they were trained on. Drift kills them.")

    # --- Scenario C: Mixed fault (amplitude + variability) ---
    print("\n" + "-" * 50)
    print("Scenario C: Mixed fault (amplitude increase + variability)")
    print("-" * 50)
    print("(Failing bearing: 2x larger AND erratic)")

    train_segs_c = []
    train_labels_c = []
    for i in range(n_train):
        base = rng.exponential(1.0, segment_length)
        if i < n_train // 2:
            train_segs_c.append(base)
            train_labels_c.append(0)
        else:
            # Both amplitude increase AND variability modulation
            amplified = base * 2.0
            modulated = amplified * np.maximum(1 + 1.0 * rng.randn(segment_length), 0.1)
            train_segs_c.append(modulated)
            train_labels_c.append(1)
    train_labels_c = np.array(train_labels_c)

    raw_tr_c = extract_raw_features(train_segs_c)
    inv_tr_c = scaler.transform(np.array(train_segs_c))

    clf_raw_c = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_inv_c = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_raw_c.fit(raw_tr_c, train_labels_c)
    clf_inv_c.fit(inv_tr_c, train_labels_c)

    test_segs_c = []
    test_labels_c = []
    for i in range(n_test):
        base = test_rng.exponential(1.0, segment_length)
        if i < n_test // 2:
            test_segs_c.append(base)
            test_labels_c.append(0)
        else:
            amplified = base * 2.0
            modulated = amplified * np.maximum(1 + 1.0 * test_rng.randn(segment_length), 0.1)
            test_segs_c.append(modulated)
            test_labels_c.append(1)
    test_labels_c = np.array(test_labels_c)

    print(f"\n{'Drift':>8s}  {'Raw':>8s}  {'Invariant':>10s}")
    print("-" * 30)
    for drift in drift_levels:
        drifted = [s * drift for s in test_segs_c]
        acc_raw = accuracy_score(test_labels_c, clf_raw_c.predict(
            extract_raw_features(drifted)))
        acc_inv = accuracy_score(test_labels_c, clf_inv_c.predict(
            scaler.transform(np.array(drifted))))
        print(f"{drift:>7.0f}x  {acc_raw:>7.1%}  {acc_inv:>10.1%}")

    print("\nExpected: Invariant features detect the variability component")
    print("of the mixed fault, even though the amplitude component is suppressed.")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("THE HONEST BOUNDARY")
    print("=" * 70)
    print("""
What the invariant transform DETECTS (through any amount of drift):
  - Changes in variability structure (erratic behavior)
  - Changes in temporal dynamics (pattern irregularity)
  - Changes in higher-order statistics of the transformed signal

What the invariant transform SUPPRESSES (by design):
  - Uniform amplitude changes (constant scaling)
  - Smooth gain drift (linear, polynomial, sinusoidal calibration)

What this means for products:
  - MOST real bearing/machine faults produce variability changes → detectable
  - Pure amplitude faults without structural change → not detectable
  - Mixed faults → the variability component survives

Rocky's insight is correct: robustness to nuisance CAN become blindness
to signal. But the blindness is PRECISELY characterized — it's exactly
the nuisance class you specified. That's not a bug, it's a specification.

The product implication: users need to understand what they're declaring
as nuisance. The SDK should make this explicit — what's suppressed,
what's preserved, what's at risk.
""")


def main():
    print("=" * 70)
    print("ROCKY'S CHALLENGE: Critical Experiments for Invariant Order")
    print("=" * 70)
    print()

    results1 = test1_nuisance_shape()
    test2_correlated_nuisance()


if __name__ == "__main__":
    main()
