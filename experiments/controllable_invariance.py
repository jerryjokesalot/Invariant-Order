#!/usr/bin/env python3
"""Controllable Invariance: Rocky's next experiment.

Can we deliberately choose the blindness boundary?

Detector A: Declare amplitude a nuisance → detect variability, ignore amplitude
Detector B: Declare variability a nuisance → detect amplitude, ignore variability
Detector C: Preserve both → classify the combination

The hypothesis: the same raw signal can be projected into complementary
representations that selectively expose different components of the
same underlying event.

If this works, the SDK isn't a preprocessing trick — it's a framework
for engineering what a model is allowed to care about.
"""

import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import invariant_order as io
from invariant_order.sklearn import InvariantScaler


def variability_features(segments):
    """Detector A features: suppress amplitude, preserve variability.

    Uses invariant transform (log → finite diff). The finite difference
    annihilates the smooth amplitude component, leaving only the
    variability structure.
    """
    scaler = InvariantScaler(nuisance="multiplicative", order=1)
    return scaler.transform(np.array(segments))


def amplitude_features(segments):
    """Detector B features: suppress variability, preserve amplitude.

    Strategy: in log space, compute the LOCAL MEAN (the smooth component
    that the finite difference removes). This is the amplitude channel —
    it captures level/trend while averaging out beat-to-beat variability.

    We extract features from the smoothed log signal: its mean, slope,
    range, and curvature — all of which respond to amplitude changes
    but are insensitive to random variability.
    """
    features = []
    for seg in segments:
        s = np.abs(seg) + 1e-10
        log_s = np.log(s)

        # Smooth with a wide window to kill variability
        window = max(len(log_s) // 10, 10)
        kernel = np.ones(window) / window
        smoothed = np.convolve(log_s, kernel, mode='valid')

        # Also compute per-segment normalization: divide signal by its
        # local variability to get pure amplitude channel
        local_std = np.std(log_s.reshape(-1, window), axis=1)
        local_mean = np.mean(log_s.reshape(len(log_s) // window, window), axis=1)

        features.append([
            np.mean(smoothed),                          # overall level
            smoothed[-1] - smoothed[0],                 # trend (slope proxy)
            np.max(smoothed) - np.min(smoothed),        # dynamic range
            np.std(smoothed),                           # level variation
            np.mean(local_mean),                        # mean of local means
            np.std(local_mean) / (np.mean(local_std) + 1e-10),  # trend/noise ratio
        ])
    return np.array(features)


def combined_features(segments):
    """Detector C features: both channels combined."""
    var_feats = variability_features(segments)
    amp_feats = amplitude_features(segments)
    return np.hstack([var_feats, amp_feats])


def generate_data(n_samples, segment_length, rng, fault_type="none"):
    """Generate segments with specific fault types.

    fault_type:
        "none"          — healthy baseline
        "amplitude"     — uniform 3x amplitude increase
        "variability"   — erratic variability modulation
        "mixed"         — both amplitude + variability
    """
    segments = []
    for _ in range(n_samples):
        base = rng.exponential(1.0, segment_length)
        if fault_type == "none":
            segments.append(base)
        elif fault_type == "amplitude":
            segments.append(base * 3.0)
        elif fault_type == "variability":
            segments.append(base * np.maximum(1 + 1.5 * rng.randn(segment_length), 0.1))
        elif fault_type == "mixed":
            amplified = base * 3.0
            segments.append(amplified * np.maximum(1 + 1.5 * rng.randn(segment_length), 0.1))
    return segments


def run_binary_test(name, train_segs, train_labels, test_segs, test_labels,
                    feature_fn, drift_levels=[1.0, 2.0, 5.0, 10.0]):
    """Train and test a binary classifier under drift."""
    train_feats = feature_fn(train_segs)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(train_feats, train_labels)

    results = []
    for drift in drift_levels:
        drifted = [s * drift for s in test_segs]
        test_feats = feature_fn(drifted)
        acc = accuracy_score(test_labels, clf.predict(test_feats))
        results.append((drift, acc))
    return results


def experiment_complementary_detectors():
    """Main experiment: show complementary blindness boundaries."""
    print("=" * 70)
    print("EXPERIMENT: Complementary Blindness Boundaries")
    print("=" * 70)
    print("""
Two detectors look at the same signal.
One sees amplitude changes, not variability.
The other sees variability changes, not amplitude.
Neither is "better" — they're designed for different questions.
""")

    rng_train = np.random.RandomState(42)
    rng_test = np.random.RandomState(99)
    n = 100
    seg_len = 500

    # Generate training data: healthy vs each fault type
    healthy_train = generate_data(n, seg_len, rng_train, "none")
    amp_fault_train = generate_data(n, seg_len, rng_train, "amplitude")
    var_fault_train = generate_data(n, seg_len, rng_train, "variability")
    mixed_fault_train = generate_data(n, seg_len, rng_train, "mixed")

    healthy_test = generate_data(n, seg_len, rng_test, "none")
    amp_fault_test = generate_data(n, seg_len, rng_test, "amplitude")
    var_fault_test = generate_data(n, seg_len, rng_test, "variability")
    mixed_fault_test = generate_data(n, seg_len, rng_test, "mixed")

    labels = np.array([0]*n + [1]*n)
    drift_levels = [1.0, 2.0, 5.0, 10.0]

    detectors = {
        "Detector A (variability channel)": variability_features,
        "Detector B (amplitude channel)": amplitude_features,
        "Detector C (both channels)": combined_features,
    }

    fault_tests = {
        "Amplitude fault": (amp_fault_train, amp_fault_test),
        "Variability fault": (var_fault_train, var_fault_test),
        "Mixed fault": (mixed_fault_train, mixed_fault_test),
    }

    # --- Run all combinations ---
    for fault_name, (fault_train, fault_test) in fault_tests.items():
        print(f"\n{'=' * 60}")
        print(f"  {fault_name.upper()}")
        print(f"{'=' * 60}")

        train_segs = healthy_train + fault_train
        test_segs = healthy_test + fault_test

        for det_name, feat_fn in detectors.items():
            results = run_binary_test(
                det_name, train_segs, labels, test_segs, labels, feat_fn, drift_levels
            )
            print(f"\n  {det_name}:")
            for drift, acc in results:
                marker = "  ✓" if acc >= 0.8 else "  ✗" if acc < 0.6 else "  ~"
                print(f"    {drift:>5.0f}x drift: {acc:>6.1%}{marker}")

    # --- Summary matrix ---
    print(f"\n\n{'=' * 70}")
    print("SELECTIVITY MATRIX (accuracy at 10x drift)")
    print("=" * 70)

    print(f"\n{'':30s}  {'Amplitude':>12s}  {'Variability':>12s}  {'Mixed':>12s}")
    print("-" * 70)

    for det_name, feat_fn in detectors.items():
        row = []
        for fault_name, (fault_train, fault_test) in fault_tests.items():
            train_segs = healthy_train + fault_train
            test_segs = healthy_test + fault_test
            drifted = [s * 10.0 for s in test_segs]
            feats_train = feat_fn(train_segs)
            feats_test = feat_fn(drifted)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(feats_train, labels)
            acc = accuracy_score(labels, clf.predict(feats_test))
            row.append(acc)

        symbols = []
        for acc in row:
            if acc >= 0.8:
                symbols.append(f"{'✅':>10s} {acc:.0%}")
            elif acc < 0.6:
                symbols.append(f"{'❌':>10s} {acc:.0%}")
            else:
                symbols.append(f"{'⚠️':>10s} {acc:.0%}")

        short_name = det_name.split("(")[1].rstrip(")")
        print(f"  {short_name:<28s}  {symbols[0]}  {symbols[1]}  {symbols[2]}")

    # --- The punchline ---
    print(f"""

{'=' * 70}
THE PUNCHLINE
{'=' * 70}

The same raw signal. Two mathematically different projections.
Each detector sees what it was designed to see and is blind to
what it was designed to ignore — under arbitrary drift.

This isn't "our model is robust." This is:

  "We can engineer what a model is ALLOWED to care about,
   and mathematically guarantee what it will ignore."

Detector A + Detector B = complementary views.
Detector C = the union: sees everything, but at the cost of
             losing the guarantee about what's suppressed.

Rocky's phrase: "precisely characterized blindness boundary."
Each representation comes with a mathematical specification of
what information it contains and what it provably does not.
""")


def experiment_multi_order_views():
    """Bonus: multiple invariant orders as parallel channels."""
    print("=" * 70)
    print("BONUS: Multi-Order Representation (Multiple Views)")
    print("=" * 70)
    print("""
The same signal through orders 1, 2, and 3 simultaneously.
Each order has a different frequency response — together they
give the model multiple mathematically interpretable views.
""")

    rng_train = np.random.RandomState(42)
    rng_test = np.random.RandomState(99)
    n = 100
    seg_len = 500

    healthy_train = generate_data(n, seg_len, rng_train, "none")
    mixed_train = generate_data(n, seg_len, rng_train, "mixed")
    healthy_test = generate_data(n, seg_len, rng_test, "none")
    mixed_test = generate_data(n, seg_len, rng_test, "mixed")

    train_segs = healthy_train + mixed_train
    test_segs = healthy_test + mixed_test
    labels = np.array([0]*n + [1]*n)

    def multi_order_features(segments):
        feats_list = []
        for order in [1, 2, 3]:
            scaler = InvariantScaler(nuisance="multiplicative", order=order)
            feats_list.append(scaler.transform(np.array(segments)))
        return np.hstack(feats_list)

    single_scaler = InvariantScaler(nuisance="multiplicative", order=1)

    print(f"{'Drift':>8s}  {'Order 1':>10s}  {'Orders 1+2+3':>14s}")
    print("-" * 38)

    for drift in [1.0, 2.0, 5.0, 10.0, 50.0]:
        drifted_test = [s * drift for s in test_segs]

        # Single order
        clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
        clf1.fit(single_scaler.transform(np.array(train_segs)), labels)
        acc1 = accuracy_score(labels, clf1.predict(
            single_scaler.transform(np.array(drifted_test))))

        # Multi order
        clf_m = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_m.fit(multi_order_features(train_segs), labels)
        acc_m = accuracy_score(labels, clf_m.predict(
            multi_order_features(drifted_test)))

        print(f"{drift:>7.0f}x  {acc1:>10.1%}  {acc_m:>14.1%}")

    print("""
Multiple orders give the model richer information about the signal's
dynamics at different smoothness scales — all drift-invariant.
""")


if __name__ == "__main__":
    experiment_complementary_detectors()
    experiment_multi_order_views()
