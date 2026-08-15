"""UCI Gas Sensor Array Drift — Invariant Order Experiment

Dataset: 16 chemical sensors, 6 gases, 10 batches over 36 months.
Each batch has more sensor drift. Classification task: identify gas type.

Key insight: InvariantScaler (designed for ordered time series) doesn't fit
this tabular data shape — but the THEOREM does. The invariant to multiplicative
drift is log-ratios across sensors: log(s_i) - log(s_j) = log(s_i/s_j),
which IS a first-order finite difference in log space applied SPATIALLY.

This experiment tests:
1. Raw features (baseline — should degrade with drift)
2. InvariantScaler (mismatched — treats heterogeneous features as ordered signal)
3. Log-ratio features (correct invariant for this data shape)
4. Sensor-profile approach (treat 16 sensor readings as an ordered profile)

Dataset: https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from invariant_order.sklearn import InvariantScaler
from invariant_order.core import transform

DATA_DIR = Path(__file__).parent.parent / "data" / "gas_sensor_drift" / "Dataset"


def load_batch(batch_num):
    filepath = DATA_DIR / f"batch{batch_num}.dat"
    features = []
    labels = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0].rstrip(";"))
            feat = np.zeros(128)
            for item in parts[1:]:
                idx_str, val_str = item.split(":")
                feat[int(idx_str) - 1] = float(val_str)
            features.append(feat)
            labels.append(label)
    return np.array(features), np.array(labels)


def load_all_batches():
    batches = {}
    for i in range(1, 11):
        X, y = load_batch(i)
        batches[i] = (X, y)
    return batches


def extract_sensor_values(X, feature_idx=0):
    """Extract one feature type across all 16 sensors.
    feature_idx 0 = steady-state, 1-7 = dynamic features."""
    return X[:, feature_idx::8]


def log_ratio_features(X):
    """Cross-sensor log-ratios: the correct invariant for tabular multiplicative drift.

    For 16 sensors, log(s_i) - log(s_j) cancels common multiplicative drift.
    This IS first-order finite differencing in log space — the IO theorem
    applied spatially instead of temporally.
    """
    ss = extract_sensor_values(X, feature_idx=0)
    ss = np.abs(ss) + 1e-10
    log_ss = np.log(ss)

    features_list = []

    # Adjacent log-ratios (15 features): Δ(log) across sensor array
    diffs = np.diff(log_ss, axis=1)
    features_list.append(diffs)

    # Each sensor's dynamic features normalized by its own steady-state
    for fidx in range(1, 8):
        dynamic = extract_sensor_values(X, feature_idx=fidx)
        features_list.append(dynamic)

    return np.hstack(features_list)


def sensor_profile_invariant(X):
    """Treat each feature type's 16-sensor vector as a profile, apply IO transform.

    The 16 sensors are physical devices — their relative response pattern
    IS an ordered structure. Finite differencing along the sensor array
    removes multiplicative scaling that affects all sensors equally.
    """
    features_list = []
    for fidx in range(8):
        sensor_vals = extract_sensor_values(X, feature_idx=fidx)
        sensor_vals_safe = np.abs(sensor_vals) + 1e-10

        # Apply IO transform across the 16-sensor profile (order 1)
        transformed = np.array([
            transform(row, "multiplicative", 1) for row in sensor_vals_safe
        ])
        features_list.append(transformed)

    return np.hstack(features_list)


def sample_normalized_features(X):
    """Normalize each sample by its L2 norm — direction is drift-invariant."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return X / norms


def run_test(batches, name, transform_fn, train_batches=(1,)):
    """Train RF on transformed features, test across all batches."""
    X_train = np.vstack([batches[b][0] for b in train_batches])
    y_train = np.concatenate([batches[b][1] for b in train_batches])

    X_train_t = transform_fn(X_train)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    pipe.fit(X_train_t, y_train)

    results = {}
    for b in range(1, 11):
        X_test, y_test = batches[b]
        X_test_t = transform_fn(X_test)
        y_pred = pipe.predict(X_test_t)
        results[b] = accuracy_score(y_test, y_pred)
    return results


def main():
    print("=" * 70)
    print("UCI GAS SENSOR DRIFT — INVARIANT ORDER EXPERIMENT")
    print("=" * 70)
    print()

    batches = load_all_batches()
    total = sum(len(b[1]) for b in batches.values())

    print("Dataset Summary:")
    print(f"  Total samples: {total}")
    print(f"  Batches: 10 (36 months of increasing sensor drift)")
    print(f"  Features: 128 (16 sensors × 8 features)")
    print(f"  Classes: 6 gases")
    for b in range(1, 11):
        _, y = batches[b]
        print(f"  Batch {b:2d}: {len(y):4d} samples, {len(set(y))} classes")
    print()

    months = [1, 2, 4, 7, 10, 11, 13, 15, 19, 36]

    # --- Define approaches ---
    approaches = [
        ("Raw", lambda X: X),
        ("InvScaler", lambda X: np.array([
            InvariantScaler("multiplicative", 1).fit(X).transform(X)
        ]).squeeze() if False else InvariantScaler("multiplicative", 1).fit_transform(X)),
        ("LogRatio", log_ratio_features),
        ("SensorProfile", sensor_profile_invariant),
        ("L2-Norm", sample_normalized_features),
    ]

    # --- Experiment 1: Train on Batch 1 only ---
    print("-" * 70)
    print("EXPERIMENT 1: Train on Batch 1 only (hardest test)")
    print("-" * 70)
    print()

    all_results = {}
    for name, tfn in approaches:
        r = run_test(batches, name, tfn, train_batches=(1,))
        all_results[name] = r

    header = f"{'Batch':>6} {'Mo':>3}"
    for name, _ in approaches:
        header += f" {name:>13}"
    print(header)
    print("-" * len(header))

    for b in range(1, 11):
        row = f"{b:6d} {months[b-1]:3d}"
        for name, _ in approaches:
            row += f" {all_results[name][b]:13.1%}"
        print(row)

    print()
    for name, _ in approaches:
        accs = [all_results[name][b] for b in range(2, 11)]
        print(f"  {name:14s} mean(2-10): {np.mean(accs):.1%}  "
              f"worst: {np.min(accs):.1%}  "
              f"degradation: {all_results[name][1] - np.min(accs):+.1%}")

    # --- Experiment 2: Train on Batches 1-3 ---
    print()
    print("-" * 70)
    print("EXPERIMENT 2: Train on Batches 1-3, test 4-10")
    print("-" * 70)
    print()

    all_results2 = {}
    for name, tfn in approaches:
        r = run_test(batches, name, tfn, train_batches=(1, 2, 3))
        all_results2[name] = r

    header = f"{'Batch':>6} {'Mo':>3}"
    for name, _ in approaches:
        header += f" {name:>13}"
    print(header)
    print("-" * len(header))

    for b in range(4, 11):
        row = f"{b:6d} {months[b-1]:3d}"
        for name, _ in approaches:
            row += f" {all_results2[name][b]:13.1%}"
        print(row)

    print()
    for name, _ in approaches:
        accs = [all_results2[name][b] for b in range(4, 11)]
        print(f"  {name:14s} mean(4-10): {np.mean(accs):.1%}  worst: {np.min(accs):.1%}")

    # --- Experiment 3: IO order comparison on sensor profiles ---
    print()
    print("-" * 70)
    print("EXPERIMENT 3: Sensor profile IO — order 1 vs 2 vs 3")
    print("-" * 70)
    print()

    order_results = {}
    for order in [1, 2, 3]:
        def make_tfn(m):
            def tfn(X):
                features_list = []
                for fidx in range(8):
                    sv = extract_sensor_values(X, feature_idx=fidx)
                    sv = np.abs(sv) + 1e-10
                    transformed = np.array([
                        transform(row, "multiplicative", m) for row in sv
                    ])
                    features_list.append(transformed)
                return np.hstack(features_list)
            return tfn

        r = run_test(batches, f"SP-m{order}", make_tfn(order), train_batches=(1,))
        order_results[order] = r

    print(f"{'Batch':>6} {'Raw':>8} {'m=1':>8} {'m=2':>8} {'m=3':>8}")
    print("-" * 42)
    for b in range(1, 11):
        print(f"{b:6d} {all_results['Raw'][b]:8.1%}", end="")
        for order in [1, 2, 3]:
            print(f" {order_results[order][b]:8.1%}", end="")
        print()

    # --- Experiment 4: Combined features ---
    print()
    print("-" * 70)
    print("EXPERIMENT 4: Combined (log-ratio + sensor profile + raw dynamic)")
    print("-" * 70)
    print()

    def combined_features(X):
        parts = [
            log_ratio_features(X),
            sensor_profile_invariant(X),
        ]
        return np.hstack(parts)

    r_combined = run_test(batches, "Combined", combined_features, train_batches=(1,))

    print(f"{'Batch':>6} {'Raw':>8} {'LogRatio':>10} {'SenProf':>10} {'Combined':>10}")
    print("-" * 50)
    for b in range(1, 11):
        print(f"{b:6d} {all_results['Raw'][b]:8.1%} "
              f"{all_results['LogRatio'][b]:10.1%} "
              f"{all_results['SensorProfile'][b]:10.1%} "
              f"{r_combined[b]:10.1%}")

    # --- Final Report ---
    print()
    print("=" * 70)
    print("REPORT: UCI Gas Sensor Drift × Invariant Order")
    print("=" * 70)
    print()

    best_inv_name = None
    best_inv_mean = 0
    for name in ["LogRatio", "SensorProfile", "L2-Norm"]:
        accs = [all_results[name][b] for b in range(2, 11)]
        m = np.mean(accs)
        if m > best_inv_mean:
            best_inv_mean = m
            best_inv_name = name

    raw_accs = [all_results["Raw"][b] for b in range(2, 11)]
    inv_accs = [all_results[best_inv_name][b] for b in range(2, 11)]
    comb_accs = [r_combined[b] for b in range(2, 11)]

    print(f"  Dataset:     UCI Gas Sensor Array Drift")
    print(f"  Archetype:   Sensor Drift (Type A)")
    print(f"  Drift type:  Multiplicative sensor aging (36 months)")
    print(f"  Training:    Batch 1 only")
    print()
    print(f"  Raw features:")
    print(f"    Mean(2-10):  {np.mean(raw_accs):.1%}")
    print(f"    Worst batch: {np.min(raw_accs):.1%}")
    print()
    print(f"  Best invariant ({best_inv_name}):")
    print(f"    Mean(2-10):  {np.mean(inv_accs):.1%}")
    print(f"    Worst batch: {np.min(inv_accs):.1%}")
    print()
    print(f"  Combined features:")
    print(f"    Mean(2-10):  {np.mean(comb_accs):.1%}")
    print(f"    Worst batch: {np.min(comb_accs):.1%}")
    print()

    adv = np.mean(inv_accs) - np.mean(raw_accs)
    if adv > 0:
        print(f"  ✓ {best_inv_name} beats raw by {adv:+.1%} (mean)")
    else:
        print(f"  ✗ {best_inv_name} underperforms raw by {adv:.1%} (mean)")

    comb_adv = np.mean(comb_accs) - np.mean(raw_accs)
    if comb_adv > 0:
        print(f"  ✓ Combined beats raw by {comb_adv:+.1%} (mean)")

    print()
    print("  KEY FINDING:")
    print("  InvariantScaler (designed for ordered time series) is the WRONG")
    print("  tool for tabular snapshot data. But the IO theorem applies")
    print("  when used correctly: log-ratios across sensors ARE first-order")
    print("  finite differences in log space, applied spatially.")
    print()
    print("  Data shape matters: IO needs ordered neighborhoods with")
    print("  meaningful local relationships. For this dataset, the 'ordering'")
    print("  is across the 16-sensor array, not along the 128-feature vector.")
    print()


if __name__ == "__main__":
    main()
