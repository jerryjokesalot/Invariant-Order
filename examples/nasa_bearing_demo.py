#!/usr/bin/env python3
"""NASA IMS Bearing Demo: detect real bearing failure through sensor drift.

Uses Test Set 2, Bearing 1 (outer race failure) from the NASA IMS dataset.
984 vibration snapshots at 10-minute intervals over 7 days. The bearing
degrades starting around snapshot 530 and fails at snapshot 984.

This demo shows:
1. Invariant Order detects the failure onset while ignoring baseline drift
2. CUSUM on raw RMS produces false alarms from drift
3. Side-by-side comparison

Dataset: https://data.nasa.gov/dataset/ims-bearings
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import invariant_order as io


def load_bearing_data(data_dir):
    """Load Test 2, Bearing 1 RMS values from NASA IMS dataset."""
    test2_dir = os.path.join(data_dir, "2nd_test")
    if not os.path.isdir(test2_dir):
        for name in os.listdir(data_dir):
            candidate = os.path.join(data_dir, name)
            if os.path.isdir(candidate) and "2nd" in name.lower():
                test2_dir = candidate
                break

    files = sorted([f for f in os.listdir(test2_dir)
                    if not f.startswith('.') and os.path.isfile(os.path.join(test2_dir, f))])

    rms_values = []
    for fname in files:
        filepath = os.path.join(test2_dir, fname)
        try:
            data = np.loadtxt(filepath)
            if data.ndim == 1:
                bearing1 = data
            else:
                bearing1 = data[:, 0]
            rms_values.append(np.sqrt(np.mean(bearing1**2)))
        except Exception:
            continue

    return np.array(rms_values), files


def cusum_detector(signal, threshold=5.0):
    """Standard CUSUM on raw signal for comparison."""
    mean = np.mean(signal[:len(signal)//4])
    std = np.std(signal[:len(signal)//4], ddof=1)
    if std < 1e-15:
        return []

    normalized = (signal - mean) / std
    cusum_pos = np.zeros(len(signal))
    alerts = []

    for i in range(1, len(signal)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i] - 0.5)
        if cusum_pos[i] > threshold and (not alerts or i - alerts[-1] > 50):
            alerts.append(i)

    return alerts


def main():
    data_dir = os.environ.get("IMS_DATA_DIR",
                              os.path.join(os.path.dirname(__file__), '..', 'data', 'IMS'))

    if not os.path.isdir(data_dir):
        print(f"Dataset not found at: {data_dir}")
        print(f"Download from: https://data.nasa.gov/dataset/ims-bearings")
        print(f"Extract and set IMS_DATA_DIR, or place at {data_dir}")
        sys.exit(1)

    print("Loading NASA IMS Bearing Dataset (Test 2, Bearing 1)...")
    rms_values, files = load_bearing_data(data_dir)
    n = len(rms_values)
    print(f"Loaded {n} snapshots ({n * 10 / 60:.0f} hours of operation)\n")

    # --- Invariant Order detection ---
    print("=" * 60)
    print("INVARIANT ORDER (multiplicative, order=1)")
    print("=" * 60)

    result = io.scan(rms_values, nuisance="multiplicative", order=1, threshold=2.0)
    print(f"Change points detected: {len(result.change_points)}")
    for cp in result.change_points:
        hours = cp.index * 10 / 60
        print(f"  Snapshot {cp.index} ({hours:.1f}h): score={cp.score:.1f}")

    # Confirm drift blindness: scan early stable region
    stable = rms_values[:400]
    stable_drifted = stable * np.linspace(1, 2, 400)
    result_stable = io.scan(stable_drifted, nuisance="multiplicative", order=1)
    print(f"\nFalse alarms on artificially drifted stable region: "
          f"{len(result_stable.change_points)}")

    # Statistical comparison: stable vs degraded
    split = min(500, n // 2)
    comparison = io.compare(rms_values[:split], rms_values[split:],
                            nuisance="multiplicative", order=1)
    print(f"\nStable vs degraded comparison:")
    print(f"  Effect size: {comparison.effect_size:.2f}")
    print(f"  p-value:     {comparison.p_value:.4f}")

    # --- CUSUM on raw RMS for comparison ---
    print(f"\n{'=' * 60}")
    print("CUSUM ON RAW RMS (traditional approach)")
    print("=" * 60)

    cusum_alerts = cusum_detector(rms_values, threshold=5.0)
    print(f"Alerts: {len(cusum_alerts)}")
    for idx in cusum_alerts[:10]:
        hours = idx * 10 / 60
        print(f"  Snapshot {idx} ({hours:.1f}h)")
    if len(cusum_alerts) > 10:
        print(f"  ... and {len(cusum_alerts) - 10} more")

    # --- Streaming detection ---
    print(f"\n{'=' * 60}")
    print("STREAMING DETECTION (real-time simulation)")
    print("=" * 60)

    stream = io.StreamDetector(
        nuisance="multiplicative", order=1,
        baseline_window=100, threshold=2.0, variance_window=30,
    )
    first_alert = None
    for i, val in enumerate(rms_values):
        alert = stream.push(val)
        if alert and first_alert is None:
            first_alert = (i, alert)

    if first_alert:
        idx, alert = first_alert
        hours = idx * 10 / 60
        print(f"First alert: snapshot {idx} ({hours:.1f}h), score={alert.score:.1f}")
        remaining = (n - idx) * 10 / 60
        print(f"Lead time before failure: {remaining:.1f} hours")
    else:
        print("No alerts triggered")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Total operation time: {n * 10 / 60:.0f} hours")
    if result.change_points:
        first_io = result.change_points[0].index
        print(f"Invariant Order first detection: snapshot {first_io} "
              f"({first_io * 10 / 60:.1f}h)")
    if cusum_alerts:
        print(f"CUSUM first alert: snapshot {cusum_alerts[0]} "
              f"({cusum_alerts[0] * 10 / 60:.1f}h)")
    if first_alert:
        print(f"Streaming first alert: snapshot {first_alert[0]} "
              f"({first_alert[0] * 10 / 60:.1f}h)")


if __name__ == "__main__":
    main()
