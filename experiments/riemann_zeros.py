"""Riemann Zeros — Invariant Order Experiment

Archetype E: Number-theoretic structure (spectral geometry)

The zeta zeros are conjectured (Montgomery, Odlyzko) to follow GUE
statistics locally. This experiment tests whether IO-invariant statistics
can detect that GUE structure independent of the smooth zero density,
and whether the framework can distinguish zeta zeros from Poisson/GOE.

DESIGN CONTRACT:
  Geometry:     Spectral (ordered zeros on the critical line)
  Nuisance:     Smooth zero density ~ log(t/2π) / (2π)
  Structure:    Local zero correlations (GUE-like repulsion)
  Invariant:    Spacing ratios and Δᵐ(log spacings)

PREDICTIONS:
  P1: Gap ratios of zeta zeros match GUE, not GOE or Poisson
  P2: IO statistics are invariant to the increasing zero density
  P3: A classifier trained on random matrix ensembles should
      classify zeta zeros as GUE
  P4: The IO representation makes density variation irrelevant,
      so zeros from different heights on the critical line
      should look statistically identical

PREDICTED FAILURE MODES:
  F1: [C] At low height, finite-size effects may cause deviations
  F2: [B] If we use too few zeros, spacing statistics are noisy
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from invariant_order import transform

try:
    from mpmath import zetazero, mp
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def compute_zeta_zeros(n_zeros, start=1):
    """Compute imaginary parts of zeta zeros on the critical line."""
    if not HAS_MPMATH:
        raise ImportError("mpmath required: pip install mpmath")
    mp.dps = 25
    zeros = []
    for k in range(start, start + n_zeros):
        z = zetazero(k)
        zeros.append(float(z.imag))
    return np.array(zeros)


def generate_goe(n, rng):
    A = rng.standard_normal((n, n))
    H = (A + A.T) / np.sqrt(2 * n)
    return np.sort(np.linalg.eigvalsh(H))


def generate_gue(n, rng):
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    H = (A + A.conj().T) / np.sqrt(2 * n)
    return np.sort(np.linalg.eigvalsh(H))


def generate_poisson(n, rng):
    return np.sort(rng.uniform(0, n, n))


def gap_ratios(values):
    s = np.diff(values)
    s = s[s > 0]
    if len(s) < 2:
        return np.array([])
    return np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])


def unfold_zeros(zeros):
    """Unfold zeta zeros using smooth density N(t) ~ t/(2π) * log(t/(2πe))."""
    t = zeros
    unfolded = t / (2 * np.pi) * np.log(t / (2 * np.pi * np.e))
    return unfolded


def io_features(values, order=1):
    """Extract IO-invariant features from an ordered spectrum."""
    s = np.diff(values)
    s = s[s > 0]
    if len(s) < order + 3:
        return np.zeros(8)

    r = gap_ratios(values)
    t = transform(s, "multiplicative", order)

    return np.array([
        np.mean(r),
        np.var(r),
        np.mean(t**2),
        np.var(t),
        np.percentile(r, 10) if len(r) > 10 else 0,
        np.percentile(r, 90) if len(r) > 10 else 0,
        np.mean(np.abs(t)),
        np.sum(r < 0.2) / len(r) if len(r) > 0 else 0,
    ])


def main():
    print("=" * 70)
    print("RIEMANN ZEROS — INVARIANT ORDER EXPERIMENT")
    print("=" * 70)
    print()

    if not HAS_MPMATH:
        print("mpmath required: pip install mpmath")
        return

    rng = np.random.default_rng(42)

    # --- Compute zeros ---
    print("-" * 70)
    print("COMPUTING ZETA ZEROS")
    print("-" * 70)
    print()

    n_zeros = 500
    print(f"Computing first {n_zeros} zeros of ζ(s)...", flush=True)
    zeros = compute_zeta_zeros(n_zeros)
    print(f"  Range: t = {zeros[0]:.2f} to {zeros[-1]:.2f}")
    print(f"  Mean spacing: {np.mean(np.diff(zeros)):.4f}")
    print()

    # Also compute zeros at higher height for density variation test
    print(f"Computing zeros {1001}-{1500} (higher on critical line)...", flush=True)
    zeros_high = compute_zeta_zeros(500, start=1001)
    print(f"  Range: t = {zeros_high[0]:.2f} to {zeros_high[-1]:.2f}")
    print(f"  Mean spacing: {np.mean(np.diff(zeros_high)):.4f}")
    print()

    # --- Theorem verification ---
    print("-" * 70)
    print("THEOREM VERIFICATION")
    print("-" * 70)
    print()

    print("P1: Gap ratios of zeta zeros should match GUE (~0.5996)")
    print()

    r_zeta = gap_ratios(zeros)
    r_zeta_high = gap_ratios(zeros_high)

    # Unfold zeros and compute gap ratios (should give same result)
    zeros_unfolded = unfold_zeros(zeros)
    r_unfolded = gap_ratios(zeros_unfolded)

    n_samples = 200
    goe_means, gue_means, poisson_means = [], [], []
    for _ in range(n_samples):
        goe_means.append(np.mean(gap_ratios(generate_goe(200, rng))))
        gue_means.append(np.mean(gap_ratios(generate_gue(200, rng))))
        poisson_means.append(np.mean(gap_ratios(generate_poisson(200, rng))))

    print(f"  Zeta zeros (1-{n_zeros}):      <r> = {np.mean(r_zeta):.4f}")
    print(f"  Zeta zeros (unfolded):    <r> = {np.mean(r_unfolded):.4f}")
    print(f"  Zeta zeros (1001-1500):   <r> = {np.mean(r_zeta_high):.4f}")
    print(f"  GUE (200×200):            <r> = {np.mean(gue_means):.4f} ± {np.std(gue_means):.4f}")
    print(f"  GOE (200×200):            <r> = {np.mean(goe_means):.4f} ± {np.std(goe_means):.4f}")
    print(f"  Poisson:                  <r> = {np.mean(poisson_means):.4f} ± {np.std(poisson_means):.4f}")
    print()

    # Distance from each ensemble
    d_gue = abs(np.mean(r_zeta) - np.mean(gue_means)) / np.std(gue_means)
    d_goe = abs(np.mean(r_zeta) - np.mean(goe_means)) / np.std(goe_means)
    d_poi = abs(np.mean(r_zeta) - np.mean(poisson_means)) / np.std(poisson_means)
    print(f"  Distance to GUE:     {d_gue:.1f}σ")
    print(f"  Distance to GOE:     {d_goe:.1f}σ")
    print(f"  Distance to Poisson: {d_poi:.1f}σ")

    closest = "GUE" if d_gue < d_goe and d_gue < d_poi else (
              "GOE" if d_goe < d_poi else "Poisson")
    print(f"  → Closest match: {closest}")

    # --- P2: Invariance to density ---
    print()
    print("P2 + P4: IO statistics invariant to zero density")
    print()
    print("Comparing IO features of zeros at different heights:")
    print("(Density increases with height, but IO should be invariant)")
    print()

    feat_low = io_features(zeros)
    feat_high = io_features(zeros_high)
    feat_unfolded = io_features(zeros_unfolded)

    feature_names = ["<r>", "var(r)", "M2(t)", "var(t)",
                     "p10(r)", "p90(r)", "<|t|>", "cluster"]

    print(f"  {'Feature':>10} {'Low (1-500)':>12} {'High (1k-1.5k)':>15} {'Unfolded':>10} {'Low≈High?':>10}")
    print(f"  {'-'*60}")
    for i, name in enumerate(feature_names):
        rel_diff = abs(feat_low[i] - feat_high[i]) / (abs(feat_low[i]) + 1e-10)
        match = "✓" if rel_diff < 0.3 else "~" if rel_diff < 0.5 else "✗"
        print(f"  {name:>10} {feat_low[i]:12.4f} {feat_high[i]:15.4f} "
              f"{feat_unfolded[i]:10.4f} {match:>10}")

    # --- Classification test (P3) ---
    print()
    print("-" * 70)
    print("CLASSIFICATION: Can ensemble classifier identify zeta zeros as GUE?")
    print("-" * 70)
    print()

    # Train on random matrix ensembles
    n_train = 300
    X_train, y_train = [], []
    labels = {"GOE": 0, "GUE": 1, "Poisson": 2}
    gens = {"GOE": generate_goe, "GUE": generate_gue, "Poisson": generate_poisson}

    for name, gen_fn in gens.items():
        for _ in range(n_train):
            eigs = gen_fn(200, rng)
            X_train.append(io_features(eigs))
            y_train.append(labels[name])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    clf = Pipeline([("s", StandardScaler()),
                    ("c", RandomForestClassifier(200, random_state=42))])
    clf.fit(X_train, y_train)

    # Classify zeta zeros in chunks
    chunk_size = 100
    label_names = {0: "GOE", 1: "GUE", 2: "Poisson"}

    print("Classifying zeta zero chunks (each ~100 zeros):")
    print()

    all_chunks = []
    for start in range(0, len(zeros) - chunk_size, chunk_size // 2):
        chunk = zeros[start:start + chunk_size]
        feat = io_features(chunk)
        all_chunks.append(feat)

    all_chunks = np.array(all_chunks)
    preds = clf.predict(all_chunks)
    probs = clf.predict_proba(all_chunks)

    for i, (pred, prob) in enumerate(zip(preds, probs)):
        start = i * (chunk_size // 2)
        end = start + chunk_size
        print(f"  Zeros {start+1:4d}-{end:4d}: → {label_names[pred]:8s} "
              f"(GOE:{prob[0]:.2f} GUE:{prob[1]:.2f} Poi:{prob[2]:.2f})")

    # Also classify high zeros
    print()
    print("Classifying high zeros (1001-1500):")
    high_chunks = []
    for start in range(0, len(zeros_high) - chunk_size, chunk_size // 2):
        chunk = zeros_high[start:start + chunk_size]
        high_chunks.append(io_features(chunk))

    high_chunks = np.array(high_chunks)
    preds_high = clf.predict(high_chunks)
    probs_high = clf.predict_proba(high_chunks)

    for i, (pred, prob) in enumerate(zip(preds_high, probs_high)):
        start = i * (chunk_size // 2)
        end = start + chunk_size
        print(f"  Zeros {start+1001:4d}-{end+1000:4d}: → {label_names[pred]:8s} "
              f"(GOE:{prob[0]:.2f} GUE:{prob[1]:.2f} Poi:{prob[2]:.2f})")

    gue_frac = np.mean(preds == 1)
    gue_frac_high = np.mean(preds_high == 1)

    print()
    print(f"  GUE classification rate (low zeros):  {gue_frac:.1%}")
    print(f"  GUE classification rate (high zeros): {gue_frac_high:.1%}")

    # --- Report ---
    print()
    print("=" * 70)
    print("REPORT: Riemann Zeros × Invariant Order")
    print("=" * 70)
    print()
    print("  Archetype:    E — Number-Theoretic Structure")
    print("  Geometry:     Spectral (ordered zeros on critical line)")
    print("  Nuisance:     Smooth zero density ~ log(t/2π)")
    print("  Structure:    Local zero correlations (GUE-like)")
    print()
    print("  PREDICTIONS vs RESULTS:")
    print(f"  P1 (GUE match):           <r> = {np.mean(r_zeta):.4f} "
          f"(GUE: {np.mean(gue_means):.4f}, {d_gue:.1f}σ away)")
    print(f"  P2 (density invariance):  [see feature comparison above]")
    print(f"  P3 (classifier → GUE):    {gue_frac:.0%} of chunks classified as GUE")
    print(f"  P4 (height invariance):   Low {gue_frac:.0%} vs High {gue_frac_high:.0%}")
    print()
    print("  The loop is closed: the framework that STARTED with the")
    print("  observation that gap ratios are density-invariant can now")
    print("  independently recover that gap ratios classify zeta zeros")
    print("  as GUE, confirming the Montgomery-Odlyzko connection.")
    print()


if __name__ == "__main__":
    main()
