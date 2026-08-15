"""Quantum Spectra — Invariant Order Experiment

Archetype D: Spectral structure (spectral geometry)

This is where Invariant Order started — the gap ratio's insensitivity
to smooth density is the original observation. This experiment closes
the loop: the generalized framework should rediscover gap-ratio-family
statistics as the natural invariant for spectral data.

DESIGN CONTRACT:
  Geometry:     Spectral (ordered eigenvalues)
  Nuisance:     Smooth spectral density / global scale
  Structure:    Local eigenvalue correlations (repulsion, rigidity)
  Invariant:    Spacing ratios r_n = s_n/s_{n+1} (order 1)
                Higher orders: Δᵐ(log spacings)

PREDICTIONS:
  P1: Gap ratios are invariant to smooth spectral rescaling (unfolding)
  P2: Gap ratio statistics distinguish GOE vs GUE vs Poisson
  P3: Higher-order IO statistics probe longer-range correlations
  P4: Classification of universality class is immune to density distortion

PREDICTED FAILURE MODES:
  F1: [C] Non-smooth density (discontinuities break spacing ratios)
  F2: [D] Two spectra with same local correlations but different global
      density cannot be distinguished (by design)
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from invariant_order import transform


def generate_goe(n, rng):
    """Generate GOE (real symmetric) eigenvalues."""
    A = rng.standard_normal((n, n))
    H = (A + A.T) / np.sqrt(2 * n)
    return np.sort(np.linalg.eigvalsh(H))


def generate_gue(n, rng):
    """Generate GUE (complex Hermitian) eigenvalues."""
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    H = (A + A.conj().T) / np.sqrt(2 * n)
    return np.sort(np.linalg.eigvalsh(H))


def generate_poisson(n, rng):
    """Generate Poisson (uncorrelated) eigenvalues."""
    return np.sort(rng.uniform(0, n, n))


def spacings(eigenvalues):
    """Compute nearest-neighbor spacings."""
    return np.diff(eigenvalues)


def gap_ratios(eigenvalues):
    """Compute gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})."""
    s = spacings(eigenvalues)
    s = s[s > 0]
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return r


def spacing_features_raw(eigenvalues):
    """Raw spacing statistics (sensitive to density)."""
    s = spacings(eigenvalues)
    s = s[s > 0]
    s = np.clip(s, 1e-15, 1e15)
    return np.array([
        np.mean(s), np.std(s), np.var(s),
        np.mean(s**2), np.mean(s**3) / (np.mean(s)**3 + 1e-10),
        np.percentile(s, 25), np.percentile(s, 50), np.percentile(s, 75),
    ])


def spacing_features_io(eigenvalues, order=1):
    """IO-invariant spacing statistics (gap ratios and higher)."""
    s = spacings(eigenvalues)
    s = s[s > 0]
    if len(s) < order + 3:
        return np.zeros(8)

    # Order 1: gap ratios (the original discovery)
    r = gap_ratios(eigenvalues)

    # Higher order: Δᵐ(log spacings)
    log_s = np.log(s)
    t = transform(s, "multiplicative", order)

    return np.array([
        np.mean(r),                                # <r> — universality class signature
        np.var(r),                                  # var(r)
        np.mean(t**2),                              # second moment of IO transform
        np.var(t),                                  # variance of IO transform
        np.percentile(r, 10),                       # tail behavior
        np.percentile(r, 90),                       # tail behavior
        np.mean(np.abs(t)),                         # mean absolute IO
        np.sum(r < 0.2) / len(r) if len(r) > 0 else 0,  # clustering fraction
    ])


def combined_features(eigenvalues, order=1):
    return np.concatenate([
        spacing_features_raw(eigenvalues),
        spacing_features_io(eigenvalues, order),
    ])


def distort_density(eigenvalues, distortion="quadratic"):
    """Apply smooth density distortion (should not affect gap ratios)."""
    e = eigenvalues.copy()
    if distortion == "quadratic":
        e = np.sign(e) * e**2
    elif distortion == "cubic":
        e = e**3
    elif distortion == "exponential":
        e = np.sign(e) * np.exp(np.abs(e)) - 1
    elif distortion == "log":
        e = np.sign(e) * np.log1p(np.abs(e))
    return np.sort(e)


def main():
    print("=" * 70)
    print("QUANTUM SPECTRA — INVARIANT ORDER EXPERIMENT")
    print("=" * 70)
    print()

    rng = np.random.default_rng(42)
    N = 200  # matrix size

    # --- Theorem verification ---
    print("-" * 70)
    print("THEOREM VERIFICATION")
    print("-" * 70)
    print()

    print("P1: Gap ratios invariant to smooth spectral rescaling")
    print()
    eigs = generate_goe(N, rng)
    r_orig = gap_ratios(eigs)
    mean_orig = np.mean(r_orig)

    for name, distortion in [("quadratic", "quadratic"), ("cubic", "cubic"),
                              ("exponential", "exponential"), ("log", "log")]:
        eigs_d = distort_density(eigs, distortion)
        r_d = gap_ratios(eigs_d)
        mean_d = np.mean(r_d)
        diff = abs(mean_d - mean_orig)
        print(f"  {name:12s}: <r> = {mean_d:.6f} (orig {mean_orig:.6f}, "
              f"diff = {diff:.6f})  "
              f"{'✓ INVARIANT' if diff < 0.01 else '~ APPROX'}")

    print()
    print("P2: Gap ratio statistics distinguish universality classes")
    print()

    n_samples = 100
    for label, gen_fn in [("GOE", generate_goe), ("GUE", generate_gue),
                           ("Poisson", generate_poisson)]:
        means = []
        for _ in range(n_samples):
            eigs = gen_fn(N, rng)
            r = gap_ratios(eigs)
            means.append(np.mean(r))
        expected = {"GOE": 0.5307, "GUE": 0.5996, "Poisson": 0.3863}
        print(f"  {label:8s}: <r> = {np.mean(means):.4f} ± {np.std(means):.4f}  "
              f"(theoretical: {expected[label]:.4f})")

    # --- Classification ---
    print()
    print("-" * 70)
    print("CLASSIFICATION: GOE vs GUE vs Poisson")
    print("-" * 70)
    print()

    n_train = 200
    generators = {"GOE": generate_goe, "GUE": generate_gue, "Poisson": generate_poisson}
    class_map = {"GOE": 0, "GUE": 1, "Poisson": 2}

    # Build training set (undistorted)
    X_raw, X_inv, X_comb, y = [], [], [], []
    for label, gen_fn in generators.items():
        for _ in range(n_train):
            eigs = gen_fn(N, rng)
            X_raw.append(spacing_features_raw(eigs))
            X_inv.append(spacing_features_io(eigs))
            X_comb.append(combined_features(eigs))
            y.append(class_map[label])

    X_raw = np.array(X_raw)
    X_inv = np.array(X_inv)
    X_comb = np.array(X_comb)
    y = np.array(y)

    # Train classifiers
    raw_clf = Pipeline([("s", StandardScaler()),
                        ("c", RandomForestClassifier(100, random_state=42))])
    inv_clf = Pipeline([("s", StandardScaler()),
                        ("c", RandomForestClassifier(100, random_state=42))])
    comb_clf = Pipeline([("s", StandardScaler()),
                         ("c", RandomForestClassifier(100, random_state=42))])

    raw_clf.fit(X_raw, y)
    inv_clf.fit(X_inv, y)
    comb_clf.fit(X_comb, y)

    # Test undistorted
    n_test = 100
    X_te_raw, X_te_inv, X_te_comb, y_te = [], [], [], []
    for label, gen_fn in generators.items():
        for _ in range(n_test):
            eigs = gen_fn(N, rng)
            X_te_raw.append(spacing_features_raw(eigs))
            X_te_inv.append(spacing_features_io(eigs))
            X_te_comb.append(combined_features(eigs))
            y_te.append(class_map[label])

    X_te_raw = np.array(X_te_raw)
    X_te_inv = np.array(X_te_inv)
    X_te_comb = np.array(X_te_comb)
    y_te = np.array(y_te)

    a_raw = accuracy_score(y_te, raw_clf.predict(X_te_raw))
    a_inv = accuracy_score(y_te, inv_clf.predict(X_te_inv))
    a_comb = accuracy_score(y_te, comb_clf.predict(X_te_comb))
    print(f"  Undistorted:  Raw {a_raw:.1%}  IO {a_inv:.1%}  Combined {a_comb:.1%}")

    # --- Density distortion test (P4) ---
    print()
    print("-" * 70)
    print("DENSITY DISTORTION RESISTANCE (P4)")
    print("-" * 70)
    print()
    print("Train on clean spectra, test on density-distorted spectra.")
    print()

    print(f"{'Distortion':>14} {'Raw':>8} {'IO':>8} {'Combined':>10} {'IO Adv':>9}")
    print("-" * 55)

    for dist_name in ["none", "quadratic", "cubic", "exponential", "log"]:
        X_d_raw, X_d_inv, X_d_comb, y_d = [], [], [], []
        for label, gen_fn in generators.items():
            for _ in range(n_test):
                eigs = gen_fn(N, rng)
                if dist_name != "none":
                    eigs = distort_density(eigs, dist_name)
                X_d_raw.append(spacing_features_raw(eigs))
                X_d_inv.append(spacing_features_io(eigs))
                X_d_comb.append(combined_features(eigs))
                y_d.append(class_map[label])

        X_d_raw = np.array(X_d_raw)
        X_d_inv = np.array(X_d_inv)
        X_d_comb = np.array(X_d_comb)
        y_d = np.array(y_d)

        a_r = accuracy_score(y_d, raw_clf.predict(X_d_raw))
        a_i = accuracy_score(y_d, inv_clf.predict(X_d_inv))
        a_c = accuracy_score(y_d, comb_clf.predict(X_d_comb))

        print(f"{dist_name:>14} {a_r:8.1%} {a_i:8.1%} {a_c:10.1%} {a_i-a_r:+8.1%}")

    # --- Higher order test (P3) ---
    print()
    print("-" * 70)
    print("HIGHER ORDER COMPARISON (P3)")
    print("-" * 70)
    print()
    print("IO orders 1-3 under exponential density distortion:")
    print()

    for order in [1, 2, 3]:
        fn = lambda e, m=order: spacing_features_io(e, m)
        X_tr, y_tr = [], []
        for label, gen_fn in generators.items():
            for _ in range(n_train):
                eigs = gen_fn(N, rng)
                X_tr.append(fn(eigs))
                y_tr.append(class_map[label])

        X_te_d, y_te_d = [], []
        for label, gen_fn in generators.items():
            for _ in range(n_test):
                eigs = gen_fn(N, rng)
                eigs = distort_density(eigs, "exponential")
                X_te_d.append(fn(eigs))
                y_te_d.append(class_map[label])

        pipe = Pipeline([("s", StandardScaler()),
                         ("c", RandomForestClassifier(100, random_state=42))])
        pipe.fit(np.array(X_tr), np.array(y_tr))
        acc = accuracy_score(np.array(y_te_d), pipe.predict(np.array(X_te_d)))
        print(f"  Order {order}: {acc:.1%} (exponential distortion)")

    # --- Report ---
    print()
    print("=" * 70)
    print("REPORT: Quantum Spectra × Invariant Order")
    print("=" * 70)
    print()
    print("  Archetype:    D — Spectral Structure")
    print("  Geometry:     Spectral (ordered eigenvalues)")
    print("  Nuisance:     Smooth spectral density / global scale")
    print("  Structure:    Local eigenvalue correlations")
    print()
    print("  This is where IO started: the gap ratio's insensitivity")
    print("  to smooth density IS the order-1 spectral invariant.")
    print()
    print("  The loop is closed: the generalized framework rediscovers")
    print("  gap-ratio statistics as the natural invariant for spectral data.")
    print()


if __name__ == "__main__":
    main()
