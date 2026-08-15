"""Higher-Order IO Invariants on Zeta Zeros — Searching for Beyond-GUE Structure

The question: do higher-order IO invariants see something in the zeta zeros
that first-order gap ratios don't?

Gap ratios (order 1) confirm GUE. But orders 2, 3, 4, 5 probe progressively
longer-range correlations. If zeta zeros match GUE at order 1 but deviate
at higher orders, that deviation encodes arithmetic structure — the primes
leaking through.

Protocol:
  1. Compute IO invariants at orders 1-5 for zeta zeros
  2. Compute the same for large GUE ensembles (reference distribution)
  3. Measure the deviation at each order in σ units
  4. If deviations grow with order → arithmetic corrections to GUE
  5. If deviations stay flat → GUE is exact to this resolution
"""

import sys
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from invariant_order import transform

from mpmath import zetazero, mp
mp.dps = 25


def compute_zeta_zeros(n, start=1):
    zeros = []
    for k in range(start, start + n):
        zeros.append(float(zetazero(k).imag))
    return np.array(zeros)


def generate_gue(n, rng):
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    H = (A + A.conj().T) / np.sqrt(2 * n)
    return np.sort(np.linalg.eigvalsh(H))


def gap_ratios(values):
    s = np.diff(values)
    s = s[s > 0]
    return np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])


def io_stats_at_order(values, order):
    """Compute IO statistics at a specific differential order.

    For order 1: this reduces to gap-ratio-like statistics.
    For order m: Δᵐ(log spacings) probes m-point correlations.
    """
    s = np.diff(values)
    s = s[s > 0]
    if len(s) < order + 5:
        return {}

    t = transform(s, "multiplicative", order)

    r = gap_ratios(values) if order == 1 else None

    result = {
        "mean": np.mean(t),
        "var": np.var(t),
        "skew": float(stats.skew(t)),
        "kurtosis": float(stats.kurtosis(t)),
        "mean_abs": np.mean(np.abs(t)),
        "p10": np.percentile(t, 10),
        "p50": np.percentile(t, 50),
        "p90": np.percentile(t, 90),
        "autocorr1": np.corrcoef(t[:-1], t[1:])[0, 1] if len(t) > 2 else 0,
    }

    if order == 1 and r is not None and len(r) > 0:
        result["gap_ratio_mean"] = np.mean(r)

    return result


def main():
    print("=" * 70)
    print("HIGHER-ORDER IO INVARIANTS ON ZETA ZEROS")
    print("Searching for beyond-GUE structure")
    print("=" * 70)
    print()

    rng = np.random.default_rng(42)

    # --- Compute zeta zeros ---
    n_zeros = 1000
    print(f"Computing {n_zeros} zeta zeros...", flush=True)
    zeros = compute_zeta_zeros(n_zeros)
    print(f"  Range: t = {zeros[0]:.2f} to {zeros[-1]:.2f}")
    print()

    # Also get zeros at different heights for consistency check
    print("Computing zeros 2001-3000 (higher height)...", flush=True)
    zeros_high = compute_zeta_zeros(1000, start=2001)
    print(f"  Range: t = {zeros_high[0]:.2f} to {zeros_high[-1]:.2f}")
    print()

    # --- Generate GUE reference ensemble ---
    n_gue = 500  # number of GUE realizations
    N_matrix = 300  # matrix size
    print(f"Generating {n_gue} GUE matrices ({N_matrix}×{N_matrix})...", flush=True)

    max_order = 5
    gue_stats = {order: {key: [] for key in
                         ["mean", "var", "skew", "kurtosis", "mean_abs",
                          "p10", "p50", "p90", "autocorr1"]}
                 for order in range(1, max_order + 1)}

    for i in range(n_gue):
        eigs = generate_gue(N_matrix, rng)
        for order in range(1, max_order + 1):
            st = io_stats_at_order(eigs, order)
            for key in gue_stats[order]:
                if key in st:
                    gue_stats[order][key].append(st[key])

    # Convert to arrays
    for order in gue_stats:
        for key in gue_stats[order]:
            gue_stats[order][key] = np.array(gue_stats[order][key])

    print(f"  Done.")
    print()

    # --- Compute zeta stats at each order ---
    print("-" * 70)
    print("IO STATISTICS: ZETA ZEROS vs GUE REFERENCE")
    print("-" * 70)
    print()

    stat_names = ["mean", "var", "skew", "kurtosis", "mean_abs",
                  "p10", "p50", "p90", "autocorr1"]

    for order in range(1, max_order + 1):
        print(f"  ORDER {order}:")
        print(f"  {'Statistic':>12} {'Zeta':>10} {'GUE mean':>10} {'GUE std':>10} "
              f"{'Deviation':>10} {'Zeta-high':>10}")
        print(f"  {'-'*65}")

        zeta_st = io_stats_at_order(zeros, order)
        zeta_high_st = io_stats_at_order(zeros_high, order)

        deviations = []
        for key in stat_names:
            if key not in zeta_st or key not in gue_stats[order]:
                continue
            gue_mean = np.mean(gue_stats[order][key])
            gue_std = np.std(gue_stats[order][key])
            zeta_val = zeta_st[key]
            zeta_high_val = zeta_high_st.get(key, float('nan'))

            if gue_std > 1e-10:
                dev = (zeta_val - gue_mean) / gue_std
            else:
                dev = 0.0

            deviations.append((key, abs(dev)))

            sig = ""
            if abs(dev) > 3:
                sig = " ***"
            elif abs(dev) > 2:
                sig = " **"
            elif abs(dev) > 1.5:
                sig = " *"

            print(f"  {key:>12} {zeta_val:10.4f} {gue_mean:10.4f} {gue_std:10.4f} "
                  f"{dev:+9.2f}σ {zeta_high_val:10.4f}{sig}")

        max_dev = max(deviations, key=lambda x: x[1])
        print(f"  → Max deviation: {max_dev[0]} at {max_dev[1]:.2f}σ")
        print()

    # --- Deviation growth analysis ---
    print("-" * 70)
    print("DEVIATION GROWTH ACROSS ORDERS")
    print("-" * 70)
    print()
    print("If deviations grow with order → arithmetic structure beyond GUE")
    print("If deviations stay flat → GUE is exact to this resolution")
    print()

    order_max_devs = []
    order_mean_devs = []

    for order in range(1, max_order + 1):
        zeta_st = io_stats_at_order(zeros, order)
        devs = []
        for key in stat_names:
            if key not in zeta_st or key not in gue_stats[order]:
                continue
            gue_mean = np.mean(gue_stats[order][key])
            gue_std = np.std(gue_stats[order][key])
            if gue_std > 1e-10:
                devs.append(abs((zeta_st[key] - gue_mean) / gue_std))
        order_max_devs.append(max(devs))
        order_mean_devs.append(np.mean(devs))

    print(f"  {'Order':>6} {'Max |dev|':>10} {'Mean |dev|':>11} {'Pattern':>10}")
    print(f"  {'-'*42}")
    for i, order in enumerate(range(1, max_order + 1)):
        pattern = ""
        if i > 0:
            if order_max_devs[i] > order_max_devs[i-1] * 1.3:
                pattern = "GROWING ↑"
            elif order_max_devs[i] < order_max_devs[i-1] * 0.7:
                pattern = "shrinking ↓"
            else:
                pattern = "flat →"
        print(f"  {order:6d} {order_max_devs[i]:10.2f}σ {order_mean_devs[i]:10.2f}σ "
              f"{pattern:>10}")

    # --- Specific probe: autocorrelation structure ---
    print()
    print("-" * 70)
    print("AUTOCORRELATION PROBE")
    print("-" * 70)
    print()
    print("Autocorrelation of IO-transformed spacings at each order.")
    print("GUE has specific correlation structure; deviations = arithmetic.")
    print()

    for order in range(1, max_order + 1):
        s = np.diff(zeros)
        s = s[s > 0]
        t = transform(s, "multiplicative", order)

        autocorrs = []
        for lag in range(1, 6):
            if len(t) > lag:
                c = np.corrcoef(t[:-lag], t[lag:])[0, 1]
                autocorrs.append(c)
            else:
                autocorrs.append(0)

        gue_ac = []
        for lag in range(1, 6):
            ac_samples = []
            for _ in range(200):
                eigs = generate_gue(N_matrix, rng)
                s_g = np.diff(eigs)
                s_g = s_g[s_g > 0]
                t_g = transform(s_g, "multiplicative", order)
                if len(t_g) > lag:
                    ac_samples.append(np.corrcoef(t_g[:-lag], t_g[lag:])[0, 1])
            gue_ac.append((np.mean(ac_samples), np.std(ac_samples)))

        print(f"  Order {order}:")
        for lag in range(5):
            gm, gs = gue_ac[lag]
            dev = (autocorrs[lag] - gm) / gs if gs > 1e-10 else 0
            sig = "***" if abs(dev) > 3 else "**" if abs(dev) > 2 else "*" if abs(dev) > 1.5 else ""
            print(f"    lag {lag+1}: zeta={autocorrs[lag]:+.4f}  "
                  f"GUE={gm:+.4f}±{gs:.4f}  dev={dev:+.1f}σ {sig}")
        print()

    # --- Height consistency ---
    print("-" * 70)
    print("HEIGHT CONSISTENCY CHECK")
    print("-" * 70)
    print()
    print("Do deviations from GUE persist at different heights?")
    print("(Consistent deviations = real structure, not finite-size noise)")
    print()

    for order in [1, 2, 3]:
        st_low = io_stats_at_order(zeros, order)
        st_high = io_stats_at_order(zeros_high, order)

        print(f"  Order {order}:")
        for key in ["var", "skew", "kurtosis", "autocorr1"]:
            gue_mean = np.mean(gue_stats[order][key])
            gue_std = np.std(gue_stats[order][key])
            dev_low = (st_low[key] - gue_mean) / gue_std if gue_std > 1e-10 else 0
            dev_high = (st_high[key] - gue_mean) / gue_std if gue_std > 1e-10 else 0

            same_dir = "CONSISTENT" if dev_low * dev_high > 0 else "opposite"
            print(f"    {key:>12}: low={dev_low:+.2f}σ  high={dev_high:+.2f}σ  {same_dir}")
        print()

    # --- Report ---
    print("=" * 70)
    print("REPORT: Higher-Order IO on Zeta Zeros")
    print("=" * 70)
    print()

    any_significant = any(d > 2.0 for d in order_max_devs)
    growing = all(order_max_devs[i] >= order_max_devs[i-1] * 0.9
                  for i in range(1, len(order_max_devs)))

    print(f"  Orders tested:  1-{max_order}")
    print(f"  Zeros analyzed: {n_zeros} (low) + 1000 (high)")
    print(f"  GUE reference:  {n_gue} matrices, {N_matrix}×{N_matrix}")
    print()

    if any_significant:
        print("  FINDING: Statistically significant deviations from GUE detected")
        print("           at higher IO orders.")
        if growing:
            print("  The deviation GROWS with order, suggesting longer-range")
            print("  arithmetic correlations beyond GUE.")
        print()
        print("  Significance: gap ratios (order 1) match GUE, but higher-order")
        print("  IO statistics reveal structure that gap ratios cannot see.")
        print("  This could represent arithmetic corrections to the GUE model.")
    else:
        print("  FINDING: No significant deviations from GUE detected through")
        print(f"  order {max_order} at this sample size ({n_zeros} zeros).")
        print()
        print("  The GUE model appears exact to the resolution of this experiment.")
        print("  Larger zero samples or higher orders may be needed to detect")
        print("  arithmetic corrections, if they exist.")

    print()


if __name__ == "__main__":
    main()
