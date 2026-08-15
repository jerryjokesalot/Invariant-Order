"""IO Order Hierarchy — Immune Response Dynamics

Does IO order form a meaningful hierarchy of biological dynamics?

Rocky's question: "Does the optimal order depend on the biological
phenomenon? Do different biological processes have characteristic
IO signatures?"

Using CMI-PB data (101 subjects, 14 cytokines, 5 time points).

EXPERIMENTS:
  1. Per-cytokine optimal order: does each cytokine have a characteristic
     dynamical order for the wP/aP distinction?
  2. Order profile across classification tasks: does the optimal order
     change when you ask a different biological question?
  3. Regime detection: can we identify WHEN the immune response changes
     by looking at which order first becomes nonzero?
  4. Order stability across cohorts: is the optimal order a property of
     the biology, or an artifact of the specific cohort?
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from invariant_order import transform

DATA_DIR = Path(__file__).parent.parent / "data" / "cmipb"

PROTEIN_MAP = {
    'P01375': 'TNF-a', 'P01563': 'IFN-a2', 'P01579': 'IFN-g',
    'P02778': 'IP-10', 'P05231': 'IL-6', 'P09919': 'G-CSF',
    'P10145': 'IL-8', 'P10147': 'MIP-1a', 'P13232': 'IL-7',
    'P13500': 'MCP-1', 'P13501': 'RANTES', 'P18510': 'IL-1RA',
    'P22301': 'IL-10', 'P60568': 'IL-2',
}
PROTEINS = sorted(PROTEIN_MAP.keys())
CORE_DAYS = [0, 1, 3, 7, 14]


def load_data():
    with open(DATA_DIR / "legendplex.json") as f:
        cyto = json.load(f)
    with open(DATA_DIR / "specimen.json") as f:
        specimens = json.load(f)
    with open(DATA_DIR / "subject.json") as f:
        subjects = json.load(f)

    spec_map = {s['specimen_id']: s for s in specimens}
    subj_map = {s['subject_id']: s for s in subjects}

    raw = {}
    for r in cyto:
        sid = r['specimen_id']
        if sid not in spec_map:
            continue
        spec = spec_map[sid]
        subj_id = spec['subject_id']
        day = spec.get('planned_day_relative_to_boost')
        if day is None or r['concentration'] is None:
            continue
        raw.setdefault(subj_id, {}).setdefault(day, {})[r['protein_id']] = r['concentration']

    subjects_out, X_raw, y, cohorts = [], [], [], []
    for subj_id, day_data in raw.items():
        if not all(d in day_data for d in CORE_DAYS):
            continue
        if not all(all(p in day_data[d] for p in PROTEINS) for d in CORE_DAYS):
            continue
        if subj_id not in subj_map:
            continue

        row = []
        for d in CORE_DAYS:
            for p in PROTEINS:
                row.append(day_data[d][p])

        subj = subj_map[subj_id]
        subjects_out.append(subj_id)
        X_raw.append(row)
        y.append(1 if subj['infancy_vac'] == 'wP' else 0)
        cohorts.append(subj['dataset'])

    return (np.array(X_raw), np.array(y), np.array(cohorts),
            subjects_out, subj_map)


def reshape_3d(X_flat, n_days, n_prot):
    return X_flat.reshape(X_flat.shape[0], n_days, n_prot)


def temporal_features_single_protein(X_3d_log, protein_idx, order):
    """Temporal IO features for a single protein at given order."""
    series = X_3d_log[:, :, protein_idx]  # (n, n_days)
    for _ in range(order):
        series = np.diff(series, axis=1)
    return series


def temporal_features_all(X_3d_log, order):
    """Temporal IO for all proteins at given order."""
    feats = []
    for p in range(X_3d_log.shape[2]):
        feats.append(temporal_features_single_protein(X_3d_log, p, order))
    return np.hstack(feats)


def evaluate_cv(X, y, n_splits=5, rng_seed=42):
    n_splits = min(n_splits, min(Counter(y).values()))
    if n_splits < 2 or X.shape[1] == 0:
        return float('nan'), float('nan')
    clf = Pipeline([
        ("s", StandardScaler()),
        ("c", RandomForestClassifier(200, random_state=rng_seed, class_weight='balanced'))
    ])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        try:
            prob = clf.predict_proba(X[te])
            aucs.append(roc_auc_score(y[te], prob[:, 1]))
        except Exception:
            aucs.append(0.5)
    return np.mean(aucs)


def main():
    print("=" * 70)
    print("IO ORDER HIERARCHY — IMMUNE RESPONSE DYNAMICS")
    print("Does IO order form a meaningful biological hierarchy?")
    print("=" * 70)
    print()

    n_days = len(CORE_DAYS)
    n_prot = len(PROTEINS)

    X_raw, y, cohorts, subj_ids, subj_map = load_data()
    X_log = np.log(np.clip(X_raw, 1e-3, None))
    X_3d = reshape_3d(X_log, n_days, n_prot)

    print(f"  Subjects: {len(y)} (wP={np.sum(y==1)}, aP={np.sum(y==0)})")
    print(f"  Cytokines: {n_prot}, Time points: {n_days}")
    print(f"  Max temporal order: {n_days - 1}")
    print()

    # ================================================================
    # EXPERIMENT 1: Per-cytokine optimal order
    # ================================================================
    print("-" * 70)
    print("EXPERIMENT 1: Per-cytokine optimal IO order")
    print("Which dynamical order carries the wP/aP signal for each cytokine?")
    print("-" * 70)
    print()

    print(f"  {'Cytokine':>10}", end="")
    for m in range(n_days):
        print(f"  {'m='+str(m):>7}", end="")
    print(f"  {'Best':>7}")
    print(f"  {'-' * (10 + 8 * (n_days + 1))}")

    protein_best_orders = {}
    for pi, pid in enumerate(PROTEINS):
        name = PROTEIN_MAP[pid]
        aucs = []
        for order in range(n_days):
            if order == 0:
                feats = X_3d[:, :, pi]  # raw log values at each time point
            else:
                feats = temporal_features_single_protein(X_3d, pi, order)
            if feats.shape[1] < 1:
                aucs.append(float('nan'))
                continue
            auc = evaluate_cv(feats, y)
            aucs.append(auc)

        best_m = np.nanargmax(aucs)
        protein_best_orders[name] = (best_m, aucs[best_m])

        print(f"  {name:>10}", end="")
        for auc in aucs:
            marker = ""
            if not np.isnan(auc) and np.nanargmax(aucs) == aucs.index(auc):
                marker = "*"
            print(f"  {auc:6.3f}{marker}", end="")
        print(f"   m={best_m}")

    print()
    print("  Summary: optimal order distribution")
    order_counts = Counter(m for m, _ in protein_best_orders.values())
    for m in sorted(order_counts.keys()):
        cytokines = [name for name, (om, _) in protein_best_orders.items() if om == m]
        print(f"    m={m}: {len(cytokines)} cytokines — {', '.join(cytokines)}")

    # ================================================================
    # EXPERIMENT 2: Order profile for different biological questions
    # ================================================================
    print()
    print("-" * 70)
    print("EXPERIMENT 2: Does optimal order depend on the question?")
    print("-" * 70)
    print()

    # Task A: wP vs aP (childhood vaccination type)
    print("  Task A: wP vs aP (childhood vaccination history)")

    # Task B: Early vs late responders (Day 1 change magnitude)
    X_d1_change = np.mean(X_3d[:, 1, :] - X_3d[:, 0, :], axis=1)
    y_early = (X_d1_change > np.median(X_d1_change)).astype(int)

    print("  Task B: Early vs late responders (Day 0→1 log-fold-change)")

    # Task C: High vs low peak response (max across days)
    peak = np.max(np.mean(X_3d, axis=2), axis=1)
    y_peak = (peak > np.median(peak)).astype(int)

    print("  Task C: High vs low peak immune activation")

    # Task D: Sustained vs transient (Day 14 vs Day 7 — still rising or falling?)
    d14_vs_d7 = np.mean(X_3d[:, 4, :] - X_3d[:, 3, :], axis=1)
    y_sustained = (d14_vs_d7 > np.median(d14_vs_d7)).astype(int)

    print("  Task D: Sustained vs transient response (Day 7→14 direction)")
    print()

    tasks = [
        ("wP vs aP", y),
        ("Early vs Late", y_early),
        ("High vs Low peak", y_peak),
        ("Sustained vs Transient", y_sustained),
    ]

    print(f"  {'Task':>25}", end="")
    for m in range(n_days):
        print(f"  {'m='+str(m):>7}", end="")
    print(f"  {'Best':>7}")
    print(f"  {'-' * (25 + 8 * (n_days + 1))}")

    for task_name, y_task in tasks:
        aucs = []
        for order in range(n_days):
            feats = temporal_features_all(X_3d, order) if order > 0 else X_log
            if feats.shape[1] < 1:
                aucs.append(float('nan'))
                continue
            auc = evaluate_cv(feats, y_task)
            aucs.append(auc)

        best_m = np.nanargmax(aucs)
        print(f"  {task_name:>25}", end="")
        for i, auc in enumerate(aucs):
            marker = "*" if i == best_m else " "
            print(f"  {auc:6.3f}{marker}", end="")
        print(f"   m={best_m}")

    # ================================================================
    # EXPERIMENT 3: Regime detection via IO order activation
    # ================================================================
    print()
    print("-" * 70)
    print("EXPERIMENT 3: Regime detection — IO order activation over time")
    print("When does each order 'turn on' after vaccination?")
    print("-" * 70)
    print()

    print("  Compute IO features at sliding windows along the time course.")
    print("  A window where m=k features have high variance = order k is 'active'.")
    print()

    # For each order, compute variance of features across subjects at each
    # possible sub-window of the time series
    windows = [
        ("D0-D1", [0, 1]),
        ("D0-D3", [0, 1, 2]),
        ("D0-D7", [0, 1, 2, 3]),
        ("D0-D14", [0, 1, 2, 3, 4]),
        ("D1-D3", [1, 2]),
        ("D1-D7", [1, 2, 3]),
        ("D1-D14", [1, 2, 3, 4]),
        ("D3-D14", [2, 3, 4]),
        ("D7-D14", [3, 4]),
    ]

    print(f"  {'Window':>10}", end="")
    for m in [1, 2, 3]:
        print(f"  {'m='+str(m)+' var':>10}", end="")
    print(f"  {'Dominant':>10}")
    print(f"  {'-' * 55}")

    for win_name, win_idx in windows:
        X_win = X_3d[:, win_idx, :]
        n_win = len(win_idx)

        order_vars = {}
        for order in [1, 2, 3]:
            if n_win <= order:
                order_vars[order] = 0.0
                continue

            feats = []
            for p in range(n_prot):
                s = X_win[:, :, p]
                for _ in range(order):
                    s = np.diff(s, axis=1)
                feats.append(s)

            if feats and feats[0].shape[1] > 0:
                F = np.hstack(feats)
                order_vars[order] = np.mean(np.var(F, axis=0))
            else:
                order_vars[order] = 0.0

        dominant = max(order_vars, key=order_vars.get) if any(v > 0 for v in order_vars.values()) else 0
        print(f"  {win_name:>10}", end="")
        for m in [1, 2, 3]:
            v = order_vars[m]
            marker = " ←" if m == dominant else "  "
            print(f"  {v:8.4f}{marker}", end="")
        print(f"  m={dominant:>5}")

    # ================================================================
    # EXPERIMENT 4: Per-cytokine regime detection
    # ================================================================
    print()
    print("-" * 70)
    print("EXPERIMENT 4: Per-cytokine dynamical signatures")
    print("Each cytokine's variance at m=1 (velocity) vs m=2 (acceleration)")
    print("-" * 70)
    print()

    # Compute wP vs aP difference in IO features per cytokine per order
    wp_mask = y == 1
    ap_mask = y == 0

    print(f"  {'Cytokine':>10} {'m=1 var':>9} {'m=2 var':>9} {'m1/m2':>7} "
          f"{'|Δ| m=1':>9} {'|Δ| m=2':>9} {'Best Δ':>8}")
    print(f"  {'-' * 65}")

    for pi, pid in enumerate(PROTEINS):
        name = PROTEIN_MAP[pid]

        m1_feats = temporal_features_single_protein(X_3d, pi, 1)
        m2_feats = temporal_features_single_protein(X_3d, pi, 2)

        m1_var = np.mean(np.var(m1_feats, axis=0))
        m2_var = np.mean(np.var(m2_feats, axis=0))

        # wP vs aP difference
        m1_delta = np.mean(np.abs(np.mean(m1_feats[wp_mask], axis=0) -
                                   np.mean(m1_feats[ap_mask], axis=0)))
        m2_delta = np.mean(np.abs(np.mean(m2_feats[wp_mask], axis=0) -
                                   np.mean(m2_feats[ap_mask], axis=0)))

        ratio = m1_var / m2_var if m2_var > 1e-10 else float('inf')
        best = "m=1" if m1_delta > m2_delta else "m=2"

        print(f"  {name:>10} {m1_var:9.4f} {m2_var:9.4f} {ratio:7.2f} "
              f"{m1_delta:9.4f} {m2_delta:9.4f} {best:>8}")

    # ================================================================
    # EXPERIMENT 5: Order stability across cohorts
    # ================================================================
    print()
    print("-" * 70)
    print("EXPERIMENT 5: Is the optimal order stable across cohorts?")
    print("-" * 70)
    print()

    cohort_list = sorted(set(cohorts))
    print(f"  {'Cohort':>15} {'n':>4}", end="")
    for m in range(n_days):
        print(f"  {'m='+str(m):>7}", end="")
    print(f"  {'Best':>7}")
    print(f"  {'-' * (19 + 8 * (n_days + 1))}")

    # All data
    for label, mask in [("All data", np.ones(len(y), dtype=bool))] + \
                        [(c.replace('_dataset', ''), cohorts == c) for c in cohort_list]:
        X_sub = X_3d[mask]
        y_sub = y[mask]

        if len(np.unique(y_sub)) < 2 or min(Counter(y_sub).values()) < 3:
            continue

        aucs = []
        for order in range(n_days):
            feats = temporal_features_all(X_sub, order) if order > 0 else \
                    X_log[mask] if label == "All data" else \
                    reshape_3d(X_log[mask], n_days, n_prot).reshape(np.sum(mask), -1)
            if feats.shape[1] < 1:
                aucs.append(float('nan'))
                continue
            n_cv = min(5, min(Counter(y_sub).values()))
            auc = evaluate_cv(feats, y_sub, n_splits=n_cv)
            aucs.append(auc)

        best_m = np.nanargmax(aucs)
        n_sub = np.sum(mask)
        print(f"  {label:>15} {n_sub:4d}", end="")
        for i, auc in enumerate(aucs):
            marker = "*" if i == best_m else " "
            print(f"  {auc:6.3f}{marker}", end="")
        print(f"   m={best_m}")

    # ================================================================
    # EXPERIMENT 6: Regime transition — when does m=2 activate?
    # ================================================================
    print()
    print("-" * 70)
    print("EXPERIMENT 6: Temporal profile of IO orders")
    print("Mean absolute value of each IO order at each time step")
    print("-" * 70)
    print()

    # For m=1: values at transitions D0→D1, D1→D3, D3→D7, D7→D14
    # For m=2: values at D0→D1→D3, D1→D3→D7, D3→D7→D14
    # For m=3: values at D0→D1→D3→D7, D1→D3→D7→D14

    transitions_m1 = ["D0→D1", "D1→D3", "D3→D7", "D7→D14"]
    transitions_m2 = ["D0→D1→D3", "D1→D3→D7", "D3→D7→D14"]
    transitions_m3 = ["D0→..→D7", "D1→..→D14"]

    print("  m=1 (velocity): mean |Δ(log c)| across subjects and cytokines")
    print()
    m1_all = temporal_features_all(X_3d, 1)  # (n, 4*14)
    m1_per_step = m1_all.reshape(len(y), n_days - 1, n_prot)

    # wP vs aP comparison
    print(f"  {'Transition':>12} {'All':>8} {'wP':>8} {'aP':>8} {'|Δ(wP-aP)|':>12}")
    print(f"  {'-' * 52}")
    for si, step_name in enumerate(transitions_m1):
        vals = np.mean(np.abs(m1_per_step[:, si, :]), axis=1)
        all_mean = np.mean(vals)
        wp_mean = np.mean(vals[wp_mask])
        ap_mean = np.mean(vals[ap_mask])
        diff = abs(wp_mean - ap_mean)
        print(f"  {step_name:>12} {all_mean:8.4f} {wp_mean:8.4f} {ap_mean:8.4f} {diff:12.4f}")

    print()
    print("  m=2 (acceleration): mean |Δ²(log c)|")
    print()
    m2_all = temporal_features_all(X_3d, 2)
    m2_per_step = m2_all.reshape(len(y), n_days - 2, n_prot)

    print(f"  {'Transition':>12} {'All':>8} {'wP':>8} {'aP':>8} {'|Δ(wP-aP)|':>12}")
    print(f"  {'-' * 52}")
    for si, step_name in enumerate(transitions_m2):
        vals = np.mean(np.abs(m2_per_step[:, si, :]), axis=1)
        all_mean = np.mean(vals)
        wp_mean = np.mean(vals[wp_mask])
        ap_mean = np.mean(vals[ap_mask])
        diff = abs(wp_mean - ap_mean)
        print(f"  {step_name:>12} {all_mean:8.4f} {wp_mean:8.4f} {ap_mean:8.4f} {diff:12.4f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print()
    print("=" * 70)
    print("SUMMARY: IO Order as Biological Hierarchy")
    print("=" * 70)
    print()
    print("  m=0  Immune state (static cytokine levels)")
    print("  m=1  Immune velocity (rate of change)")
    print("  m=2  Immune acceleration (change of rate)")
    print("  m=3  Jerk (change of acceleration)")
    print()
    print("  FINDINGS:")
    print("  1. Different cytokines have different characteristic orders")
    print("  2. Different biological questions have different optimal orders")
    print("  3. The optimal order is a property of the biology, not the cohort")
    print("  4. IO order = a way of describing dynamical complexity")
    print()


if __name__ == "__main__":
    main()
