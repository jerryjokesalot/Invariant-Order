"""Immune System (CMI-PB) — Invariant Order Experiment

Archetype F: Immunological monitoring (spatial + temporal geometry)
Scale-up of FluPRINT experiment with proper temporal dynamics.

DATA: CMI-PB (Computational Models of Immunity — Pertussis Boost)
  - 101 subjects across 3 cohorts (2021, 2022, 2023)
  - 14 serum cytokines via LEGENDplex in raw pg/mL
  - 5 time points: Day 0, 1, 3, 7, 14 post-Tdap booster
  - Classification: wP-primed (52) vs aP-primed (49) infancy vaccination
  - Raw concentrations — real multiplicative assay variation

DESIGN CONTRACTS:

  SPATIAL (cross-cytokine ratios at each time point):
    Geometry:     Spatial (14 parallel cytokine measurements)
    Nuisance:     Multiplicative (assay gain, sample dilution, subject baseline)
    Structure:    Relative cytokine coordination (immune signature)
    Invariant:    log(c_i) - log(c_j) across cytokines

  TEMPORAL (cytokine dynamics along 5 time points):
    Geometry:     Temporal (Day 0 → 1 → 3 → 7 → 14)
    Nuisance:     Multiplicative baseline (each subject's set point)
    Structure:    Vaccination response dynamics
    Invariant:    Δᵐ(log c) along time axis, order m

  COMBINED (spatial × temporal):
    Both invariants applied on orthogonal axes within the same dataset

PREDICTIONS:
  P1: Spatial IO features classify wP vs aP better than raw pg/mL
  P2: Temporal IO features are invariant to multiplicative drift
      (theorem verification on 5-point time series)
  P3: Cross-cohort transfer: IO features transfer between cohorts
      better than raw (2021→2022, 2022→2023, etc.)
  P4: Combined spatial+temporal outperforms either alone
  P5: Higher-order temporal IO (m=2) captures curvature of immune
      response that m=1 misses

PREDICTED FAILURE MODES:
  F1: [D] If wP/aP distinction is in absolute magnitudes (not ratios/dynamics),
      IO is blind to it
  F2: [C] Non-multiplicative batch differences between cohorts
  F3: [B] 5 time points limits temporal order to m≤3
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
    """Load CMI-PB LEGENDplex cytokine data."""
    with open(DATA_DIR / "legendplex.json") as f:
        cyto = json.load(f)
    with open(DATA_DIR / "specimen.json") as f:
        specimens = json.load(f)
    with open(DATA_DIR / "subject.json") as f:
        subjects = json.load(f)

    spec_map = {s['specimen_id']: s for s in specimens}
    subj_map = {s['subject_id']: s for s in subjects}

    # subject -> day -> protein -> concentration
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

    # Filter to subjects with all 5 core days and all 14 proteins
    subjects_out = []
    X_raw = []  # (n_subjects, n_days * n_proteins) in raw pg/mL
    y = []      # 0=aP, 1=wP
    cohorts = []

    for subj_id, day_data in raw.items():
        if not all(d in day_data for d in CORE_DAYS):
            continue
        if not all(all(p in day_data[d] for p in PROTEINS) for d in CORE_DAYS):
            continue

        row = []
        for d in CORE_DAYS:
            for p in PROTEINS:
                row.append(day_data[d][p])

        if subj_id not in subj_map:
            continue

        subj = subj_map[subj_id]
        label = 1 if subj['infancy_vac'] == 'wP' else 0

        subjects_out.append(subj_id)
        X_raw.append(row)
        y.append(label)
        cohorts.append(subj['dataset'])

    return (np.array(X_raw), np.array(y), np.array(cohorts),
            subjects_out, subj_map)


def reshape_3d(X, n_days, n_proteins):
    """Reshape flat (n, days*proteins) → (n, days, proteins)."""
    return X.reshape(X.shape[0], n_days, n_proteins)


def spatial_log_ratios(X_2d):
    """Cross-cytokine log-differences at a single time point.

    X_2d: (n, n_proteins) in log space
    Returns: (n, n_proteins-1) adjacent log-ratios
    """
    return np.diff(X_2d, axis=1)


def spatial_features(X_3d_log):
    """Spatial IO features: cross-cytokine log-ratios at each time point.

    X_3d_log: (n, n_days, n_proteins) in log space
    Returns: (n, n_days * (n_proteins-1))
    """
    feats = []
    for d in range(X_3d_log.shape[1]):
        feats.append(spatial_log_ratios(X_3d_log[:, d, :]))
    return np.hstack(feats)


def temporal_features(X_3d_log, order=1):
    """Temporal IO features: Δᵐ(log) along time for each protein.

    X_3d_log: (n, n_days, n_proteins) in log space
    Returns: (n, (n_days-order) * n_proteins)
    """
    feats = []
    for p in range(X_3d_log.shape[2]):
        series = X_3d_log[:, :, p]  # (n, n_days)
        for _ in range(order):
            series = np.diff(series, axis=1)
        feats.append(series)
    return np.hstack(feats)


def combined_features(X_3d_log, order=1):
    """Full IO: spatial at each time + temporal for each protein."""
    return np.hstack([
        spatial_features(X_3d_log),
        temporal_features(X_3d_log, order=1),
    ])


def apply_drift(X_raw, drift_factor, rng):
    """Multiplicative drift on raw pg/mL values.

    Each protein gets a random per-protein scale factor (same across time
    for that protein — simulating assay-level gain).
    """
    n_days = len(CORE_DAYS)
    n_proteins = len(PROTEINS)
    X_3d = X_raw.reshape(-1, n_days, n_proteins)

    scales = np.exp(rng.normal(0, drift_factor, n_proteins))
    X_drifted = X_3d * scales[np.newaxis, np.newaxis, :]
    return X_drifted.reshape(X_raw.shape)


def evaluate_cv(X, y, n_splits=5, rng_seed=42):
    """Stratified cross-validation."""
    clf = Pipeline([
        ("s", StandardScaler()),
        ("c", RandomForestClassifier(200, random_state=rng_seed, class_weight='balanced'))
    ])
    n_splits = min(n_splits, min(Counter(y).values()))
    if n_splits < 2:
        return float('nan'), float('nan')

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
    aucs, accs = [], []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        try:
            prob = clf.predict_proba(X[te])
            aucs.append(roc_auc_score(y[te], prob[:, 1]))
        except Exception:
            aucs.append(accs[-1])
    return np.mean(accs), np.mean(aucs)


def main():
    print("=" * 70)
    print("CMI-PB IMMUNE VACCINATION — INVARIANT ORDER EXPERIMENT")
    print("14 cytokines × 5 time points × 101 subjects × 3 cohorts")
    print("=" * 70)
    print()

    rng = np.random.default_rng(42)
    n_days = len(CORE_DAYS)
    n_prot = len(PROTEINS)

    # --- Load data ---
    X_raw, y, cohorts, subj_ids, subj_map = load_data()
    X_log = np.log(np.clip(X_raw, 1e-3, None))
    X_3d_log = reshape_3d(X_log, n_days, n_prot)

    print(f"  Subjects: {len(y)} (wP={np.sum(y==1)}, aP={np.sum(y==0)})")
    print(f"  Cohorts: {dict(Counter(cohorts))}")
    print(f"  Cytokines: {n_prot}")
    print(f"  Time points: {CORE_DAYS}")
    print(f"  Feature space: {X_raw.shape[1]} raw values ({n_days}d × {n_prot}p)")
    print()

    # ================================================================
    # PART 1: CLASSIFICATION (wP vs aP)
    # ================================================================
    print("-" * 70)
    print("PART 1: CLASSIFICATION — wP vs aP infancy vaccination")
    print("-" * 70)
    print()

    X_spatial = spatial_features(X_3d_log)
    X_temporal_m1 = temporal_features(X_3d_log, order=1)
    X_temporal_m2 = temporal_features(X_3d_log, order=2)
    X_combined = combined_features(X_3d_log)

    # Day 0 only features
    X_d0_raw = X_raw[:, :n_prot]
    X_d0_log = X_log[:, :n_prot]
    X_d0_spatial = spatial_log_ratios(X_d0_log)

    print(f"  {'Features':>30} {'Acc':>8} {'AUC':>8} {'Dims':>6}")
    print(f"  {'-'*58}")

    for name, X_f in [
        ("Day 0 raw pg/mL", X_d0_raw),
        ("Day 0 log(pg/mL)", X_d0_log),
        ("Day 0 IO Spatial", X_d0_spatial),
        ("All days raw", X_raw),
        ("All days log", X_log),
        ("IO Spatial (all days)", X_spatial),
        ("IO Temporal m=1", X_temporal_m1),
        ("IO Temporal m=2", X_temporal_m2),
        ("IO Combined (S+T)", X_combined),
    ]:
        acc, auc = evaluate_cv(X_f, y)
        print(f"  {name:>30} {acc:8.1%} {auc:8.3f} {X_f.shape[1]:6d}")

    # ================================================================
    # PART 2: THEOREM VERIFICATION — DRIFT INVARIANCE
    # ================================================================
    print()
    print("-" * 70)
    print("PART 2: THEOREM VERIFICATION — Multiplicative drift invariance")
    print("Drift = per-protein random scale factor (same across time)")
    print("-" * 70)
    print()

    # 2a: Numerical stability
    print("  2a. Feature RMSD under drift (lower = more stable)")
    print()
    print(f"  {'Drift':>8} {'Raw RMSD':>10} {'Spat RMSD':>10} {'Temp RMSD':>10} {'S/R':>8} {'T/R':>8}")
    print(f"  {'-'*60}")

    for drift in [0.1, 0.5, 1.0, 2.0, 5.0]:
        raw_d, spat_d, temp_d = [], [], []
        for trial in range(20):
            trial_rng = np.random.default_rng(100 + trial)
            X_drifted = apply_drift(X_raw, drift, trial_rng)
            X_drifted_log = np.log(np.clip(X_drifted, 1e-3, None))
            X_drifted_3d = reshape_3d(X_drifted_log, n_days, n_prot)

            raw_d.append(np.sqrt(np.mean((X_drifted - X_raw) ** 2)))
            spat_d.append(np.sqrt(np.mean((spatial_features(X_drifted_3d) - X_spatial) ** 2)))
            temp_d.append(np.sqrt(np.mean((temporal_features(X_drifted_3d) - X_temporal_m1) ** 2)))

        r = np.mean(raw_d)
        s = np.mean(spat_d)
        t = np.mean(temp_d)
        print(f"  {drift:8.1f} {r:10.1f} {s:10.4f} {t:10.4f} {s/r:8.5f} {t/r:8.5f}")

    # 2b: Classification under drift (drift on test set only)
    print()
    print("  2b. Classification AUC under drift (train clean, test drifted)")
    print()
    print(f"  {'Drift':>8} {'Raw AUC':>9} {'Spatial':>9} {'Temp m=1':>9} {'Combined':>9}")
    print(f"  {'-'*50}")

    for drift in [0.0, 0.5, 1.0, 2.0, 5.0]:
        results = {}
        for feat_name, feat_fn in [
            ("Raw", lambda X: X),
            ("Spatial", lambda X: spatial_features(reshape_3d(np.log(np.clip(X, 1e-3, None)), n_days, n_prot))),
            ("Temp", lambda X: temporal_features(reshape_3d(np.log(np.clip(X, 1e-3, None)), n_days, n_prot))),
            ("Combined", lambda X: combined_features(reshape_3d(np.log(np.clip(X, 1e-3, None)), n_days, n_prot))),
        ]:
            aucs_trials = []
            for trial in range(5):
                trial_rng = np.random.default_rng(42 + trial)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                fold_aucs = []

                for tr_idx, te_idx in skf.split(X_raw, y):
                    X_tr = feat_fn(X_raw[tr_idx])
                    if drift > 0:
                        X_te_raw = apply_drift(X_raw[te_idx], drift, trial_rng)
                    else:
                        X_te_raw = X_raw[te_idx].copy()
                    X_te = feat_fn(X_te_raw)

                    clf = Pipeline([("s", StandardScaler()),
                        ("c", RandomForestClassifier(200, random_state=42, class_weight='balanced'))])
                    clf.fit(X_tr, y[tr_idx])
                    try:
                        prob = clf.predict_proba(X_te)
                        fold_aucs.append(roc_auc_score(y[te_idx], prob[:, 1]))
                    except:
                        fold_aucs.append(0.5)

                aucs_trials.append(np.mean(fold_aucs))
            results[feat_name] = np.mean(aucs_trials)

        print(f"  {drift:8.1f} {results['Raw']:9.3f} {results['Spatial']:9.3f} "
              f"{results['Temp']:9.3f} {results['Combined']:9.3f}")

    # ================================================================
    # PART 3: CROSS-COHORT TRANSFER
    # ================================================================
    print()
    print("-" * 70)
    print("PART 3: CROSS-COHORT TRANSFER (leave-one-cohort-out)")
    print("-" * 70)
    print()

    cohort_list = sorted(set(cohorts))
    print(f"  {'Test Cohort':>15} {'Raw AUC':>9} {'IO Spat':>9} {'IO Temp':>9} {'IO Comb':>9}")
    print(f"  {'-'*55}")

    summary = {k: [] for k in ['Raw', 'Spatial', 'Temp', 'Combined']}

    for test_cohort in cohort_list:
        test_mask = cohorts == test_cohort
        train_mask = ~test_mask

        if np.sum(test_mask) < 5 or len(np.unique(y[test_mask])) < 2:
            continue

        row = {}
        for feat_name, feat_fn in [
            ("Raw", lambda X: X),
            ("Spatial", lambda X: spatial_features(reshape_3d(np.log(np.clip(X, 1e-3, None)), n_days, n_prot))),
            ("Temp", lambda X: temporal_features(reshape_3d(np.log(np.clip(X, 1e-3, None)), n_days, n_prot))),
            ("Combined", lambda X: combined_features(reshape_3d(np.log(np.clip(X, 1e-3, None)), n_days, n_prot))),
        ]:
            X_tr = feat_fn(X_raw[train_mask])
            X_te = feat_fn(X_raw[test_mask])
            clf = Pipeline([("s", StandardScaler()),
                ("c", RandomForestClassifier(200, random_state=42, class_weight='balanced'))])
            clf.fit(X_tr, y[train_mask])
            try:
                prob = clf.predict_proba(X_te)
                auc = roc_auc_score(y[test_mask], prob[:, 1])
            except:
                auc = accuracy_score(y[test_mask], clf.predict(X_te))
            row[feat_name] = auc
            summary[feat_name].append(auc)

        yr = test_cohort.replace('_dataset', '')
        print(f"  {yr:>15} {row['Raw']:9.3f} {row['Spatial']:9.3f} "
              f"{row['Temp']:9.3f} {row['Combined']:9.3f}")

    print(f"  {'Mean':>15}", end="")
    for k in ['Raw', 'Spatial', 'Temp', 'Combined']:
        print(f" {np.mean(summary[k]):9.3f}", end="")
    print()

    # ================================================================
    # PART 4: TEMPORAL ORDER COMPARISON
    # ================================================================
    print()
    print("-" * 70)
    print("PART 4: TEMPORAL ORDER COMPARISON (m=1 vs m=2 vs m=3)")
    print("-" * 70)
    print()
    print("  Higher order = longer-range temporal correlations")
    print("  5 time points → max order = 3")
    print()

    print(f"  {'Order':>8} {'Acc':>8} {'AUC':>8} {'Dims':>6} {'Under 2x drift':>15}")
    print(f"  {'-'*50}")

    for order in [1, 2, 3]:
        X_t = temporal_features(X_3d_log, order=order)
        acc, auc = evaluate_cv(X_t, y)

        # Under drift
        drift_aucs = []
        for trial in range(5):
            trial_rng = np.random.default_rng(42 + trial)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_aucs = []
            for tr_idx, te_idx in skf.split(X_raw, y):
                X_tr = temporal_features(X_3d_log[tr_idx], order=order)
                X_te_raw = apply_drift(X_raw[te_idx], 2.0, trial_rng)
                X_te_3d = reshape_3d(np.log(np.clip(X_te_raw, 1e-3, None)), n_days, n_prot)
                X_te = temporal_features(X_te_3d, order=order)

                clf = Pipeline([("s", StandardScaler()),
                    ("c", RandomForestClassifier(200, random_state=42, class_weight='balanced'))])
                clf.fit(X_tr, y[tr_idx])
                try:
                    prob = clf.predict_proba(X_te)
                    fold_aucs.append(roc_auc_score(y[te_idx], prob[:, 1]))
                except:
                    fold_aucs.append(0.5)
            drift_aucs.append(np.mean(fold_aucs))

        print(f"  m={order:5d} {acc:8.1%} {auc:8.3f} {X_t.shape[1]:6d} {np.mean(drift_aucs):15.3f}")

    # ================================================================
    # PART 5: BATCH EFFECT VISIBILITY
    # ================================================================
    print()
    print("-" * 70)
    print("PART 5: BATCH EFFECT — Can classifier distinguish cohorts?")
    print("-" * 70)
    print()
    print("  High AUC = batch effect visible (bad for generalization)")
    print("  Low AUC = batch effect eliminated (good)")
    print()

    # Pairwise cohort classification
    for c1, c2 in [('2021_dataset', '2022_dataset'),
                     ('2022_dataset', '2023_dataset'),
                     ('2021_dataset', '2023_dataset')]:
        mask = (cohorts == c1) | (cohorts == c2)
        X_sub = X_raw[mask]
        y_cohort = (cohorts[mask] == c2).astype(int)

        if len(np.unique(y_cohort)) < 2:
            continue

        yr1 = c1.replace('_dataset', '')
        yr2 = c2.replace('_dataset', '')
        print(f"  {yr1} vs {yr2}:")

        for feat_name, X_f in [
            ("Raw pg/mL", X_sub),
            ("Log pg/mL", np.log(np.clip(X_sub, 1e-3, None))),
            ("IO Spatial", spatial_features(reshape_3d(np.log(np.clip(X_sub, 1e-3, None)), n_days, n_prot))),
            ("IO Temporal", temporal_features(reshape_3d(np.log(np.clip(X_sub, 1e-3, None)), n_days, n_prot))),
        ]:
            _, auc = evaluate_cv(X_f, y_cohort, n_splits=min(5, min(Counter(y_cohort).values())))
            marker = " ← batch visible" if auc > 0.65 else " ← batch reduced" if auc < 0.55 else ""
            print(f"    {feat_name:>15}: AUC={auc:.3f}{marker}")
        print()

    # ================================================================
    # REPORT
    # ================================================================
    print()
    print("=" * 70)
    print("REPORT: CMI-PB Immune Vaccination × Invariant Order")
    print("=" * 70)
    print()
    print("  Domain:       6 — Immunology (scale-up)")
    print("  Data:         CMI-PB (pertussis booster)")
    print("  Subjects:     101 (52 wP, 49 aP)")
    print("  Cytokines:    14 (raw pg/mL)")
    print("  Time points:  5 (Day 0, 1, 3, 7, 14)")
    print("  Cohorts:      3 (2021, 2022, 2023)")
    print()
    print("  GEOMETRIES TESTED:")
    print("    Spatial: cross-cytokine log-ratios at each time point")
    print("    Temporal: Δᵐ(log concentration) along time, orders 1-3")
    print("    Combined: spatial + temporal on orthogonal axes")
    print()
    print("  PREDICTION SCORECARD:")
    print("    P1 (spatial classification):      [see Part 1]")
    print("    P2 (temporal drift invariance):    [see Part 2]")
    print("    P3 (cross-cohort transfer):        [see Part 3]")
    print("    P4 (combined > individual):         [see Part 1]")
    print("    P5 (higher-order temporal):          [see Part 4]")
    print()
    print("  KEY: Same algebra (log-difference), orthogonal axes.")
    print("  Each geometry annihilates its corresponding nuisance.")
    print()


if __name__ == "__main__":
    main()
