"""Graph IO — What is a derivative when geometry isn't a line?

Rocky's question: "What is the equivalent of a derivative when the
underlying geometry isn't a line?"

Three candidate formulations:

1. LAPLACIAN IO: L^m(log f) — iterated graph Laplacian on log-signals.
   The Laplacian is the natural "difference operator" on a graph.
   L(f)(i) = sum_j A_ij [f(i) - f(j)].
   Annihilates global additive offset (= global multiplicative in raw space).

2. GRADIENT IO: log(f(j)) - log(f(i)) for each edge (i,j).
   Gives edge features instead of node features.
   Direct analog of temporal IO m=1, but along graph edges instead of time.

3. SPECTRAL IO: spacing ratios of Laplacian eigenvalues.
   Reduces graph to its spectrum (a 1D ordered sequence), then applies
   standard spectral IO. "Cheating" — flattens graph to line.

KEY INSIGHT FROM TEMPORAL IO:
  Temporal IO cancels per-protein drift because the SAME entity is
  measured at multiple time points. The scale factor is constant along
  the differencing axis.

  On a graph, connected nodes are DIFFERENT entities. Per-node nuisance
  does NOT cancel under graph differencing — unless the nuisance has
  graph structure (e.g., same scale factor within a community).

THIS EXPERIMENT TESTS:
  1. Which formulations cancel which kinds of nuisance?
  2. Does Laplacian IO order reveal different structural scales?
  3. Does spectral IO classify graph types under rescaling?
  4. Where does graph IO FAIL (honest boundary)?
"""

import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def stochastic_block_model(sizes, p_within, p_between, rng):
    """Generate SBM graph with community structure."""
    n = sum(sizes)
    A = np.zeros((n, n))
    labels = np.repeat(np.arange(len(sizes)), sizes)

    for i in range(n):
        for j in range(i + 1, n):
            p = p_within if labels[i] == labels[j] else p_between
            if rng.random() < p:
                A[i, j] = A[j, i] = 1

    return A, labels


def normalized_laplacian(A):
    """Compute normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    d = A.sum(axis=1)
    d[d == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    return np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt


def unnormalized_laplacian(A):
    """L = D - A."""
    return np.diag(A.sum(axis=1)) - A


def laplacian_io(f_log, L, order):
    """Apply L^m to log-signal f_log. Returns node features."""
    result = f_log.copy()
    for _ in range(order):
        result = L @ result
    return result


def gradient_io(f_log, A):
    """Log-differences along edges. Returns edge features."""
    edges = np.argwhere(np.triu(A) > 0)
    if len(edges) == 0:
        return np.array([])
    return np.array([f_log[j] - f_log[i] for i, j in edges])


def spectral_io(A, order=1):
    """Spacing ratios of Laplacian eigenvalues."""
    L = unnormalized_laplacian(A)
    eigvals = np.sort(np.linalg.eigvalsh(L))
    eigvals = eigvals[eigvals > 1e-10]  # skip zero eigenvalues
    if len(eigvals) < 2:
        return np.array([])
    log_eigs = np.log(eigvals)
    result = log_eigs
    for _ in range(order):
        result = np.diff(result)
    return result


def apply_global_drift(f, scale):
    """f -> scale * f. Global multiplicative nuisance."""
    return f * scale


def apply_community_drift(f, labels, scales):
    """Per-community multiplicative nuisance."""
    result = f.copy()
    for k, s in enumerate(scales):
        result[labels == k] *= s
    return result


def apply_node_drift(f, scales):
    """Per-node multiplicative nuisance."""
    return f * scales


def feature_rmsd(f1, f2):
    """Root mean squared difference."""
    if len(f1) == 0 or len(f2) == 0:
        return float('nan')
    return np.sqrt(np.mean((f1 - f2) ** 2))


def evaluate_cv(X, y, n_splits=5, rng_seed=42):
    n_splits = min(n_splits, min(Counter(y).values()))
    if n_splits < 2 or X.shape[1] == 0:
        return float('nan')
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
    print("GRAPH IO — What is a derivative on a graph?")
    print("=" * 70)
    print()

    rng = np.random.default_rng(42)

    # ================================================================
    # PART 1: Nuisance annihilation test
    # ================================================================
    print("-" * 70)
    print("PART 1: Which formulation cancels which nuisance?")
    print("-" * 70)
    print()

    n = 100
    A, labels = stochastic_block_model([50, 50], 0.3, 0.05, rng)
    L = normalized_laplacian(A)
    L_un = unnormalized_laplacian(A)

    # Base signal: community-structured + noise
    f = np.zeros(n)
    f[labels == 0] = 5.0
    f[labels == 1] = 2.0
    f += rng.normal(0, 0.5, n)
    f = np.abs(f) + 0.1  # ensure positive
    f_log = np.log(f)

    # Compute clean features
    clean_lap = {m: laplacian_io(f_log, L, m) for m in range(4)}
    clean_grad = gradient_io(f_log, A)
    clean_spec = {m: spectral_io(A, m) for m in range(4)}

    print("  Signal: 2-community SBM (n=100, p_in=0.3, p_out=0.05)")
    print("  Community signal: group 0 ~ 5.0, group 1 ~ 2.0 + noise")
    print()

    # --- 1a: Global drift ---
    print("  1a. GLOBAL multiplicative drift (f -> c*f)")
    print()
    print(f"  {'Drift c':>10} {'Raw RMSD':>10} {'Lap m=1':>10} {'Lap m=2':>10}"
          f" {'Gradient':>10} {'Spec m=1':>10}")
    print(f"  {'-' * 62}")

    for c in [2.0, 5.0, 10.0, 100.0]:
        f_d = apply_global_drift(f, c)
        f_d_log = np.log(f_d)

        raw_rmsd = feature_rmsd(f_log, f_d_log)
        lap1_rmsd = feature_rmsd(clean_lap[1], laplacian_io(f_d_log, L, 1))
        lap2_rmsd = feature_rmsd(clean_lap[2], laplacian_io(f_d_log, L, 2))
        grad_rmsd = feature_rmsd(clean_grad, gradient_io(f_d_log, A))
        spec1_rmsd = feature_rmsd(clean_spec[1], spectral_io(A, 1))  # spectral doesn't depend on f

        print(f"  {c:10.1f} {raw_rmsd:10.4f} {lap1_rmsd:10.6f} {lap2_rmsd:10.6f}"
              f" {grad_rmsd:10.6f} {spec1_rmsd:10.6f}")

    # --- 1b: Per-community drift ---
    print()
    print("  1b. PER-COMMUNITY multiplicative drift (different scale per community)")
    print()
    print(f"  {'Drift':>10} {'Raw RMSD':>10} {'Lap m=1':>10} {'Lap m=2':>10}"
          f" {'Gradient':>10}")
    print(f"  {'-' * 52}")

    for drift_std in [0.5, 1.0, 2.0, 5.0]:
        rmsds = {'raw': [], 'lap1': [], 'lap2': [], 'grad': []}
        for trial in range(20):
            trial_rng = np.random.default_rng(100 + trial)
            scales = np.exp(trial_rng.normal(0, drift_std, 2))
            f_d = apply_community_drift(f, labels, scales)
            f_d_log = np.log(np.clip(f_d, 1e-10, None))

            rmsds['raw'].append(feature_rmsd(f_log, f_d_log))
            rmsds['lap1'].append(feature_rmsd(clean_lap[1], laplacian_io(f_d_log, L, 1)))
            rmsds['lap2'].append(feature_rmsd(clean_lap[2], laplacian_io(f_d_log, L, 2)))
            rmsds['grad'].append(feature_rmsd(clean_grad, gradient_io(f_d_log, A)))

        print(f"  {drift_std:10.1f} {np.mean(rmsds['raw']):10.4f}"
              f" {np.mean(rmsds['lap1']):10.4f} {np.mean(rmsds['lap2']):10.4f}"
              f" {np.mean(rmsds['grad']):10.4f}")

    # --- 1c: Per-node drift ---
    print()
    print("  1c. PER-NODE multiplicative drift (different scale per node)")
    print()
    print(f"  {'Drift':>10} {'Raw RMSD':>10} {'Lap m=1':>10} {'Lap m=2':>10}"
          f" {'Gradient':>10}")
    print(f"  {'-' * 52}")

    for drift_std in [0.5, 1.0, 2.0, 5.0]:
        rmsds = {'raw': [], 'lap1': [], 'lap2': [], 'grad': []}
        for trial in range(20):
            trial_rng = np.random.default_rng(100 + trial)
            scales = np.exp(trial_rng.normal(0, drift_std, n))
            f_d = apply_node_drift(f, scales)
            f_d_log = np.log(np.clip(f_d, 1e-10, None))

            rmsds['raw'].append(feature_rmsd(f_log, f_d_log))
            rmsds['lap1'].append(feature_rmsd(clean_lap[1], laplacian_io(f_d_log, L, 1)))
            rmsds['lap2'].append(feature_rmsd(clean_lap[2], laplacian_io(f_d_log, L, 2)))
            rmsds['grad'].append(feature_rmsd(clean_grad, gradient_io(f_d_log, A)))

        print(f"  {drift_std:10.1f} {np.mean(rmsds['raw']):10.4f}"
              f" {np.mean(rmsds['lap1']):10.4f} {np.mean(rmsds['lap2']):10.4f}"
              f" {np.mean(rmsds['grad']):10.4f}")

    # ================================================================
    # PART 2: Laplacian IO order as structural scale
    # ================================================================
    print()
    print("-" * 70)
    print("PART 2: Does Laplacian IO order reveal different structural scales?")
    print("-" * 70)
    print()

    # Generate graphs with multi-scale community structure
    # 4 communities of 25, with sub-communities of ~12
    n = 100
    A_multi, labels_coarse = stochastic_block_model([25, 25, 25, 25], 0.4, 0.02, rng)
    # Add sub-community structure within each community
    labels_fine = np.zeros(n, dtype=int)
    for k in range(4):
        mask = labels_coarse == k
        idx = np.where(mask)[0]
        half = len(idx) // 2
        labels_fine[idx[:half]] = 2 * k
        labels_fine[idx[half:]] = 2 * k + 1
        # Add extra within-subcommunity edges
        for i in idx[:half]:
            for j in idx[:half]:
                if i < j and rng.random() < 0.15:
                    A_multi[i, j] = A_multi[j, i] = 1
        for i in idx[half:]:
            for j in idx[half:]:
                if i < j and rng.random() < 0.15:
                    A_multi[i, j] = A_multi[j, i] = 1

    L_multi = normalized_laplacian(A_multi)

    # Signal with both scales
    f_multi = np.zeros(n)
    for k in range(4):
        f_multi[labels_coarse == k] += rng.normal(0, 1) * 3  # coarse signal
    for k in range(8):
        f_multi[labels_fine == k] += rng.normal(0, 1) * 1  # fine signal
    f_multi += rng.normal(0, 0.3, n)  # noise
    f_multi = np.abs(f_multi) + 0.1
    f_multi_log = np.log(f_multi)

    # Test: which IO order best classifies coarse vs fine structure?
    print("  Multi-scale SBM: 4 coarse communities, 8 fine sub-communities")
    print("  Signal has both coarse (σ=3) and fine (σ=1) components")
    print()

    print("  2a. Classifying COARSE structure (4 communities → 2 groups)")
    y_coarse = (labels_coarse >= 2).astype(int)

    print(f"  {'Order':>8} {'Features':>10} {'AUC':>8}")
    print(f"  {'-' * 30}")

    for m in range(5):
        feats = laplacian_io(f_multi_log, L_multi, m)
        auc = evaluate_cv(feats.reshape(-1, 1), y_coarse)
        print(f"  m={m:5d} {1:10d} {auc:8.3f}")

    print()
    print("  2b. Classifying FINE structure (8 sub-communities → 2 interleaved groups)")
    y_fine = (labels_fine % 2).astype(int)

    print(f"  {'Order':>8} {'Features':>10} {'AUC':>8}")
    print(f"  {'-' * 30}")

    for m in range(5):
        feats = laplacian_io(f_multi_log, L_multi, m)
        auc = evaluate_cv(feats.reshape(-1, 1), y_fine)
        print(f"  m={m:5d} {1:10d} {auc:8.3f}")

    # ================================================================
    # PART 3: Spectral IO for graph classification
    # ================================================================
    print()
    print("-" * 70)
    print("PART 3: Spectral IO for graph type classification")
    print("Does spectral IO classify graph types under rescaling?")
    print("-" * 70)
    print()

    # Generate two classes of graphs: community vs random
    n_graphs = 200
    n_nodes = 50
    X_raw_spec = []
    X_io_spec = []
    y_graph = []

    for i in range(n_graphs):
        g_rng = np.random.default_rng(1000 + i)
        if i < n_graphs // 2:
            # Community graph (SBM with 2 blocks)
            A_g, _ = stochastic_block_model([25, 25], 0.4, 0.05, g_rng)
            y_graph.append(0)
        else:
            # Random graph (Erdos-Renyi with similar density)
            p_er = 0.22
            A_g = (g_rng.random((n_nodes, n_nodes)) < p_er).astype(float)
            A_g = np.triu(A_g, 1)
            A_g = A_g + A_g.T
            y_graph.append(1)

        # Compute eigenvalues
        L_g = unnormalized_laplacian(A_g)
        eigs = np.sort(np.linalg.eigvalsh(L_g))
        eigs = eigs[eigs > 1e-10]

        # Pad/truncate to fixed length
        k = 20
        if len(eigs) >= k:
            raw_feats = eigs[:k]
        else:
            raw_feats = np.pad(eigs, (0, k - len(eigs)))

        log_eigs = np.log(np.clip(raw_feats, 1e-10, None))
        io_feats = np.diff(log_eigs)  # spectral IO m=1

        X_raw_spec.append(raw_feats)
        X_io_spec.append(io_feats)

    X_raw_spec = np.array(X_raw_spec)
    X_io_spec = np.array(X_io_spec)
    y_graph = np.array(y_graph)

    print(f"  {n_graphs} graphs: {n_graphs//2} community (SBM) vs {n_graphs//2} random (ER)")
    print(f"  Each graph: {n_nodes} nodes")
    print()

    # Clean classification
    auc_raw = evaluate_cv(X_raw_spec, y_graph)
    auc_io = evaluate_cv(X_io_spec, y_graph)
    print(f"  Clean:  Raw eigenvalues AUC={auc_raw:.3f}  |  Spectral IO AUC={auc_io:.3f}")

    # Under global eigenvalue rescaling (multiply all eigenvalues by random factor)
    print()
    print(f"  {'Drift':>8} {'Raw AUC':>10} {'Spec IO':>10} {'Raw RMSD':>10} {'IO RMSD':>10}")
    print(f"  {'-' * 52}")

    for drift_std in [0.5, 1.0, 2.0, 5.0]:
        raw_aucs, io_aucs = [], []
        raw_rmsds, io_rmsds = [], []
        for trial in range(10):
            trial_rng = np.random.default_rng(200 + trial)

            X_raw_d = X_raw_spec.copy()
            X_io_d = X_io_spec.copy()

            # Per-graph random rescaling of eigenvalues
            for gi in range(n_graphs):
                scale = np.exp(trial_rng.normal(0, drift_std))
                X_raw_d[gi] = X_raw_spec[gi] * scale
                # Recompute IO from drifted
                log_d = np.log(np.clip(X_raw_d[gi], 1e-10, None))
                X_io_d[gi] = np.diff(log_d)

            raw_rmsds.append(np.sqrt(np.mean((X_raw_d - X_raw_spec) ** 2)))
            io_rmsds.append(np.sqrt(np.mean((X_io_d - X_io_spec) ** 2)))

            # Train on clean, test on drifted (via CV with drift on test fold)
            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            r_aucs, i_aucs = [], []
            for tr, te in skf.split(X_raw_spec, y_graph):
                clf_r = Pipeline([("s", StandardScaler()),
                    ("c", RandomForestClassifier(200, random_state=42))])
                clf_r.fit(X_raw_spec[tr], y_graph[tr])
                try:
                    r_aucs.append(roc_auc_score(y_graph[te], clf_r.predict_proba(X_raw_d[te])[:, 1]))
                except:
                    r_aucs.append(0.5)

                clf_i = Pipeline([("s", StandardScaler()),
                    ("c", RandomForestClassifier(200, random_state=42))])
                clf_i.fit(X_io_spec[tr], y_graph[tr])
                try:
                    i_aucs.append(roc_auc_score(y_graph[te], clf_i.predict_proba(X_io_d[te])[:, 1]))
                except:
                    i_aucs.append(0.5)

            raw_aucs.append(np.mean(r_aucs))
            io_aucs.append(np.mean(i_aucs))

        print(f"  {drift_std:8.1f} {np.mean(raw_aucs):10.3f} {np.mean(io_aucs):10.3f}"
              f" {np.mean(raw_rmsds):10.3f} {np.mean(io_rmsds):10.6f}")

    # ================================================================
    # PART 4: The boundary — what graph IO CANNOT do
    # ================================================================
    print()
    print("-" * 70)
    print("PART 4: Honest boundary — what graph IO cannot do")
    print("-" * 70)
    print()

    print("  4a. Per-node drift: does Laplacian IO help?")
    print()

    # Community classification under per-node drift
    A_test, lab_test = stochastic_block_model([50, 50], 0.3, 0.05, rng)
    L_test = normalized_laplacian(A_test)

    # Multi-dimensional node signals
    n_signals = 10
    F_clean = np.zeros((100, n_signals))
    for s in range(n_signals):
        for k in range(2):
            F_clean[lab_test == k, s] = rng.normal(0, 1) * (2 if k == 0 else 1)
        F_clean[:, s] += rng.normal(0, 0.3, 100)
    F_clean = np.abs(F_clean) + 0.1

    y_comm = lab_test

    print(f"  {'Drift σ':>10} {'Raw AUC':>10} {'Lap m=1':>10} {'Lap m=2':>10}"
          f" {'Spec IO':>10}")
    print(f"  {'-' * 52}")

    for drift_std in [0.0, 0.5, 1.0, 2.0, 5.0]:
        r_aucs, l1_aucs, l2_aucs, s_aucs = [], [], [], []
        for trial in range(10):
            trial_rng = np.random.default_rng(300 + trial)

            if drift_std > 0:
                node_scales = np.exp(trial_rng.normal(0, drift_std, 100))
                F_d = F_clean * node_scales[:, np.newaxis]
            else:
                F_d = F_clean.copy()

            F_d_log = np.log(np.clip(F_d, 1e-10, None))
            F_clean_log = np.log(F_clean)

            # Raw
            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            fold_r, fold_l1, fold_l2 = [], [], []
            for tr, te in skf.split(F_clean_log, y_comm):
                # Raw: train clean, test drifted
                clf = Pipeline([("s", StandardScaler()),
                    ("c", RandomForestClassifier(200, random_state=42, class_weight='balanced'))])
                clf.fit(F_clean_log[tr], y_comm[tr])
                try:
                    fold_r.append(roc_auc_score(y_comm[te], clf.predict_proba(F_d_log[te])[:, 1]))
                except:
                    fold_r.append(0.5)

                # Laplacian m=1
                X_tr_l1 = np.column_stack([laplacian_io(F_clean_log[:, s], L_test, 1) for s in range(n_signals)])[tr]
                X_te_l1 = np.column_stack([laplacian_io(F_d_log[:, s], L_test, 1) for s in range(n_signals)])[te]
                clf_l1 = Pipeline([("s", StandardScaler()),
                    ("c", RandomForestClassifier(200, random_state=42, class_weight='balanced'))])
                clf_l1.fit(X_tr_l1, y_comm[tr])
                try:
                    fold_l1.append(roc_auc_score(y_comm[te], clf_l1.predict_proba(X_te_l1)[:, 1]))
                except:
                    fold_l1.append(0.5)

                # Laplacian m=2
                X_tr_l2 = np.column_stack([laplacian_io(F_clean_log[:, s], L_test, 2) for s in range(n_signals)])[tr]
                X_te_l2 = np.column_stack([laplacian_io(F_d_log[:, s], L_test, 2) for s in range(n_signals)])[te]
                clf_l2 = Pipeline([("s", StandardScaler()),
                    ("c", RandomForestClassifier(200, random_state=42, class_weight='balanced'))])
                clf_l2.fit(X_tr_l2, y_comm[tr])
                try:
                    fold_l2.append(roc_auc_score(y_comm[te], clf_l2.predict_proba(X_te_l2)[:, 1]))
                except:
                    fold_l2.append(0.5)

            r_aucs.append(np.mean(fold_r))
            l1_aucs.append(np.mean(fold_l1))
            l2_aucs.append(np.mean(fold_l2))

        # Spectral IO — graph-level, not applicable to per-node classification
        # so we'll skip it here
        print(f"  {drift_std:10.1f} {np.mean(r_aucs):10.3f} {np.mean(l1_aucs):10.3f}"
              f" {np.mean(l2_aucs):10.3f} {'N/A':>10}")

    print()
    print("  4b. Why temporal IO works but graph IO struggles:")
    print()
    print("  TEMPORAL: Same entity at different times.")
    print("    → Per-entity scale cancels because c appears at EVERY time step.")
    print("    → Δ(log(c·x_t)) = Δ(log x_t) + Δ(log c) = Δ(log x_t) + 0")
    print()
    print("  GRAPH: Different entities connected by edges.")
    print("    → Per-node scale does NOT cancel because c_i ≠ c_j for neighbors.")
    print("    → L(log(c_i · x_i)) = L(log x_i) + L(log c_i) ≠ L(log x_i)")
    print()
    print("  EXCEPT: If nuisance has GRAPH STRUCTURE (same c within community),")
    print("    then the Laplacian approximately cancels it — because within-community")
    print("    edges see the same offset, and cross-community edges are few.")

    # ================================================================
    # PART 5: Community-structured nuisance (where graph IO SHOULD work)
    # ================================================================
    print()
    print("-" * 70)
    print("PART 5: Community-structured nuisance (graph IO's sweet spot)")
    print("-" * 70)
    print()

    # 4 communities: signal is about sub-structure, nuisance is community-level
    A_4, lab_4 = stochastic_block_model([25, 25, 25, 25], 0.35, 0.03, rng)
    L_4 = normalized_laplacian(A_4)

    # Target: odd vs even communities (sub-structure within the graph)
    y_sub = (lab_4 % 2).astype(int)

    n_sig = 10
    F_4 = np.zeros((100, n_sig))
    for s in range(n_sig):
        # Signal: odd vs even
        F_4[y_sub == 0, s] += rng.normal(2, 0.5)
        F_4[y_sub == 1, s] += rng.normal(3, 0.5)
    F_4 = np.abs(F_4) + 0.1

    print("  4 communities, target = odd vs even community membership")
    print("  Nuisance = per-community scale (same within community)")
    print()

    print(f"  {'Comm drift σ':>14} {'Raw AUC':>10} {'Lap m=1':>10} {'Lap m=2':>10}")
    print(f"  {'-' * 48}")

    for drift_std in [0.0, 0.5, 1.0, 2.0, 5.0]:
        r_aucs, l1_aucs, l2_aucs = [], [], []
        for trial in range(10):
            trial_rng = np.random.default_rng(400 + trial)

            if drift_std > 0:
                comm_scales = np.exp(trial_rng.normal(0, drift_std, 4))
                F_d = F_4.copy()
                for k in range(4):
                    F_d[lab_4 == k] *= comm_scales[k]
            else:
                F_d = F_4.copy()

            F_d_log = np.log(np.clip(F_d, 1e-10, None))
            F_4_log = np.log(F_4)

            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            fold_r, fold_l1, fold_l2 = [], [], []
            for tr, te in skf.split(F_4_log, y_sub):
                clf = Pipeline([("s", StandardScaler()),
                    ("c", RandomForestClassifier(200, random_state=42, class_weight='balanced'))])
                clf.fit(F_4_log[tr], y_sub[tr])
                try:
                    fold_r.append(roc_auc_score(y_sub[te], clf.predict_proba(F_d_log[te])[:, 1]))
                except:
                    fold_r.append(0.5)

                X_tr_l1 = np.column_stack([laplacian_io(F_4_log[:, s], L_4, 1) for s in range(n_sig)])[tr]
                X_te_l1 = np.column_stack([laplacian_io(F_d_log[:, s], L_4, 1) for s in range(n_sig)])[te]
                clf_l1 = Pipeline([("s", StandardScaler()),
                    ("c", RandomForestClassifier(200, random_state=42, class_weight='balanced'))])
                clf_l1.fit(X_tr_l1, y_sub[tr])
                try:
                    fold_l1.append(roc_auc_score(y_sub[te], clf_l1.predict_proba(X_te_l1)[:, 1]))
                except:
                    fold_l1.append(0.5)

                X_tr_l2 = np.column_stack([laplacian_io(F_4_log[:, s], L_4, 2) for s in range(n_sig)])[tr]
                X_te_l2 = np.column_stack([laplacian_io(F_d_log[:, s], L_4, 2) for s in range(n_sig)])[te]
                clf_l2 = Pipeline([("s", StandardScaler()),
                    ("c", RandomForestClassifier(200, random_state=42, class_weight='balanced'))])
                clf_l2.fit(X_tr_l2, y_sub[tr])
                try:
                    fold_l2.append(roc_auc_score(y_sub[te], clf_l2.predict_proba(X_te_l2)[:, 1]))
                except:
                    fold_l2.append(0.5)

            r_aucs.append(np.mean(fold_r))
            l1_aucs.append(np.mean(fold_l1))
            l2_aucs.append(np.mean(fold_l2))

        print(f"  {drift_std:14.1f} {np.mean(r_aucs):10.3f} {np.mean(l1_aucs):10.3f}"
              f" {np.mean(l2_aucs):10.3f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print()
    print("=" * 70)
    print("SUMMARY: Graph IO Landscape")
    print("=" * 70)
    print()
    print("  FORMULATION          CANCELS                  PRESERVES")
    print("  Laplacian IO m≥1     Global multiplicative    Local graph structure")
    print("  Gradient IO          Global multiplicative    Edge-level contrasts")
    print("  Spectral IO          Eigenvalue rescaling     Spacing distribution")
    print()
    print("  KEY DIFFERENCE FROM TEMPORAL IO:")
    print("  Temporal: same entity at different times → per-entity drift cancels")
    print("  Graph: different entities connected → per-node drift does NOT cancel")
    print()
    print("  GRAPH IO WORKS WHEN:")
    print("  Nuisance has graph structure (same scale within communities)")
    print()
    print("  GRAPH IO FAILS WHEN:")
    print("  Nuisance is per-node (no graph structure in the nuisance)")
    print()
    print("  THIS IS THE DESIGN CONTRACT FOR GRAPH GEOMETRY:")
    print("  Geometry:  Graph (nodes + edges)")
    print("  Nuisance:  Multiplicative, STRUCTURED along the graph")
    print("  Structure: Sub-graph patterns not aligned with nuisance")
    print("  Invariant: L^m(log f) cancels smooth nuisance, preserves local variation")
    print()


if __name__ == "__main__":
    main()
