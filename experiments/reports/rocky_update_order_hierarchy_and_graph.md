# Rocky Update: Order Hierarchy + Graph IO + Research Framework

Two new experiments and a reframing of the entire project.

## 1. IO Order Hierarchy (CMI-PB immune data)

Your question: "Does IO order form a meaningful hierarchy of biological
dynamics?"

We ran 6 sub-experiments on the same CMI-PB data (101 subjects, 14
cytokines, 5 time points). Results:

### Different biological questions have different optimal orders

| Question | Optimal Order | AUC | Why it makes sense |
|----------|--------------|-----|-------------------|
| What state is the system in? (peak level) | m=0 | 0.987 | Static measurement |
| How fast is it changing? (early/late) | m=1 | 0.992 | Rate of change |
| How is that change itself changing? (wP vs aP) | m=2 | 0.665 | Acceleration pattern |
| Sustained vs transient response? | m=1 | 0.978 | Rate of decay |

The optimal order is not a parameter to tune. It's a coordinate that
tells you what kind of dynamical question you're asking.

### Different cytokines carry the wP/aP signal at different orders

All 5 orders are represented across the 14 cytokines:

| Order | Cytokines | Interpretation |
|-------|-----------|----------------|
| m=0 (state) | IFN-γ, IL-7 | Static markers |
| m=1 (velocity) | IP-10, IL-6, MCP-1, IL-10, IL-2 | Rate signals |
| m=2 (acceleration) | IFN-α2, MIP-1α | Curvature signals |
| m=3 (jerk) | G-CSF, IL-1RA | Change-of-acceleration |
| m=4 (snap) | TNF-α, IL-8, RANTES | Higher-order transients |

Each molecule has a "characteristic dynamical order" for this
biological question. You aren't saying "the immune system is an m=2
system." You're finding that different components of the same system
participate in different orders of dynamics.

### The immune system traverses the order hierarchy over time

After vaccination, the dominant IO order changes:

| Window | Dominant order | What's happening |
|--------|---------------|-----------------|
| Day 0→1 | m=1 | Immediate velocity response |
| Day 0→3 | m=2 | Acceleration pattern emerges |
| Day 0→7 | m=3 | Higher-order complexity (adaptive immunity engaging) |
| Day 7→14 | m=1 | Settling back to simple velocity |

The system literally climbs and descends the order hierarchy. That's a
dynamical phase portrait readable purely from which IO order is most
active at each time window.

### Optimal order is stable across cohorts

m=2 is optimal for wP vs aP in 3 of 4 cohort groupings (All data,
2021, 2023). Only the 2022 cohort (n=18, smallest) picks m=1. The
optimal order appears to be a property of the biology, not the
measurement batch.

### wP–aP separation is larger at m=2 than m=1 at every time step

The biggest separation is at the D1→D3→D7 acceleration window
(|Δ| = 0.208 at m=2 vs 0.122 at m=1). The immune acceleration peaks
exactly when the adaptive response is engaging.

---

## 2. Graph IO: What is a derivative when geometry isn't a line?

Your question: "What is the equivalent of a derivative when the
underlying geometry isn't a line?"

We tested three formulations on synthetic graph data:

### Three candidates

**Laplacian IO: L^m(log f)**
Iterated graph Laplacian on log-signals.
Result: approximate invariance only. Helps when nuisance is
community-structured. Worse than raw under per-node nuisance.

**Gradient IO: log(f_j) - log(f_i) per edge**
Direct analog of temporal IO along edges.
Result: exact for global drift only. Useless for structured drift.

**Spectral IO: spacing ratios of Laplacian eigenvalues**
Graph → eigenvalue spectrum → standard IO on ordered sequence.
Result: **exact invariance.**

### Spectral IO results

| Drift | Raw AUC | Spectral IO AUC | Raw RMSD | IO RMSD |
|-------|---------|-----------------|----------|---------|
| 0.0 | 0.984 | 0.987 | — | — |
| 0.5 | 0.775 | **0.983** | 4.6 | **0.000000** |
| 1.0 | 0.677 | **0.983** | 16.6 | **0.000000** |
| 2.0 | 0.604 | **0.983** | 255 | **0.000000** |
| 5.0 | 0.545 | **0.983** | **8,805,279** | **0.000000** |

Graph type classification (community vs random) is perfectly stable
under eigenvalue rescaling. IO RMSD is literally zero.

### Why temporal IO works but direct graph IO doesn't

**Temporal:** Same entity at different times. Per-entity scale c
appears at EVERY time step. Δ(log(c·x_t)) = Δ(log x_t). Cancels exactly.

**Graph:** Different entities connected by edges. Node i has scale c_i,
node j has c_j, c_i ≠ c_j. Per-node nuisance does not cancel.

**The "line matters" theorem:** IO achieves exact cancellation when
(1) the differencing axis has a canonical ordering, and (2) nuisance
is constant along that axis for each channel. Graphs fail both
conditions — but their spectra satisfy both.

### The spectral bridge

The answer to "what is a derivative on a graph?" is:
**don't differentiate on the graph — differentiate on its spectrum.**

Graph → Laplacian eigenvalues → IO on ordered eigenvalue sequence.

This gives exact invariance because eigenvalues are naturally ordered
on a line. The graph's structural information (community structure,
connectivity, expansion) is preserved in the eigenvalue pattern.

### The design contract for graph geometry

```
Geometry:   Graph (nodes + edges)
Nuisance:   Eigenvalue rescaling (graph-level)
Structure:  Eigenvalue spacing pattern (graph type, community structure)
Invariant:  Spectral IO: Δᵐ(log λ_k)
Boundary:   Node-level features lost in spectral projection.
            Per-node nuisance requires community-structured assumption.
```

### Laplacian IO has a niche

When nuisance IS community-structured (same scale within clusters),
Laplacian IO gives moderate resistance:

| Community drift σ | Raw AUC | Laplacian m=1 |
|-------------------|---------|---------------|
| 0.0 | 1.000 | 0.949 |
| 0.5 | 0.612 | **0.852** |
| 1.0 | 0.538 | **0.713** |
| 2.0 | 0.499 | 0.599 |

Approximate, not exact — but better than nothing when the nuisance
has graph structure.

---

## 3. The Geometry Matrix

We formalized the organizing framework. Every IO experiment maps into:

| Geometry | "Neighboring" | IO cancellation | Domains validated |
|----------|--------------|----------------|-------------------|
| Temporal | earlier/later | **Exact** | ECG, bearings, immune |
| Spatial | parallel channels | **Exact** | Gas sensors, immune |
| Spectral | adjacent eigenvalues | **Exact** | Quantum, zeta, graph spectra |
| Graph (direct) | connected nodes | **Approximate** | Community-structured nuisance only |

Cross-cutting result: IO order is a coordinate for dynamical complexity.
The optimal order depends on what kind of change you're asking about,
not on the dataset.

---

## 4. Paper Pitch (reframed)

Old framing (retire): "Drift-resistant ML preprocessing."

New framing: **IO is a measurement framework. Given a geometry, a
declared nuisance symmetry, and a target signal, it constructs
observables whose differential order determines what kind of dynamical
phenomenon they measure.**

The paper leads with the immune experiment (most intuitive), validates
across 8 domains and 3+ geometries, and uses graph IO as the honest
boundary that reveals the scope of exact invariance.

The one-line version for the abstract: "We present Invariant Order, a
measurement framework that constructs observables whose differential
order determines sensitivity to specific kinds of dynamical change
while provably annihilating declared measurement nuisance."

---

## 5. Proposed Next Step: Blind Challenge

You suggested the next experiment should test whether someone who
wasn't involved in developing IO can use the Design framework to
correctly predict observables and their behavior on unseen data.

We agree. Here's the proposal:

### The challenge

We give you:
- 3 datasets from domains IO has not been tested on
- The IO Design protocol (7 questions)
- The SDK (pip install invariant-order)

You produce, for each dataset, BEFORE running any experiments:
- Geometry identification
- Nuisance declaration
- Structure specification
- Candidate invariant
- Predicted optimal order
- Predicted successes (what should work)
- Predicted failure modes (what should break)

Then you (or we) run the experiments and compare your contracts to
reality.

### Candidate datasets (to be finalized)

1. **EEG motor imagery** — temporal + spatial. Multi-electrode
   recordings, subject-specific amplitude scaling, classification of
   imagined left vs right hand movement.

2. **Mass spectrometry proteomics** — spectral geometry. Instrument
   calibration drift across runs, protein identification from peak
   patterns.

3. **Acoustic machine monitoring** — temporal + spectral. Microphone
   gain variation, ambient noise, machine fault detection from sound.

### Why this matters

Every IO result so far was produced by people who designed the
framework. The blind challenge tests whether the Design protocol is
a transferable methodology — whether the framework produces correct
predictions in the hands of someone who knows the concepts but didn't
build the tool.

If the contracts match reality, IO isn't a collection of techniques.
It's a general methodology for constructing observables.

That's the claim worth making.
