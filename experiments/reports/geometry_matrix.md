# IO Geometry × Nuisance × Structure Matrix

The organizing framework for IO as a measurement system.

## The Principle

Given a geometry, a symmetry (nuisance), and a desired structural signal,
construct observables whose differential order determines exactly what kinds
of change they are sensitive to — and what kinds of change they are
mathematically incapable of seeing.

IO order is not a parameter to tune. It is a coordinate that tells you
what kind of dynamical question you are asking.

## The Matrix

### Temporal Geometry

"Neighboring" = earlier/later in time.
IO operation = Δᵐ(log x) along time axis.

| Domain | Dataset | n | Nuisance | Structure | Best Order | Key Result |
|--------|---------|---|----------|-----------|-----------|------------|
| Cardiology | MIT-BIH ECG | 109k beats | Baseline wander, electrode drift | Beat morphology dynamics | m=1 | AUC 0.97, drift-invariant |
| Mechanical | CWRU Bearings | 4 conditions | Sensor gain, mounting variation | Fault progression | m=1 | Perfect classification under 5x drift |
| Immunology | CMI-PB | 101 subjects | Assay gain, subject baseline | Vaccination response dynamics | **m=2** | AUC 0.665; curvature > velocity |
| Immunology | FluPRINT | 92 donors | Inter-study calibration | Response fold-change | m=1 | 3M:1 drift stability ratio |

**What temporal IO measures:** How a signal changes over ordered time.
m=0 = state, m=1 = velocity, m=2 = acceleration, m=3 = jerk.

**What temporal IO cancels:** Any multiplicative factor constant across time
for a given channel. Per-protein assay gain cancels exactly because the same
scale factor appears at every time point.

**Design contract boundary:** Non-multiplicative batch effects (different
instruments measuring different things) are outside the annihilation guarantee.

### Spatial Geometry

"Neighboring" = parallel measurements of the same event.
IO operation = log(x_i) - log(x_j) across channels.

| Domain | Dataset | n | Nuisance | Structure | Key Result |
|--------|---------|---|----------|-----------|------------|
| Chemical | UCI Gas Sensor | 13,910 | Sensor drift over 36 months | Gas identity from 16-sensor array | +14.2% accuracy, 59.8% AUC gain |
| Immunology | FluPRINT | 43 donors | Assay gain, subject baseline | Relative cytokine profile | AUC 0.638 vs 0.300; cross-study 0.901 |
| Immunology | CMI-PB | 101 subjects | Per-sample dilution | Static cytokine ratios | Modest (+0.010); signal is temporal here |

**What spatial IO measures:** Relative relationships between parallel channels.
"How does sensor A compare to sensor B?" rather than "What is sensor A reading?"

**What spatial IO cancels:** Any common multiplicative factor across channels
within a single measurement instance (sample dilution, overall gain).

**Design contract boundary:** Per-channel independent drift leaves residuals
in adjacent ratios.

### Spectral Geometry

"Neighboring" = adjacent eigenvalues/energy levels.
IO operation = log(s_{n+1}) - log(s_n) across the spectrum.

| Domain | Dataset | n | Nuisance | Structure | Key Result |
|--------|---------|---|----------|-----------|------------|
| Quantum | Hydrogen/Helium/Lithium | 3 atoms | Energy scale, unit system | Level spacing pattern | Perfect atom identification; 10⁸:1 stability |
| Number Theory | Riemann zeta zeros | 10k zeros | Height-dependent density | Local spacing statistics | GUE statistics recovered; connected to random matrix theory |

**What spectral IO measures:** The pattern of spacings between ordered values.
Not where the levels are, but how they are distributed relative to each other.

**What spectral IO cancels:** Any global rescaling of the spectrum.
Changing energy units (eV → Hartree) doesn't change spacing ratios.

**Design contract boundary:** Non-uniform spectral density shifts
(levels spreading out at one end but not the other) can distort ratios.

### Graph Geometry — THE FRONTIER

"Neighboring" = connected nodes.
IO operation = ???

| Domain | Dataset | n | Status |
|--------|---------|---|--------|
| Synthetic graphs | Erdős–Rényi, scale-free | varied | **Negative result** — RTD fails on graphs |

**The open question:** What is the equivalent of a derivative when the
underlying geometry isn't a line?

On a line (temporal), neighboring means "the next time step" and the
derivative is unambiguous. On a graph, a node has multiple neighbors
and there is no canonical ordering.

**Candidates for graph IO:**
- Laplacian eigenvectors as the "spectral basis" → differences in the
  spectral domain (but this reduces to spectral geometry)
- Message-passing aggregation → f(node) - mean(f(neighbors)) as a
  graph derivative (but what's the multiplicative nuisance?)
- Heat kernel diffusion → IO at different diffusion times (t plays the
  role of differential order)

**Why this matters:** If IO extends to graphs, it becomes a measurement
framework for network-structured data (molecular graphs, social networks,
neural connectivity, transportation). If it doesn't, the "line matters"
result becomes a theorem about when IO is applicable.

### Manifold Geometry — SPECULATIVE

"Neighboring" = nearby states in a continuous manifold.
IO operation = directional derivatives + gauge invariance?

No experiments yet. The question is whether IO's algebra extends to
curved spaces where "neighboring" depends on position. This connects
to gauge theory: multiplicative nuisance on a manifold is literally
a gauge transformation.

## Cross-Cutting Results

### IO Order as Biological Spectrometer (CMI-PB)

The immune experiment revealed that IO order is not a single number
for a system. Different questions about the same system have different
optimal orders:

| Question | Optimal Order | AUC | Interpretation |
|----------|--------------|-----|----------------|
| What state is the system in? (peak level) | m=0 | 0.987 | Static measurement |
| How fast is it changing? (early/late response) | m=1 | 0.992 | Rate of change |
| How is that change itself changing? (wP vs aP) | m=2 | 0.665 | Acceleration pattern |

Different cytokines carry the wP/aP signal at different orders:
- m=0: IFN-γ, IL-7 (state variables)
- m=1: IP-10, IL-6, MCP-1, IL-10, IL-2 (rate variables)
- m=2: IFN-α2, MIP-1α (acceleration variables)
- m=3: G-CSF, IL-1RA (jerk variables)
- m=4: TNF-α, IL-8, RANTES (higher-order transients)

### Regime Detection

The immune system traverses the IO order hierarchy after vaccination:
- Day 0→1: m=1 dominates (immediate velocity response)
- Day 0→3: m=2 dominates (acceleration pattern emerges)
- Day 0→7+: m=3 dominates (higher-order complexity)
- Day 7→14: back to m=1 (settling)

This suggests IO order can detect dynamical regime transitions —
the system literally climbs and descends the order hierarchy over time.

### Validated Negative Results

1. **Graph ecology fails** — RTD on synthetic graphs produces no
   discriminative signal. Confirms "the line matters."
2. **Spatial IO on temporal problems** — CMI-PB spatial IO adds noise
   to temporal classification (combined 0.597 < temporal 0.629).
   IO correctly identifies which geometry carries the signal.
3. **m=3 with 5 time points** — Third-order differences on 5 points
   yield only 2 values per protein. Overfits. Need denser sampling
   for higher orders.

## Research Program

### Populated cells (strong evidence)

- Temporal × multiplicative drift × rate/curvature dynamics
- Spatial × common-mode gain × relative channel profile
- Spectral × energy rescaling × spacing distribution

### Priority targets

1. **Graph geometry** — define what "IO derivative" means on a graph
2. **Spatial transcriptomics** — spatial geometry on gene expression
   (thousands of genes, physical location as spatial coordinate)
3. **Neural recordings** — temporal IO on multi-electrode arrays
   (temporal + spatial simultaneously, like immunology)
4. **Financial time series** — temporal IO on returns (already log-differences;
   the question is whether higher-order IO reveals regime changes)
5. **Climate** — temporal IO on multi-station temperature/precipitation
   (decades of data, station-specific calibration drift)

### For each new cell

1. What is the geometry?
2. What transformation is nuisance?
3. What structure are we trying to preserve?
4. What invariant should therefore exist?
5. What differential order should expose it?
6. What does the experiment predict?
7. What actually happens?
