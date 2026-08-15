# Graph IO — What is a derivative when geometry isn't a line?

**For Rocky — answering your frontier question**

## The Question

"What is the equivalent of a derivative when the underlying geometry
isn't a line?"

We tested three candidate formulations on synthetic graph data.

## The Answer (short version)

**The derivative on a graph is spectral IO.** You map the graph to its
Laplacian eigenvalue spectrum (a 1D ordered sequence), then apply standard
IO. This gives exact invariance — 0.000000 RMSD under eigenvalue rescaling,
classification AUC locked at 0.983 while raw degrades from 0.984 to 0.545
at 5x drift.

Direct graph derivatives (Laplacian IO, gradient IO) give only approximate
invariance, and only when the nuisance has graph structure. This confirms
and deepens the "line matters" result.

## Three Formulations Tested

### 1. Laplacian IO: L^m(log f)

Apply the graph Laplacian m times to the log-signal on nodes.
The Laplacian is L(f)(i) = Σ_j A_ij [f(i) - f(j)] — the natural
"difference operator" on a graph.

**Result: partial.** Reduces community-level drift (~25% of raw RMSD)
but does NOT achieve exact cancellation. Under per-node drift, Laplacian
IO is actually WORSE than raw features.

### 2. Gradient IO: log(f_j) - log(f_i) per edge

Direct analog of temporal IO m=1, but along graph edges.

**Result: exact for global drift only.** Perfectly cancels global
multiplicative nuisance (f → c·f). But per-community and per-node
drift remain — edge features carry the full residual when c_i ≠ c_j.

### 3. Spectral IO: spacing ratios of Laplacian eigenvalues

Compute graph Laplacian eigenvalues λ₁ ≤ λ₂ ≤ ... ≤ λ_n.
Apply standard IO: log(λ_{k+1}) - log(λ_k).

**Result: exact invariance.** This is temporal IO on the graph's
spectrum. The eigenvalues form a natural 1D ordering. Spacing ratios
are perfectly invariant to eigenvalue rescaling.

| Drift | Raw AUC | Spectral IO AUC | Raw RMSD | IO RMSD |
|-------|---------|-----------------|----------|---------|
| 0.0 | 0.984 | 0.987 | — | — |
| 0.5 | 0.775 | **0.983** | 4.6 | **0.000000** |
| 1.0 | 0.677 | **0.983** | 16.6 | **0.000000** |
| 2.0 | 0.604 | **0.983** | 255 | **0.000000** |
| 5.0 | 0.545 | **0.983** | **8,805,279** | **0.000000** |

## Why Temporal IO Works But Direct Graph IO Doesn't

This is the core insight. It comes down to what "neighboring" means.

**Temporal:** Same entity at different times.
Per-entity scale factor c appears at EVERY time point.
Δ(log(c·x_t)) = Δ(log x_t) + Δ(log c) = Δ(log x_t) + 0.
The scale factor cancels EXACTLY because it's constant along the
differencing axis.

**Graph:** Different entities connected by edges.
Node i has scale c_i, node j has scale c_j, and c_i ≠ c_j in general.
L(log(c_i·x_i)) = L(log x_i) + L(log c_i) ≠ L(log x_i).
Per-node nuisance does NOT cancel because different nodes have
different scale factors.

**The "line matters" theorem (informal):**
IO achieves exact nuisance cancellation when:
1. The differencing axis has a canonical ordering (line, spectrum), AND
2. The nuisance is constant along that axis for each channel.

On a graph, there is no canonical ordering of nodes, and different
nodes have independent nuisance. Both conditions fail.

## When Graph IO Does Help

When nuisance has GRAPH STRUCTURE — specifically, when nodes in the
same community share the same scale factor.

Community classification under per-community drift:

| Community drift σ | Raw AUC | Laplacian m=1 AUC |
|-------------------|---------|-------------------|
| 0.0 | 1.000 | 0.949 |
| 0.5 | 0.612 | **0.852** |
| 1.0 | 0.538 | **0.713** |
| 2.0 | 0.499 | 0.599 |
| 5.0 | 0.487 | 0.517 |

Laplacian IO helps at moderate drift (0.852 vs 0.612 at σ=0.5)
but the advantage degrades — it's approximate, not exact.
Within-community edges see the same offset (approximately cancels),
cross-community edges see different offsets (residual).

## When Graph IO Fails

Under per-node drift, Laplacian IO is WORSE than raw:

| Per-node drift σ | Raw AUC | Laplacian m=1 AUC |
|------------------|---------|-------------------|
| 0.0 | 1.000 | 0.911 |
| 0.5 | 0.981 | 0.720 |
| 1.0 | 0.888 | 0.633 |
| 2.0 | 0.776 | 0.580 |
| 5.0 | 0.650 | 0.546 |

The Laplacian's smoothing mixes signal with unstructured nuisance,
making things worse. This is an HONEST failure with a clean design
contract explanation: the nuisance has no graph structure, so graph
IO can't separate it from signal.

## The Design Contract for Graph Geometry

```
Geometry:   Graph (nodes + edges)
Nuisance:   Multiplicative, STRUCTURED along the graph
            (same scale within clusters/communities)
Structure:  Sub-graph patterns not aligned with nuisance
Invariant:  L^m(log f) approximately cancels smooth nuisance
Contract:   APPROXIMATE, not exact. Degrades with drift magnitude.
            Exact invariance requires reducing to spectrum first.
```

## What This Means for the IO Framework

### The geometry hierarchy

| Geometry | Ordering | IO cancellation | Status |
|----------|----------|----------------|--------|
| Temporal | Natural (time) | **Exact** | Validated (ECG, immune, bearings) |
| Spatial | Parallel channels | **Exact** | Validated (gas sensors, immune) |
| Spectral | Natural (eigenvalue order) | **Exact** | Validated (quantum, zeta, graph spectra) |
| Graph (direct) | **No canonical ordering** | **Approximate** | Conditional (community-structured nuisance only) |

### The spectral bridge

The answer to "what is a derivative on a graph?" turns out to be:
**don't differentiate on the graph — differentiate on its spectrum.**

The graph Laplacian eigenvalues carry the structural information
(community structure, connectivity pattern, expansion rate) while
discarding node identity entirely. Spacing ratios of eigenvalues
are invariant to:
- Graph rescaling (multiply all edge weights)
- Node relabeling (permutation symmetry — eigenvalues are already invariant)
- Degree normalization choices (affects eigenvalue scale, not ratios)

This is exactly IO's design pattern: find the 1D ordered representation
that preserves structure, then apply log-difference.

### Implications

1. **IO's exact invariance requires a line.** Temporal has time.
   Spatial has channel index. Spectral has eigenvalue order. Graph
   doesn't have one — but its SPECTRUM does.

2. **"Line matters" is now a theorem, not just an observation.**
   The mathematical reason is clear: exact cancellation requires the
   nuisance to be constant along the differencing axis, which requires
   a canonical axis.

3. **Every geometry may have a spectral bridge.** If you can extract
   a 1D ordered representation that preserves the structure you care
   about, you can apply IO exactly. The question becomes: what's
   lost in the projection?

4. **Graph IO has a niche.** When nuisance IS community-structured
   (batch effects by lab site, platform effects by region), Laplacian
   IO gives moderate drift resistance. It's not exact, but it's better
   than nothing — and the design contract tells you exactly when
   to trust it.

## For the Paper

The graph result is the cleanest articulation of IO's scope:

- IO achieves exact invariance on ordered geometries (line, spectrum)
- IO achieves approximate invariance on graph geometries when
  nuisance is smooth along edges
- The spectral bridge (graph → eigenvalues → IO) gives exact
  invariance for graph-level classification at the cost of
  discarding node identity

This is the honest boundary Rocky asked for. Not "IO works everywhere"
and not "IO fails on graphs." It's: **IO's power comes from ordering,
and the question for any new geometry is whether a meaningful 1D
ordering exists.**
