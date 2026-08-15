# Paper Pitch: Invariant Order as a Measurement Framework for Dynamical Systems

## The Old Framing (retire this)

"We built a preprocessing technique that removes measurement drift
from scientific data using log-differences."

This is true but small. It positions IO as a tool. Reviewers will ask
"why not just use batch normalization?" and they'd be right to.

## The New Framing

**Given a measurement geometry, a declared nuisance symmetry, and a target
signal, Invariant Order constructs observables whose differential order
determines what kind of dynamical phenomenon they are sensitive to — and
what kind they are mathematically blind to.**

IO order is a coordinate system for describing change.

## Why This Is Different

Most measurement frameworks ask: "What features are important?"

IO asks: "At what order of change does the signal live?"

That's a fundamentally different question. It doesn't select variables —
it selects the *kind of dynamics* you're measuring.

## The Killer Example

You want to know if childhood vaccination history affects adult immune
response to a booster shot.

**m=0 (state):** Look at cytokine levels after vaccination.
Answer: no signal (AUC 0.516). Absolute levels don't distinguish groups.

**m=1 (velocity):** Look at how fast cytokines change.
Answer: moderate signal (AUC 0.629). Rate of change carries information.

**m=2 (acceleration):** Look at how the rate of change is itself changing.
Answer: strongest signal (AUC 0.665). The *curvature* of the immune
response trajectory is what distinguishes vaccination histories.

This isn't parameter tuning. You're discovering that the biological
phenomenon — the imprint of childhood vaccination on adult immunity —
manifests as a difference in *immune response dynamics*, specifically
in how the system accelerates and decelerates after boosting.

And then: different cytokines carry this signal at different orders.
IFN-γ is a state variable (m=0). IL-6 is a rate variable (m=1).
MIP-1α is an acceleration variable (m=2). G-CSF is a jerk variable (m=3).

IO doesn't just tell you which molecules matter. It tells you *what kind
of dynamics* each molecule participates in.

## The Theorem (what makes it rigorous)

For any measurement where nuisance acts as a multiplicative transformation
constant along the measurement axis:

- The IO transform of order m exactly annihilates that nuisance
- The residual features preserve all structural variation of order ≥ m
- This is mathematically exact, not approximate

Verified empirically:
- 287,000,000:1 stability ratio (immune cytokines under assay drift)
- 10⁸:1 stability ratio (quantum energy levels under unit rescaling)
- 3,000,000:1 stability ratio (temporal immune dynamics under batch effects)
- Classification AUC unchanged at 5× multiplicative drift (multiple domains)

## The Geometry Dimension

IO isn't one technique. It's a family indexed by geometry:

| Geometry | "Neighboring" means | IO derivative | Domains tested |
|----------|-------------------|---------------|----------------|
| Temporal | earlier/later | Δᵐ(log x) along time | ECG, bearings, immune |
| Spatial | parallel sensors | log(x_i) - log(x_j) across channels | gas sensors, immune |
| Spectral | adjacent eigenvalues | spacing ratios | quantum spectra, zeta zeros |
| Graph | connected nodes | ??? (open problem) | negative result validates boundary |

Same algebra. Different axes. Each geometry annihilates nuisance along
its own axis and preserves structure along all others.

The immune experiment is the first domain where two geometries (spatial
and temporal) operate simultaneously on the same data — and IO correctly
predicts which nuisance each geometry cancels.

## The Order Hierarchy Result

IO order forms a meaningful hierarchy of dynamical complexity:

1. **Different biological questions have different optimal orders.**
   Peak magnitude → m=0. Response speed → m=1. Vaccination history → m=2.
   The optimal order is a property of the phenomenon, not the method.

2. **Different variables in the same system live at different orders.**
   14 cytokines distribute across all 5 orders (m=0 through m=4).
   Each molecule's "characteristic order" describes what kind of dynamics
   it participates in.

3. **Systems traverse the order hierarchy over time.**
   After vaccination, the dominant IO order climbs from m=1 (immediate
   velocity) to m=2 (acceleration, day 1-3) to m=3 (higher-order
   complexity, day 3-7) then descends back to m=1 (settling, day 7-14).
   The order hierarchy reads as a dynamical phase portrait.

4. **The optimal order is stable across independent cohorts.**
   m=2 is optimal for wP vs aP in 3 of 4 cohort groupings.
   It's a property of the biology, not the measurement batch.

## Paper Structure (proposed)

1. **The principle.** Geometry × nuisance × structure → invariant.
   IO order as a coordinate for dynamical complexity. Theorem statement.

2. **Three geometries.** Temporal, spatial, spectral. Same algebra,
   different axes. Each geometry annihilates its corresponding nuisance.

3. **The order hierarchy.** IO order isn't a parameter — it's a
   measurement of what kind of change you're observing. Immune
   experiment as the primary illustration.

4. **Cross-domain validation.** 8 domains, 6 geometries tested.
   Prediction scorecard across all experiments. Honest negatives
   (graph geometry, spatial IO on temporal problems, m=3 with sparse
   time points).

5. **The boundary.** What IO can't do: non-multiplicative nuisance,
   graph-structured data (open problem), higher orders with sparse
   sampling. The design contract framework as honest scope declaration.

6. **The frontier.** Graph geometry as the open mathematical question.
   What is a derivative when the underlying space isn't a line?

## What This Is NOT

- Not "a better normalization method" (too small)
- Not "a universal theory of everything" (too big)
- Not "drift-resistant ML preprocessing" (correct but misleading)

## What This IS

A measurement framework that:
1. Takes a declared geometry and nuisance symmetry as input
2. Constructs observables that are provably invariant to the nuisance
3. Uses differential order as a coordinate for dynamical complexity
4. Has been validated across temporal, spatial, and spectral geometries
5. Has an honest, falsifiable boundary (design contracts)

The contribution is the framework itself — and the empirical evidence
that differential order is a meaningful coordinate for describing
what kind of change a system is undergoing.

## One-Line Versions

**For mathematicians:** "A family of invariant observables indexed by
measurement geometry and differential order, with exact annihilation
guarantees under declared symmetry groups."

**For scientists:** "A way to automatically ask 'what kind of change is
happening?' while ignoring measurement artifacts you've declared irrelevant."

**For engineers:** "Drift-proof features whose differential order tells you
whether you're measuring state, velocity, acceleration, or higher dynamics."

**For the abstract:** "We present Invariant Order, a measurement framework
that constructs observables whose differential order determines sensitivity
to specific kinds of dynamical change while provably annihilating declared
measurement nuisance. Validated across 8 domains spanning temporal, spatial,
and spectral geometries, IO order functions as a coordinate system for
dynamical complexity — different phenomena in the same system live at
different orders, and systems traverse the order hierarchy over time."
