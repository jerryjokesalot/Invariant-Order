"""Observable designer: mathematical contract for invariant representations.

The Design class constructs invariant observables by specifying three axes:
  1. Nuisance — what transformation should be irrelevant
  2. Geometry — the relationship that defines which observations are comparable
  3. Structure — what structural property should be preserved

The output is a falsifiable mathematical contract: what's suppressed,
what's preserved, what becomes fundamentally unidentifiable, and why
the chosen geometry is legitimate for this data.
"""

from dataclasses import dataclass, field
import numpy as np
from .core import transform, frequency_response


NUISANCE_INFO = {
    "multiplicative": {
        "group": "Multiplicative scale",
        "space": "log",
        "suppressed_by_order": {
            1: ["constant gain (s → c·s)"],
            2: ["constant gain (s → c·s)", "linear gain drift (s → (a+bt)·s)"],
            3: ["constant gain (s → c·s)", "linear gain drift (s → (a+bt)·s)",
                "quadratic gain drift (s → (a+bt+ct²)·s)"],
        },
        "identifiability_warnings": [
            "Pure multiplicative signal changes cannot be distinguished "
            "from sensor gain drift using this observation alone.",
            "A fault that uniformly scales amplitude without changing "
            "variability structure will be suppressed.",
        ],
        "breaks_symmetry_with": [
            "Reference sensor (compare channels to isolate sensor-specific drift)",
            "Calibration history (known gain changes can be factored out)",
            "Physics model (if faults change frequency content but gain doesn't)",
            "Multiple sensors (correlated drift → sensor; uncorrelated → signal)",
        ],
    },
    "additive": {
        "group": "Additive translation",
        "space": "linear",
        "suppressed_by_order": {
            1: ["constant offset (s → s + c)"],
            2: ["constant offset (s → s + c)", "linear baseline drift (s → s + a + bt)"],
            3: ["constant offset (s → s + c)", "linear baseline drift (s → s + a + bt)",
                "quadratic baseline drift (s → s + a + bt + ct²)"],
        },
        "identifiability_warnings": [
            "Pure additive signal changes cannot be distinguished "
            "from baseline offset drift using this observation alone.",
            "A fault that uniformly shifts the signal level without changing "
            "variability structure will be suppressed.",
        ],
        "breaks_symmetry_with": [
            "Reference sensor (compare to isolate sensor-specific offset)",
            "Known baseline (subtract calibrated zero-point)",
            "Physics model (if faults change dynamics but offset doesn't)",
        ],
    },
}


GEOMETRY_INFO = {
    "temporal": {
        "name": "Temporal (ordered time series)",
        "axis": "time",
        "neighborhood": "consecutive measurements in time",
        "why_legitimate": (
            "Observations at t_n and t_{n+1} are neighbors because they are "
            "consecutive measurements of the same process. Local relationships "
            "(differences, rates of change) carry physical meaning."
        ),
        "invariant_method": "Δᵐ in working space along temporal axis",
        "examples": [
            "Sensor readings over time",
            "Heartbeat intervals (RR series)",
            "Vibration measurements",
            "Neural spike trains",
        ],
    },
    "spatial": {
        "name": "Spatial (parallel sensor array)",
        "axis": "sensors/channels",
        "neighborhood": "parallel measurements of the same physical event",
        "why_legitimate": (
            "Sensors A and B are not geometrically adjacent, but they are "
            "parallel measurements of the same latent physical event. "
            "Cross-sensor ratios cancel common multiplicative drift because "
            "the drift acts on each sensor independently while the event is shared."
        ),
        "invariant_method": "log(x_i/x_j) = log(x_i) - log(x_j) across sensors",
        "examples": [
            "Chemical sensor arrays (e-nose)",
            "Multi-electrode neural recordings",
            "Distributed sensor networks",
            "Multi-spectral imaging channels",
        ],
    },
    "spectral": {
        "name": "Spectral (ordered eigenvalue/frequency spectrum)",
        "axis": "eigenvalue index / frequency",
        "neighborhood": "adjacent eigenvalues in the ordered spectrum",
        "why_legitimate": (
            "Eigenvalues λ_n and λ_{n+1} are neighbors by spectral ordering. "
            "Their local correlations (spacings, ratios) encode the symmetry class "
            "of the underlying operator, independent of smooth spectral density."
        ),
        "invariant_method": "Spacing ratios r_n = s_n/s_{n+1} or Δᵐ(log spacing)",
        "examples": [
            "Random matrix eigenvalues",
            "Quantum energy levels",
            "Riemann zeta zeros",
            "Resonance frequencies of physical systems",
        ],
    },
}


STRUCTURE_INFO = {
    "local_change": {
        "name": "Local change detection",
        "description": "Detect points where the local statistical structure changes",
        "requires": "Stationary baseline for comparison",
    },
    "rhythm": {
        "name": "Rhythm / periodicity",
        "description": "Detect changes in periodic or quasi-periodic structure",
        "requires": "Quasi-periodic signal with meaningful intervals",
    },
    "curvature": {
        "name": "Local curvature / shape",
        "description": "Detect changes in the local shape of the signal",
        "requires": "Smooth enough signal that second-order structure exists",
    },
    "relative_magnitude": {
        "name": "Relative magnitude",
        "description": "Detect changes in the ratio between observations",
        "requires": "Multiple channels or repeated measurements",
    },
    "correlation": {
        "name": "Local correlation structure",
        "description": "Detect changes in how neighboring observations relate",
        "requires": "Meaningful neighborhood with statistical dependence",
    },
}


@dataclass
class DesignContract:
    """Falsifiable mathematical contract for an invariant observable.

    This is the prediction. It specifies what the invariant representation
    will and won't do BEFORE any implementation or data is touched.
    """

    nuisance: str
    nuisance_group: str
    geometry: str
    geometry_name: str
    geometry_why: str
    structure: str
    structure_name: str
    order: int
    working_space: str
    invariant_method: str
    suppressed: list
    preserved: list
    identifiability_warnings: list
    symmetry_breakers: list
    frequency_response_formula: str
    low_freq_attenuation: str
    predictions: list = field(default_factory=list)
    failure_modes: list = field(default_factory=list)

    # Optional: attached signal analysis
    signal_length: int = 0
    transformed_length: int = 0
    transformed: np.ndarray = field(default=None, repr=False)

    def __str__(self):
        lines = []
        lines.append("=" * 65)
        lines.append("INVARIANT OBSERVABLE DESIGN CONTRACT")
        lines.append("=" * 65)

        lines.append(f"\n  Nuisance group:      {self.nuisance_group}")
        lines.append(f"  Geometry:            {self.geometry_name}")
        lines.append(f"  Structural target:   {self.structure_name}")
        lines.append(f"  Differential order:  {self.order}")
        lines.append(f"  Working space:       {self.working_space}")
        lines.append(f"  Invariant method:    {self.invariant_method}")

        if self.signal_length:
            lines.append(f"  Signal:              {self.signal_length} → "
                         f"{self.transformed_length} (transformed)")

        lines.append(f"\n  WHY THIS GEOMETRY IS LEGITIMATE:")
        lines.append(f"    {self.geometry_why}")

        lines.append(f"\n  SUPPRESSED (by algebraic guarantee):")
        for s in self.suppressed:
            lines.append(f"    • {s}")

        lines.append(f"\n  PRESERVED:")
        for p in self.preserved:
            lines.append(f"    • {p}")

        lines.append(f"\n  FREQUENCY RESPONSE: {self.frequency_response_formula}")
        lines.append(f"  LOW-FREQUENCY:      {self.low_freq_attenuation}")

        lines.append(f"\n  ⚠  IDENTIFIABILITY BOUNDARY:")
        for w in self.identifiability_warnings:
            lines.append(f"    {w}")

        lines.append(f"\n  SYMMETRY BREAKERS:")
        for b in self.symmetry_breakers:
            lines.append(f"    → {b}")

        if self.predictions:
            lines.append(f"\n  FALSIFIABLE PREDICTIONS:")
            for i, p in enumerate(self.predictions, 1):
                lines.append(f"    {i}. {p}")

        if self.failure_modes:
            lines.append(f"\n  PREDICTED FAILURE MODES:")
            for f in self.failure_modes:
                lines.append(f"    ✗ {f}")

        lines.append("")
        return "\n".join(lines)


# Keep backward-compatible alias
DesignReport = DesignContract


class Design:
    """Design an invariant observable with explicit mathematical guarantees.

    Constructs observables by specifying three axes:
      - nuisance: what transformation should be irrelevant
      - geometry: the neighborhood structure of the data
      - structure: what structural property to preserve

    The output is a falsifiable contract: predictions about what will
    and won't work, BEFORE any implementation runs.

    Parameters
    ----------
    nuisance : str
        Transformation group to suppress: "multiplicative" or "additive".
    order : int
        Differential order. Higher order suppresses more polynomial drift.
    geometry : str, optional
        Data geometry: "temporal", "spatial", or "spectral".
        Default "temporal" for backward compatibility.
    structure : str, optional
        Structural target: "local_change", "rhythm", "curvature",
        "relative_magnitude", or "correlation".
        Default "local_change".
    """

    def __init__(self, nuisance="multiplicative", order=1,
                 geometry="temporal", structure="local_change"):
        if nuisance not in NUISANCE_INFO:
            raise ValueError(f"nuisance must be one of {list(NUISANCE_INFO)}, "
                             f"got '{nuisance}'")
        if order < 1:
            raise ValueError("order must be >= 1")
        if geometry not in GEOMETRY_INFO:
            raise ValueError(f"geometry must be one of {list(GEOMETRY_INFO)}, "
                             f"got '{geometry}'")
        if structure not in STRUCTURE_INFO:
            raise ValueError(f"structure must be one of {list(STRUCTURE_INFO)}, "
                             f"got '{structure}'")

        self.nuisance = nuisance
        self.order = order
        self.geometry = geometry
        self.structure = structure
        self._ninfo = NUISANCE_INFO[nuisance]
        self._ginfo = GEOMETRY_INFO[geometry]
        self._sinfo = STRUCTURE_INFO[structure]

    def analyze(self, signal=None):
        """Produce the mathematical contract (optionally with signal analysis).

        Returns a DesignContract: the falsifiable specification of what
        this representation guarantees, suppresses, and cannot resolve.
        """
        suppressed = self._ninfo["suppressed_by_order"].get(
            self.order,
            self._ninfo["suppressed_by_order"][max(self._ninfo["suppressed_by_order"])]
            + [f"polynomial drift up to degree {self.order - 1}"]
        )

        preserved = self._get_preserved()
        predictions = self._get_predictions()
        failure_modes = self._get_failure_modes()

        freq_formula = f"|2 sin(ω/2)|^{self.order}"
        low_freq = f"O(ω^{self.order})"

        transformed = None
        sig_len = 0
        trans_len = 0

        if signal is not None and self.geometry == "temporal":
            signal = np.asarray(signal, dtype=float)
            sig_len = len(signal)
            transformed = transform(signal, self.nuisance, self.order)
            trans_len = len(transformed)

        return DesignContract(
            nuisance=self.nuisance,
            nuisance_group=self._ninfo["group"],
            geometry=self.geometry,
            geometry_name=self._ginfo["name"],
            geometry_why=self._ginfo["why_legitimate"],
            structure=self.structure,
            structure_name=self._sinfo["name"],
            order=self.order,
            working_space=self._ninfo["space"],
            invariant_method=self._ginfo["invariant_method"],
            suppressed=suppressed,
            preserved=preserved,
            identifiability_warnings=self._ninfo["identifiability_warnings"],
            symmetry_breakers=self._ninfo["breaks_symmetry_with"],
            frequency_response_formula=freq_formula,
            low_freq_attenuation=low_freq,
            predictions=predictions,
            failure_modes=failure_modes,
            signal_length=sig_len,
            transformed_length=trans_len,
            transformed=transformed,
        )

    def _get_preserved(self):
        base = []
        if self.geometry == "temporal":
            base = [
                "Variability structure (beat-to-beat irregularity)",
                "Temporal dynamics (pattern changes, transients)",
                "Higher-order statistics of log-differences",
            ]
        elif self.geometry == "spatial":
            base = [
                "Cross-sensor relative response pattern",
                "Sensor array 'fingerprint' for each event class",
                "Ratios between parallel measurements",
            ]
        elif self.geometry == "spectral":
            base = [
                "Local eigenvalue correlations (spacing statistics)",
                "Spectral rigidity and repulsion structure",
                "Universality class signatures",
            ]

        base.append(f"Frequency content above O(ω^{self.order}) suppression")
        return base

    def _get_predictions(self):
        preds = []

        if self.geometry == "temporal" and self.nuisance == "multiplicative":
            preds.append(
                "Invariant features will maintain classification/detection accuracy "
                "under arbitrary multiplicative scaling of the signal."
            )
            preds.append(
                f"Polynomial gain drift of degree < {self.order} will produce "
                f"zero response in the transformed signal (exact annihilation)."
            )
            if self.structure == "rhythm":
                preds.append(
                    "Beat-to-beat variability (e.g., HRV) will be preserved "
                    "because it lives in the local difference structure."
                )
                preds.append(
                    "Baseline heart rate changes (slow scaling) will be suppressed, "
                    "isolating rhythm irregularity from rate."
                )

        elif self.geometry == "temporal" and self.nuisance == "additive":
            preds.append(
                "Invariant features will maintain accuracy under "
                "arbitrary additive baseline shifts."
            )
            preds.append(
                f"Polynomial baseline drift of degree < {self.order} will be "
                f"exactly annihilated."
            )

        elif self.geometry == "spatial":
            preds.append(
                "Cross-sensor log-ratios will cancel common multiplicative drift, "
                "preserving the relative response pattern."
            )
            preds.append(
                "Classification accuracy will degrade less over time compared "
                "to raw features, especially on heavily drifted batches."
            )
            preds.append(
                "Sensors with different drift rates will partially break "
                "the invariance (residual drift in ratios)."
            )

        elif self.geometry == "spectral":
            preds.append(
                "Spacing ratios will be invariant to smooth rescaling of "
                "the spectral density (unfolding)."
            )
            preds.append(
                "Local spectral statistics will distinguish universality classes "
                "(GOE vs GUE vs Poisson) independent of global density."
            )
            preds.append(
                "The gap ratio r_n = s_n/s_{n+1} is the order-1 case; "
                "higher orders probe longer-range spectral correlations."
            )

        return preds

    def _get_failure_modes(self):
        modes = []

        modes.append(
            f"[B] Geometry failure: data lacks meaningful {self._ginfo['axis']} "
            f"ordering — finite differences become noise"
        )
        modes.append(
            f"[C] Nuisance-model failure: real nuisance is not "
            f"{self.nuisance} (e.g., nonlinear, non-stationary)"
        )
        modes.append(
            "[D] Information-theoretic blindness: the signal of interest "
            "belongs to the same transformation class as the nuisance"
        )

        if self.geometry == "temporal" and self.structure == "rhythm":
            modes.append(
                "[D] Amplitude changes in rhythm (e.g., ectopic beats that "
                "only change amplitude, not timing) will be suppressed"
            )

        if self.geometry == "spatial":
            modes.append(
                "[C] Non-uniform drift: if sensors drift at very different "
                "rates, ratios retain residual drift"
            )

        if self.geometry == "spectral":
            modes.append(
                "[C] Non-smooth spectral density: if the unfolding function "
                "has discontinuities, spacing ratios become unreliable"
            )

        return modes

    def frequency_response(self, omegas=None):
        """Get the frequency response curve for this design."""
        return frequency_response(self.order, omegas)

    def compare_orders(self, max_order=4, omegas=None):
        """Compare frequency responses across multiple orders."""
        if omegas is None:
            omegas = np.linspace(0.01, np.pi, 200)

        results = {}
        for m in range(1, max_order + 1):
            _, mags = frequency_response(m, omegas)
            results[m] = mags
        return omegas, results
