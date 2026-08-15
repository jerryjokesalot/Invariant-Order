"""Observable designer: mathematical contract for invariant representations.

The Design class doesn't just transform data — it tells you what the
transformation guarantees, what it suppresses, and what becomes
fundamentally unidentifiable as a consequence.
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


@dataclass
class DesignReport:
    """Mathematical contract for an invariant representation."""

    nuisance: str
    group: str
    order: int
    space: str
    suppressed: list
    preserved: list
    frequency_response_formula: str
    low_freq_attenuation: str
    identifiability_warnings: list
    symmetry_breakers: list
    signal_length: int = 0
    transformed_length: int = 0
    transformed: np.ndarray = field(default=None, repr=False)

    def __str__(self):
        lines = []
        lines.append("=" * 60)
        lines.append("INVARIANT REPRESENTATION DESIGN")
        lines.append("=" * 60)

        lines.append(f"\n  Nuisance group:    {self.group}")
        lines.append(f"  Differential order: {self.order}")
        lines.append(f"  Working space:      {self.space}")

        if self.signal_length:
            lines.append(f"  Signal length:      {self.signal_length} → "
                         f"{self.transformed_length} (transformed)")

        lines.append(f"\n  Suppressed (by design):")
        for s in self.suppressed:
            lines.append(f"    • {s}")

        lines.append(f"\n  Preserved:")
        for p in self.preserved:
            lines.append(f"    • {p}")

        lines.append(f"\n  Frequency response: {self.frequency_response_formula}")
        lines.append(f"  Low-frequency:      {self.low_freq_attenuation}")

        lines.append(f"\n  ⚠  Identifiability boundary:")
        for w in self.identifiability_warnings:
            lines.append(f"    {w}")

        lines.append(f"\n  Symmetry breakers (additional info that resolves ambiguity):")
        for b in self.symmetry_breakers:
            lines.append(f"    → {b}")

        lines.append("")
        return "\n".join(lines)


class Design:
    """Design an invariant representation with explicit guarantees.

    This is the central object of the framework. It doesn't just transform
    data — it produces a mathematical contract specifying what the
    representation suppresses, what it preserves, and what becomes
    fundamentally unidentifiable.

    Parameters
    ----------
    nuisance : str
        Transformation group to suppress: "multiplicative" or "additive".
    order : int
        Differential order. Higher order suppresses more polynomial drift.
    """

    def __init__(self, nuisance="multiplicative", order=1):
        if nuisance not in NUISANCE_INFO:
            raise ValueError(f"nuisance must be 'multiplicative' or 'additive', "
                             f"got '{nuisance}'")
        if order < 1:
            raise ValueError("order must be >= 1")

        self.nuisance = nuisance
        self.order = order
        self._info = NUISANCE_INFO[nuisance]

    def analyze(self, signal=None):
        """Analyze a signal (or produce the contract without data).

        Returns a DesignReport: the mathematical specification of what
        this representation guarantees, suppresses, and cannot resolve.
        """
        suppressed = self._info["suppressed_by_order"].get(
            self.order,
            self._info["suppressed_by_order"][max(self._info["suppressed_by_order"])]
            + [f"polynomial drift up to degree {self.order - 1}"]
        )

        preserved = self._get_preserved()

        freq_formula = f"|2 sin(ω/2)|^{self.order}"
        low_freq = f"O(ω^{self.order})"

        transformed = None
        sig_len = 0
        trans_len = 0

        if signal is not None:
            signal = np.asarray(signal, dtype=float)
            sig_len = len(signal)
            transformed = transform(signal, self.nuisance, self.order)
            trans_len = len(transformed)

        return DesignReport(
            nuisance=self.nuisance,
            group=self._info["group"],
            order=self.order,
            space=self._info["space"],
            suppressed=suppressed,
            preserved=preserved,
            frequency_response_formula=freq_formula,
            low_freq_attenuation=low_freq,
            identifiability_warnings=self._info["identifiability_warnings"],
            symmetry_breakers=self._info["breaks_symmetry_with"],
            signal_length=sig_len,
            transformed_length=trans_len,
            transformed=transformed,
        )

    def _get_preserved(self):
        if self.nuisance == "multiplicative":
            return [
                "Variability structure (beat-to-beat irregularity)",
                "Temporal dynamics (pattern changes, transients)",
                "Higher-order statistics of log-differences",
                f"Frequency content above O(ω^{self.order}) suppression",
            ]
        else:
            return [
                "Variability structure (sample-to-sample irregularity)",
                "Temporal dynamics (pattern changes, transients)",
                "Higher-order statistics of finite differences",
                f"Frequency content above O(ω^{self.order}) suppression",
            ]

    def frequency_response(self, omegas=None):
        """Get the frequency response curve for this design."""
        return frequency_response(self.order, omegas)

    def compare_orders(self, max_order=4, omegas=None):
        """Compare frequency responses across multiple orders.

        Useful for choosing the right order: higher order suppresses
        more drift but also attenuates more low-frequency signal.
        """
        if omegas is None:
            omegas = np.linspace(0.01, np.pi, 200)

        results = {}
        for m in range(1, max_order + 1):
            _, mags = frequency_response(m, omegas)
            results[m] = mags
        return omegas, results
