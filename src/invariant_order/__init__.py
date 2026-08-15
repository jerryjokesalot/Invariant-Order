"""Invariant Order: detect structural change, ignore nuisance drift.

Quick start:
    import invariant_order as io

    # Batch: scan a signal for change points
    result = io.scan(signal, nuisance='multiplicative', order=1)

    # Compare two segments
    result = io.compare(before, after, nuisance='multiplicative', order=1)

    # Stream: real-time detection
    stream = io.StreamDetector(nuisance='multiplicative', order=1)
    for value in data:
        alert = stream.push(value)
"""

from .core import (
    transform,
    coefficient_moments,
    invariant_order,
    frequency_response,
)
from .detector import Detector, ScanResult, ChangePoint
from .stream import StreamDetector, Alert
from .confidence import compare, ComparisonResult


def scan(signal, nuisance="multiplicative", order=1, **kwargs):
    """Scan a signal for structural change points.

    Convenience wrapper around Detector.scan().
    """
    return Detector(nuisance=nuisance, order=order, **kwargs).scan(signal)


__version__ = "0.1.0"

__all__ = [
    "transform",
    "scan",
    "compare",
    "coefficient_moments",
    "invariant_order",
    "frequency_response",
    "Detector",
    "StreamDetector",
    "ScanResult",
    "ChangePoint",
    "ComparisonResult",
    "Alert",
]
