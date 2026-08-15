"""Core computation: finite differences, moment constraints, frequency response."""

from math import comb
import numpy as np


def transform(signal, nuisance="multiplicative", order=1):
    """Apply invariant-order transformation to a signal.

    Args:
        signal: 1D array of positive values (for multiplicative) or any reals (for additive).
        nuisance: 'multiplicative' (work in log space) or 'additive' (work in linear space).
        order: Differential order m. Higher order = blind to more polynomial drift.

    Returns:
        Transformed signal of length len(signal) - order.
    """
    signal = np.asarray(signal, dtype=float)
    if order < 1:
        raise ValueError("order must be >= 1")
    if len(signal) <= order:
        raise ValueError(f"signal length {len(signal)} must exceed order {order}")

    if nuisance == "multiplicative":
        if np.any(signal <= 0):
            raise ValueError("multiplicative nuisance requires positive signal values")
        return _finite_diff(np.log(signal), order)
    elif nuisance == "additive":
        return _finite_diff(signal, order)
    else:
        raise ValueError(f"nuisance must be 'multiplicative' or 'additive', got '{nuisance}'")


def _finite_diff(seq, m):
    """m-th finite difference using binomial coefficients."""
    n = len(seq) - m
    coeffs = _diff_coeffs(m)
    result = np.zeros(n)
    for j in range(m + 1):
        result += coeffs[j] * seq[j:j + n]
    return result


def _diff_coeffs(m):
    """Binomial coefficients for m-th finite difference: (-1)^(m-j) * C(m,j)."""
    return np.array([(-1) ** (m - j) * comb(m, j) for j in range(m + 1)], dtype=float)


def coefficient_moments(exponents, max_p=None):
    """Compute coefficient moments mu_p = sum(a_j * j^p) for an exponent vector.

    Args:
        exponents: Array of exponents a_0, a_1, ..., a_k.
        max_p: Maximum moment order to compute (default: len(exponents)).

    Returns:
        Array of moments [mu_0, mu_1, ..., mu_{max_p}].
    """
    a = np.asarray(exponents, dtype=float)
    k = len(a)
    if max_p is None:
        max_p = k
    j = np.arange(k, dtype=float)
    return np.array([np.sum(a * j ** p) for p in range(max_p + 1)])


def invariant_order(exponents):
    """Determine the invariant differential order of an exponent vector.

    Returns the smallest p >= 1 such that mu_p != 0, or None if all moments vanish.
    """
    moments = coefficient_moments(exponents)
    for p in range(1, len(moments)):
        if abs(moments[p]) > 1e-10:
            return p
    return None


def frequency_response(order, omegas=None):
    """Compute the theoretical frequency response |H(omega)| for a given order.

    For the m-th difference stencil: |H(omega)| = [2*sin(omega/2)]^m.

    Args:
        order: Differential order m.
        omegas: Array of angular frequencies. Default: 200 points from 0.01 to pi.

    Returns:
        (omegas, magnitudes) tuple.
    """
    if omegas is None:
        omegas = np.linspace(0.01, np.pi, 200)
    omegas = np.asarray(omegas, dtype=float)
    magnitudes = (2 * np.sin(omegas / 2)) ** order
    return omegas, magnitudes
