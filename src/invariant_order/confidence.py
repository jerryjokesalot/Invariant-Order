"""Statistical comparison of signal segments with confidence bounds."""

from dataclasses import dataclass
import numpy as np
from .core import transform


@dataclass
class ComparisonResult:
    effect_size: float
    p_value: float
    stat_before: float
    stat_after: float
    n_permutations: int


def compare(before, after, nuisance="multiplicative", order=1,
            n_permutations=1000, stat="variance"):
    """Compare two signal segments for structural difference.

    Applies the invariant transform to each segment separately, then
    uses a permutation test to determine whether the transformed
    statistics differ significantly.

    Args:
        before: Signal segment before the suspected change.
        after: Signal segment after the suspected change.
        nuisance: 'multiplicative' or 'additive'.
        order: Differential order m.
        n_permutations: Number of permutations for the test.
        stat: Statistic to compare. 'variance' (default) or 'mean_abs'.

    Returns:
        ComparisonResult with effect_size and p_value.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)

    t_before = transform(before, nuisance, order)
    t_after = transform(after, nuisance, order)

    stat_fn = _STAT_FUNCTIONS.get(stat)
    if stat_fn is None:
        raise ValueError(f"stat must be one of {list(_STAT_FUNCTIONS)}, got '{stat}'")

    s_before = stat_fn(t_before)
    s_after = stat_fn(t_after)
    observed = abs(s_after - s_before)

    pooled_std = np.sqrt((np.var(t_before, ddof=1) + np.var(t_after, ddof=1)) / 2)
    effect_size = (s_after - s_before) / pooled_std if pooled_std > 0 else 0.0

    # Permutation test
    combined = np.concatenate([t_before, t_after])
    n_b = len(t_before)
    rng = np.random.default_rng(42)
    count_extreme = 0

    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        perm_before = perm[:n_b]
        perm_after = perm[n_b:]
        perm_diff = abs(stat_fn(perm_after) - stat_fn(perm_before))
        if perm_diff >= observed:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return ComparisonResult(
        effect_size=effect_size,
        p_value=p_value,
        stat_before=s_before,
        stat_after=s_after,
        n_permutations=n_permutations,
    )


def _variance(x):
    return np.var(x, ddof=1) if len(x) > 1 else 0.0


def _mean_abs(x):
    return np.mean(np.abs(x))


_STAT_FUNCTIONS = {
    "variance": _variance,
    "mean_abs": _mean_abs,
}
