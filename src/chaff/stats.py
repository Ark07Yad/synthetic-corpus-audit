"""Small statistics helpers, implemented in pure Python.

These exist because chaff refuses a numpy dependency (OBJECTIVE §2 G2): the tool has
to be droppable onto a data node next to a crawl dump with no build step. Every
function here is O(n) or O(n log n) on a list of floats and is hot enough to matter,
so they stay allocation-light.
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: Sequence[float]) -> float:
    """Population variance. Returns 0.0 for fewer than two observations."""
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values) / len(values)


def stdev(values: Sequence[float]) -> float:
    return math.sqrt(variance(values))


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile; ``q`` in [0, 100]."""
    if not values:
        return 0.0
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be within [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def mad(values: Sequence[float], scale: bool = True) -> float:
    """Median absolute deviation.

    With ``scale=True`` the result is multiplied by 1.4826 so it estimates the
    standard deviation of a normal distribution. chaff uses MAD rather than the
    standard deviation throughout fusion because contamination scores are computed
    *on the same corpus being audited* — if 30% of that corpus is synthetic, the
    contaminated documents drag the mean and variance toward themselves and mask
    the very anomaly being measured. The median and MAD have a 50% breakdown point,
    so they survive that.
    """
    if not values:
        return 0.0
    med = median(values)
    deviation = median([abs(v - med) for v in values])
    return deviation * 1.4826 if scale else deviation


def robust_z(value: float, med: float, dispersion: float) -> float:
    """Robust z-score. Degenerate (zero-dispersion) inputs score 0.0 rather than
    infinity, which keeps a constant-valued signal from dominating fusion."""
    if dispersion <= 0.0:
        return 0.0
    return (value - med) / dispersion


def shannon_entropy(counts: Iterable[float], base: float = 2.0) -> float:
    """Shannon entropy of a count vector, in bits by default."""
    values = [c for c in counts if c > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    log_base = math.log(base)
    return -sum((c / total) * (math.log(c / total) / log_base) for c in values)


def linear_regression(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    """Ordinary least squares fit. Returns ``(slope, intercept, r_squared)``.

    Used for the Zipf log-log rank/frequency fit and Heaps' law in phase 2. Returns
    zeros when the fit is undefined (fewer than two points, or no variance in x).
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0, 0.0, 0.0

    mean_x = mean(xs)
    mean_y = mean(ys)
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    if ss_xx <= 0.0:
        return 0.0, 0.0, 0.0

    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot <= 0.0:
        return slope, intercept, 1.0
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return slope, intercept, 1.0 - (ss_res / ss_tot)


def describe(values: Sequence[float]) -> Dict[str, float]:
    """Summary block used in corpus-level report sections."""
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0,
                "min": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": float(len(values)),
        "mean": mean(values),
        "median": median(values),
        "stdev": stdev(values),
        "min": min(values),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
        "p95": percentile(values, 95),
        "max": max(values),
    }
