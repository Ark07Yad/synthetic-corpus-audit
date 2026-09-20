"""Statistics helpers. These are load-bearing for fusion, so the robustness
properties are asserted explicitly rather than assumed."""

from __future__ import annotations

import math

from chaff.stats import (
    describe,
    linear_regression,
    mad,
    median,
    percentile,
    robust_z,
    shannon_entropy,
    stdev,
    variance,
)


def test_median_handles_even_and_odd_lengths():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5
    assert median([]) == 0.0


def test_percentile_interpolates():
    assert percentile([0, 10], 50) == 5.0
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4


def test_mad_resists_outliers_where_stdev_does_not():
    clean = [10, 11, 12, 13, 14]
    dirty = clean + [10_000]
    # The contaminated observation moves the standard deviation by orders of
    # magnitude but barely moves the MAD. This is why fusion uses MAD.
    assert stdev(dirty) > stdev(clean) * 100
    assert mad(dirty) < mad(clean) * 2


def test_robust_z_is_zero_for_degenerate_dispersion():
    # A constant-valued signal must not dominate fusion with an infinite score.
    assert robust_z(5.0, 5.0, 0.0) == 0.0


def test_variance_of_single_observation_is_zero():
    assert variance([7]) == 0.0


def test_shannon_entropy_bounds():
    assert shannon_entropy([1, 1, 1, 1]) == 2.0          # uniform over 4 -> log2(4)
    assert shannon_entropy([1]) == 0.0                   # no uncertainty
    assert shannon_entropy([]) == 0.0
    assert shannon_entropy([97, 1, 1, 1]) < 0.5          # heavily skewed


def test_zipfian_series_fits_slope_near_minus_one():
    ranks = [math.log(r) for r in range(1, 201)]
    freqs = [math.log(1000.0 / r) for r in range(1, 201)]
    slope, _intercept, r2 = linear_regression(ranks, freqs)
    assert abs(slope - (-1.0)) < 1e-9
    assert r2 > 0.999


def test_regression_is_defined_for_degenerate_input():
    assert linear_regression([1], [1]) == (0.0, 0.0, 0.0)
    assert linear_regression([2, 2, 2], [1, 2, 3]) == (0.0, 0.0, 0.0)


def test_describe_on_empty_input_is_all_zero():
    assert describe([])["n"] == 0
