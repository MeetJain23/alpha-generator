"""Tests for the numba kernels.

The evaluator tests already check every operator against hand-computed values.
What is left here is what only the kernels can get wrong: agreement with a
brute-force window over ragged data, the compilation flags the NaN policy
depends on, float64 accumulation, and the counters.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alpha.expr import kernels as K

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

KERNEL_NAMES = (
    "ts_mean",
    "ts_std",
    "ts_rank",
    "ts_min",
    "ts_max",
    "ts_argmax",
    "decay_linear",
    "correlation",
    "window_fill",
)


@pytest.fixture(scope="module")
def ragged() -> tuple[np.ndarray, np.ndarray]:
    """Two planes with independent gaps, so pairwise counting has work to do."""
    rng = np.random.default_rng(20260918)
    x = rng.normal(100.0, 5.0, size=(140, 6)).astype(np.float32)
    y = rng.normal(50.0, 3.0, size=(140, 6)).astype(np.float32)
    x[rng.random(x.shape) < 0.15] = np.nan
    y[rng.random(y.shape) < 0.10] = np.nan
    return x, y


def brute_force(x: np.ndarray, d: int, floor: int, reduce) -> np.ndarray:
    """The obvious O(n * d) implementation, which the kernels must match."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for t in range(d - 1, x.shape[0]):
        for j in range(x.shape[1]):
            window = x[t - d + 1 : t + 1, j]
            present = window[~np.isnan(window)]
            if present.size >= floor:
                out[t, j] = reduce(window, present)
    return out


def compare(got: np.ndarray, expected: np.ndarray, tol: float = 2e-5) -> None:
    assert np.array_equal(np.isnan(got), np.isnan(expected)), "NaN patterns differ"
    both = ~np.isnan(got)
    if both.any():
        scale = np.maximum(1.0, np.abs(expected[both]))
        assert np.max(np.abs(got[both] - expected[both]) / scale) < tol


# --------------------------------------------------------------------------
# agreement with brute force
# --------------------------------------------------------------------------


@pytest.mark.parametrize("d", [5, 20, 60])
def test_ts_mean_matches_brute_force(ragged, d: int) -> None:
    x, _ = ragged
    floor = math.ceil(0.8 * d)
    got, _, _, _ = K.ts_mean(x, d, floor, d - 1)
    compare(got, brute_force(x, d, floor, lambda w, p: p.mean()))


@pytest.mark.parametrize("d", [5, 20, 60])
def test_ts_std_matches_brute_force(ragged, d: int) -> None:
    x, _ = ragged
    floor = math.ceil(0.8 * d)
    got, _, _, clamped = K.ts_std(x, d, floor, d - 1)
    compare(got, brute_force(x, d, floor, lambda w, p: p.std(ddof=1)))
    assert clamped == 0, "float64 accumulation should never need the clamp here"


@pytest.mark.parametrize("d", [5, 20, 60])
def test_ts_max_and_ts_min_match_brute_force(ragged, d: int) -> None:
    x, _ = ragged
    floor = math.ceil(0.8 * d)
    got, _, _, _ = K.ts_max(x, d, floor, d - 1)
    compare(got, brute_force(x, d, floor, lambda w, p: p.max()))
    got, _, _, _ = K.ts_min(x, d, floor, d - 1)
    compare(got, brute_force(x, d, floor, lambda w, p: p.min()))


@pytest.mark.parametrize("d", [5, 20, 60])
def test_ts_argmax_matches_brute_force(ragged, d: int) -> None:
    x, _ = ragged
    floor = math.ceil(0.8 * d)
    got, _, _, _ = K.ts_argmax(x, d, floor, d - 1)
    expected = brute_force(
        x, d, floor, lambda w, p: float(len(w) - 1 - int(np.nanargmax(w)))
    )
    compare(got, expected)


@pytest.mark.parametrize("d", [5, 20, 60])
def test_decay_linear_matches_brute_force(ragged, d: int) -> None:
    x, _ = ragged
    floor = math.ceil(0.8 * d)

    def reference(window, present):
        weights = np.arange(1, len(window) + 1, dtype=np.float64)
        keep = ~np.isnan(window)
        return float((weights[keep] * window[keep]).sum() / weights[keep].sum())

    got, _, _, _ = K.decay_linear(x, d, floor, d - 1)
    compare(got, brute_force(x, d, floor, reference), tol=1e-4)


@pytest.mark.parametrize("d", [5, 20, 60])
def test_ts_rank_matches_brute_force(ragged, d: int) -> None:
    x, _ = ragged
    floor = math.ceil(0.8 * d)

    def reference(window, present):
        value = window[-1]
        if np.isnan(value):
            return np.nan
        less = int((present < value).sum())
        equal = int((present == value).sum())
        return (less + (equal + 1) * 0.5) / present.size

    got, _, _, _ = K.ts_rank(np.ascontiguousarray(x.T), d, floor, d - 1)
    compare(np.ascontiguousarray(got.T), brute_force(x, d, floor, reference))


@pytest.mark.parametrize("d", [5, 20, 60])
def test_correlation_matches_brute_force(ragged, d: int) -> None:
    x, y = ragged
    floor = math.ceil(0.8 * d)
    got, _, _, _ = K.correlation(x, y, d, floor, d - 1)

    expected = np.full(x.shape, np.nan, dtype=np.float64)
    for t in range(d - 1, x.shape[0]):
        for j in range(x.shape[1]):
            a = x[t - d + 1 : t + 1, j].astype(np.float64)
            b = y[t - d + 1 : t + 1, j].astype(np.float64)
            keep = ~np.isnan(a) & ~np.isnan(b)
            if keep.sum() >= max(floor, 2) and a[keep].std() > 0 and b[keep].std() > 0:
                expected[t, j] = np.corrcoef(a[keep], b[keep])[0, 1]
    compare(got, expected, tol=1e-4)


# --------------------------------------------------------------------------
# flags the NaN policy depends on
# --------------------------------------------------------------------------


def test_fastmath_is_off_on_every_kernel() -> None:
    """fastmath assumes no NaNs are present. Every kernel tests for NaN with
    v == v, so fastmath would fold those tests to True and void the NaN policy
    silently, producing wrong numbers that look like right ones."""
    assert K.KERNEL_FLAGS["fastmath"] is False
    for name in KERNEL_NAMES:
        assert getattr(K, name).targetoptions.get("fastmath") is False, name


def test_caching_is_on_for_every_kernel() -> None:
    """Half a second of compilation per function would dwarf the evaluation
    budget in the short-lived worker processes the generator runs in."""
    assert K.KERNEL_FLAGS["cache"] is True
    for name in KERNEL_NAMES:
        # numba keeps the cache flag as the dispatcher's cache object rather
        # than in targetoptions: NullCache means compilation happens every
        # time the process starts.
        assert type(getattr(K, name)._cache).__name__ != "NullCache", name


def test_nan_is_never_treated_as_a_value() -> None:
    """A column that is entirely NaN stays NaN rather than becoming zero."""
    x = np.full((30, 2), np.nan, dtype=np.float32)
    x[:, 0] = 1.0
    got, _, _, _ = K.ts_mean(x, 10, 8, 9)
    assert np.isnan(got[:, 1]).all()
    assert np.allclose(got[9:, 0], 1.0)


# --------------------------------------------------------------------------
# accumulation
# --------------------------------------------------------------------------


def test_variance_survives_price_level_data_over_a_long_window() -> None:
    """The case float32 accumulation gets wrong: a small dispersion riding on
    a large level, where the sum of squares and the squared sum are nearly
    equal and their difference is what is wanted."""
    rng = np.random.default_rng(3)
    x = (5000.0 + rng.normal(0.0, 0.01, size=(600, 3))).astype(np.float32)
    got, _, _, clamped = K.ts_std(x, 250, 200, 249)
    assert clamped == 0
    finite = got[np.isfinite(got)]
    assert finite.size > 0
    assert (finite >= 0).all()


def test_a_constant_series_gives_exactly_zero_not_a_nan() -> None:
    """A NaN manufactured by cancellation would be indistinguishable from a
    NaN meaning there was nothing to compute from."""
    x = np.full((80, 2), 1234.5, dtype=np.float32)
    got, _, _, clamped = K.ts_std(x, 20, 16, 19)
    assert clamped == 0
    assert (got[19:] == 0.0).all()


def test_a_constant_leg_makes_correlation_nan_not_zero() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(60, 2)).astype(np.float32)
    y = np.full((60, 2), 7.0, dtype=np.float32)
    got, _, _, clamped = K.correlation(x, y, 20, 16, 19)
    assert clamped == 0
    assert np.isnan(got).all()


def test_correlation_stays_inside_its_range(ragged) -> None:
    x, y = ragged
    got, _, _, _ = K.correlation(x, y, 20, 16, 19)
    finite = got[np.isfinite(got)]
    assert (finite >= -1.0).all() and (finite <= 1.0).all()


# --------------------------------------------------------------------------
# counters and the first reportable row
# --------------------------------------------------------------------------


def test_a_clean_window_degrades_nothing() -> None:
    x = np.arange(200, dtype=np.float32).reshape(100, 2)
    _, degraded, considered, _ = K.ts_mean(x, 10, 8, 9)
    assert considered > 0
    assert degraded == 0


def test_every_cell_degrades_when_every_window_has_a_gap() -> None:
    x = np.arange(200, dtype=np.float32).reshape(100, 2)
    x[::5] = np.nan
    _, degraded, considered, _ = K.ts_mean(x, 10, 8, 9)
    assert degraded == considered > 0


def test_nothing_is_emitted_before_the_first_reportable_row() -> None:
    """warmup is the authority on what may be emitted. min_periods only says
    what may be computed from what is emitted."""
    x = np.arange(200, dtype=np.float32).reshape(100, 2)
    got, _, _, _ = K.ts_mean(x, 10, 8, 40)
    assert np.isnan(got[:40]).all()
    assert np.isfinite(got[40:]).all()


def test_cells_before_the_first_row_are_not_counted() -> None:
    """Or the degradation fraction would be diluted by cells nobody can use."""
    x = np.arange(200, dtype=np.float32).reshape(100, 2)
    _, _, early, _ = K.ts_mean(x, 10, 8, 9)
    _, _, late, _ = K.ts_mean(x, 10, 8, 40)
    assert late < early


def test_window_fill_reports_the_present_fraction() -> None:
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    x[3, 0] = np.nan
    fill = K.window_fill(x, 5, 4)
    assert np.isnan(fill[:4]).all()
    assert fill[4, 0] == pytest.approx(0.8)
    assert fill[4, 1] == pytest.approx(1.0)
    assert fill[8, 0] == pytest.approx(1.0)
