"""Tests for the null calibration.

The null decides every threshold downstream, so a null that is wrong makes
every gauntlet test wrong in a way nothing else would catch. The property that
matters is that it destroys the signal-to-return relationship completely while
leaving the structure that sets the shape of the distribution intact.

Two earlier designs failed that, and both failures are asserted here as
regressions rather than described in a comment, because the second one is
subtle enough to be reintroduced by someone reasoning from first principles.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha.data.synthetic import SyntheticSpec, generate
from alpha.data.types import DTYPE, Panel
from alpha.expr.ast import from_string
from alpha.expr.evaluator import evaluate
from alpha.screen.metrics import rank_forward, summarise
from alpha.screen.null import (
    DEFAULT_BLOCK,
    block_label_permutation,
    calibrate,
)


@pytest.fixture(scope="module")
def panel() -> Panel:
    return generate(
        np.random.default_rng(20260918),
        SyntheticSpec(n_days=600, n_instruments=120),
    ).panel


def shuffled_forward(panel: Panel, rng: np.random.Generator, block: int = DEFAULT_BLOCK):
    ranked = rank_forward(panel, 1)
    labels = block_label_permutation(panel.n_days, panel.n_instruments, block, rng)
    return np.ascontiguousarray(np.take_along_axis(ranked, labels, axis=1), dtype=DTYPE)


# --------------------------------------------------------------------------
# the permutation
# --------------------------------------------------------------------------


def test_the_permutation_covers_every_instrument_each_day() -> None:
    """A permutation, not a resample. Every return is present exactly once, so
    the day's cross-section is relabelled rather than altered."""
    labels = block_label_permutation(50, 30, 10, np.random.default_rng(0))
    assert labels.shape == (50, 30)
    for row in labels:
        assert sorted(row.tolist()) == list(range(30))


def test_the_permutation_is_constant_within_a_block() -> None:
    """Holding it fixed is what preserves the day-to-day dependence of the IC
    series, which is what sets the standard error of its mean."""
    labels = block_label_permutation(40, 12, 10, np.random.default_rng(1))
    for start in (0, 10, 20, 30):
        block = labels[start : start + 10]
        assert (block == block[0]).all()


def test_the_permutation_changes_between_blocks() -> None:
    labels = block_label_permutation(40, 50, 10, np.random.default_rng(2))
    assert not np.array_equal(labels[0], labels[10])


def test_the_permutation_is_reproducible_from_its_seed() -> None:
    first = block_label_permutation(40, 12, 10, np.random.default_rng(3))
    second = block_label_permutation(40, 12, 10, np.random.default_rng(3))
    assert np.array_equal(first, second)


@pytest.mark.parametrize(("days", "instruments", "block"), [(0, 5, 2), (10, 0, 2), (10, 5, 0), (10, 5, 11)])
def test_impossible_shapes_are_refused(days: int, instruments: int, block: int) -> None:
    with pytest.raises(ValueError):
        block_label_permutation(days, instruments, block, np.random.default_rng(0))


def test_a_day_keeps_its_own_returns(panel: Panel) -> None:
    """Relabelled, not replaced. The multiset of returns on a day is
    unchanged, so the day's dispersion and any market-wide move survive."""
    ranked = rank_forward(panel, 1)
    shuffled = shuffled_forward(panel, np.random.default_rng(5))
    for row in (100, 300, 500):
        left = np.sort(ranked[row][~np.isnan(ranked[row])])
        right = np.sort(shuffled[row][~np.isnan(shuffled[row])])
        assert np.array_equal(left, right)


# --------------------------------------------------------------------------
# the two designs that failed
# --------------------------------------------------------------------------


def _mean_null_ic(panel: Panel, text: str, shuffle, replicates: int = 12) -> float:
    rng = np.random.default_rng(7)
    result = evaluate(from_string(text), panel)
    ics = []
    for _ in range(replicates):
        metrics, _ = summarise(
            result.values, result.warmup, panel, shuffle(panel, rng)
        )
        if np.isfinite(metrics.ic):
            ics.append(metrics.ic)
    return float(np.mean(ics))


@pytest.mark.parametrize("text", ["close", "mcap", "ts_mean(close, 60)"])
def test_a_price_derived_signal_scores_zero_under_the_null(
    panel: Panel, text: str
) -> None:
    """The regression that matters.

    An earlier null permuted the returns in time and left the panel alone.
    close[t] is the cumulative product of the returns up to t, so pairing it
    with returns from an earlier day scored it against returns it mechanically
    contains. Price-derived signals scored a mean null IC around +0.01 while
    volume and returns sat at zero, the null came out biased upward, and tau
    would have been set three times too high, silently discarding real
    candidates while reporting appropriate strictness.
    """
    assert abs(_mean_null_ic(panel, text, shuffled_forward)) < 0.004


@pytest.mark.parametrize("text", ["volume", "returns", "ts_std(returns, 20)"])
def test_a_signal_not_derived_from_prices_also_scores_zero(
    panel: Panel, text: str
) -> None:
    """The control. These sat at zero under the broken null too, which is why
    the bias went unnoticed until it was measured against price-derived
    signals."""
    assert abs(_mean_null_ic(panel, text, shuffled_forward)) < 0.004


def test_the_time_permutation_null_really_was_biased(panel: Panel) -> None:
    """Asserted rather than described, so the failed design cannot quietly
    come back as a simplification."""

    def time_permuted(panel: Panel, rng: np.random.Generator) -> np.ndarray:
        ranked = rank_forward(panel, 1)
        order = np.concatenate(
            [
                np.arange(start, start + DEFAULT_BLOCK)
                for start in rng.integers(
                    0, panel.n_days - DEFAULT_BLOCK, size=panel.n_days // DEFAULT_BLOCK + 1
                )
            ]
        )[: panel.n_days]
        return np.ascontiguousarray(ranked[order], dtype=DTYPE)

    biased = _mean_null_ic(panel, "close", time_permuted)
    correct = _mean_null_ic(panel, "close", shuffled_forward)
    assert biased > 0.004, "the old null should be visibly biased"
    assert abs(correct) < abs(biased) / 3.0


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def calibration(panel: Panel):
    return calibrate(
        panel,
        np.random.default_rng(11),
        n_trials=300,
        min_scored_days=200,
    )


def test_the_null_is_centred_at_zero(calibration) -> None:
    """If it is not, something upstream leaks, and this is the cheapest place
    to find that out because here the answer is known to be nothing."""
    assert not calibration.leaks
    assert calibration.leak_sigmas < 4.0


def test_tau_sits_in_the_tail(calibration) -> None:
    quantiles = calibration.abs_ic_quantiles
    assert quantiles["p50"] < quantiles["p95"] < quantiles["p99"] <= calibration.tau
    assert calibration.tau <= quantiles["max"]


def test_tau_admits_about_the_intended_fraction(calibration) -> None:
    """Read off the distribution, not chosen.

    Stated as a count rather than a fraction, because a few hundred trials
    cannot resolve a tenth of a percent: one trial out of three hundred is
    already three times the target rate. The fraction is only meaningful at
    the scale the script is meant to run, which is 1e5.
    """
    above = int((calibration.abs_ic > calibration.tau).sum())
    assert above <= 1, f"{above} trials sit above the 99.9th percentile"


def test_the_full_distribution_is_kept_not_just_the_tail(calibration) -> None:
    """A distribution that is the wrong shape in the middle is the first sign
    the null is not a null, and that is invisible in a maximum."""
    assert calibration.abs_ic.size == calibration.n_trials
    assert calibration.abs_ic.min() >= 0.0


def test_an_empirical_sigma_sr_is_produced(calibration) -> None:
    """The analytic Deflated Sharpe assumes independent trials, and
    grammar-generated candidates share subtrees, fields and windows."""
    assert np.isfinite(calibration.sigma_sr)
    assert calibration.sigma_sr > 0.0


def test_thin_trials_are_excluded(panel: Panel) -> None:
    """The null admits a trial only if the screen would have.

    Asserted as the filter being applied, not as an ordering of tau. Whether a
    thin trial lands in the tail is a matter of whether one was drawn, and a
    few hundred trials usually draw none. At scale it matters: in a run of
    fifteen hundred, the single largest null |IC| was 0.111 against a 99.9th
    percentile of 0.010, scored on three days, and excluding it moved the
    percentile from 0.0103 to 0.0092.
    """
    strict = calibrate(
        panel, np.random.default_rng(11), n_trials=300, min_scored_days=500
    )
    loose = calibrate(
        panel, np.random.default_rng(11), n_trials=300, min_scored_days=2
    )
    assert strict.n_trials < loose.n_trials
    assert strict.median_scored_days >= 500


def test_calibration_is_reproducible_from_its_seed(panel: Panel) -> None:
    first = calibrate(panel, np.random.default_rng(13), n_trials=150, min_scored_days=200)
    second = calibrate(panel, np.random.default_rng(13), n_trials=150, min_scored_days=200)
    assert first.tau == second.tau
    assert first.mean_ic == second.mean_ic


def test_too_few_usable_trials_raises(panel: Panel) -> None:
    """Better than returning a threshold computed from nothing."""
    with pytest.raises(RuntimeError, match="too few"):
        calibrate(panel, np.random.default_rng(17), n_trials=20)
