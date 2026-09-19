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
    # Stated as a ratio rather than an absolute level. The size of the bias is
    # a property of the panel and moves whenever the generator does; what has
    # to hold is that the old null is biased and the current one is not.
    assert biased > 0.002, "the old null should be visibly biased"
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


# --------------------------------------------------------------------------
# a calibration describes one dataset and refuses to describe another
# --------------------------------------------------------------------------


def test_a_calibration_records_what_it_describes(panel: Panel, calibration) -> None:
    """Not metadata. Without these a tau is a number with no referent."""
    from alpha.data.cache import snapshot_id
    from alpha.registry.db import grammar_fingerprint

    assert calibration.data_snapshot_id == snapshot_id(panel)
    assert calibration.grammar_fingerprint == grammar_fingerprint()
    assert calibration.block_length == DEFAULT_BLOCK
    assert calibration.horizon == 1
    assert calibration.source_name == "random_tree"
    assert calibration.panel_shape == panel.shape


def test_a_calibration_accepts_the_panel_it_was_measured_on(
    panel: Panel, calibration
) -> None:
    calibration.assert_applies_to(panel, horizon=1)


def test_a_calibration_refuses_a_different_panel(panel: Panel, calibration) -> None:
    """The guard that matters. A tau from a synthetic panel applied to real
    data thresholds against a distribution that was never measured, and
    nothing downstream would look wrong."""
    from alpha.screen.null import CalibrationMismatch

    other = generate(
        np.random.default_rng(999), SyntheticSpec(n_days=600, n_instruments=120)
    ).panel
    with pytest.raises(CalibrationMismatch, match="different dataset"):
        calibration.assert_applies_to(other, horizon=1)


def test_a_calibration_refuses_a_different_horizon(panel: Panel, calibration) -> None:
    from alpha.screen.null import CalibrationMismatch

    with pytest.raises(CalibrationMismatch, match="horizon"):
        calibration.assert_applies_to(panel, horizon=5)


def test_a_calibration_refuses_a_changed_grammar(panel: Panel, calibration) -> None:
    """If the operator set changed, the candidate space changed, and so did
    the distribution of the maximum over it."""
    from dataclasses import replace

    from alpha.screen.null import CalibrationMismatch

    stale = replace(calibration, grammar_fingerprint="0" * 64)
    with pytest.raises(CalibrationMismatch, match="grammar"):
        stale.assert_applies_to(panel, horizon=1)


def test_the_binding_reaches_the_run_config(panel: Panel, calibration) -> None:
    binding = calibration.binding()
    assert binding["null_data_snapshot_id"] == calibration.data_snapshot_id
    assert binding["null_block_length"] == DEFAULT_BLOCK
    assert binding["null_source"] == "random_tree"
    assert binding["null_tau"] == calibration.tau


def test_a_screen_refuses_a_calibration_from_another_panel(panel: Panel) -> None:
    """Made impossible at the point of use, not documented."""
    from alpha.registry.db import Registry
    from alpha.screen.null import CalibrationMismatch
    from alpha.screen.screen import Screen, ScreenConfig

    calibration = calibrate(
        panel, np.random.default_rng(31), n_trials=200, min_scored_days=200
    )
    other = generate(
        np.random.default_rng(998), SyntheticSpec(n_days=600, n_instruments=120)
    ).panel

    registry = Registry.open(":memory:")
    config = ScreenConfig.from_calibration(calibration)
    run = registry.start_run({**config.as_config(), **calibration.binding()})

    Screen(panel, registry, run.id, config, calibration=calibration)
    with pytest.raises(CalibrationMismatch):
        Screen(other, registry, run.id, config, calibration=calibration)
    registry.close()


def test_a_screen_refuses_a_threshold_that_was_not_measured(panel: Panel) -> None:
    """Passing a calibration and then ignoring its tau is the same error in a
    more comfortable disguise."""
    from alpha.registry.db import Registry
    from alpha.screen.screen import Screen, ScreenConfig

    calibration = calibrate(
        panel, np.random.default_rng(33), n_trials=200, min_scored_days=200
    )
    registry = Registry.open(":memory:")
    run = registry.start_run({})
    with pytest.raises(ValueError, match="from_calibration"):
        Screen(panel, registry, run.id, ScreenConfig(min_abs_ic=0.5), calibration=calibration)
    registry.close()


# --------------------------------------------------------------------------
# the maximum, measured rather than assumed
# --------------------------------------------------------------------------


def test_the_maximum_is_a_curve_not_a_point(calibration) -> None:
    """The bar depends on how many candidates were tried, and one number
    hides the search size it assumed."""
    curve = calibration.max_abs_ic
    assert len(curve.sizes) >= 2
    assert all(b >= 8 for b in curve.n_batches)


def test_the_maximum_grows_with_search_size(calibration) -> None:
    curve = calibration.max_abs_ic
    assert list(curve.mean) == sorted(curve.mean)
    assert list(calibration.max_ic_ir.mean) == sorted(calibration.max_ic_ir.mean)


def test_the_maximum_exceeds_the_typical_trial(calibration) -> None:
    assert calibration.max_abs_ic.mean[0] > calibration.abs_ic_quantiles["p50"]


def test_reading_the_curve_interpolates_in_log_size(calibration) -> None:
    curve = calibration.max_abs_ic
    low, high = curve.sizes[0], curve.sizes[-1]
    middle = curve.expected_at(int(np.sqrt(low * high)))
    assert curve.mean[0] <= middle <= curve.mean[-1]


def test_reading_the_curve_beyond_the_ladder_is_refused(calibration) -> None:
    """Extrapolating past the measurement is the substitution this module
    exists to remove."""
    from alpha.screen.null import CalibrationMismatch

    with pytest.raises(CalibrationMismatch, match="outside the measured ladder"):
        calibration.max_abs_ic.expected_at(10_000_000)


def test_the_analytic_form_is_reported_but_not_authoritative(calibration) -> None:
    """sqrt(2 ln N) assumes independent trials and Gaussian Sharpes, and
    grammar candidates share subtrees. It is a sanity check beside the
    measurement, not a substitute for it."""
    for size in calibration.max_ic_ir.sizes:
        analytic = calibration.analytic_max_ic_ir(size)
        assert np.isfinite(analytic)
        assert analytic > 0.0


def test_sigma_sr_is_the_dispersion_of_the_signed_statistic(calibration) -> None:
    """Taking absolute values first would halve it and make the analytic
    comparator look far too small for no reason but the accounting."""
    finite = calibration.ic_ir[np.isfinite(calibration.ic_ir)]
    assert calibration.sigma_sr == pytest.approx(float(finite.std(ddof=1)))
    assert calibration.sigma_sr > float(np.abs(finite).std(ddof=1))
