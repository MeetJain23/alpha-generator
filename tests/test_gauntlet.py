"""Tests for the gauntlet and its purged splits.

Two properties carry the weight.

No threshold is frozen. The gauntlet measures everything and refuses to return
a verdict until each threshold has been set from a calibration, because a
number written by hand acquires the authority of having been written down.

The two tests that misbehave if written the obvious way behave. Delay is a
shape rather than a level, so a slow signal and a fast signal both pass.
Jitter asserts a plateau rather than "IC exceeds tau everywhere", which would
fail every real alpha whose IC sits near tau.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha.data.synthetic import SyntheticSpec, generate
from alpha.data.types import Panel
from alpha.expr.ast import Node, from_string
from alpha.gauntlet.folds import Fold, purged_folds, regime_split
from alpha.gauntlet.gauntlet import (
    TEST_DIRECTIONS,
    THRESHOLD_FOR_TEST,
    Gauntlet,
    GauntletThresholds,
    NotCalibrated,
    _jittered,
)

MOMENTUM = "div(ts_delay(close, 20), ts_delay(close, 250))"


@pytest.fixture(scope="module")
def panel() -> Panel:
    return generate(
        np.random.default_rng(20260919),
        SyntheticSpec(n_days=1400, n_instruments=150, momentum_strength=0.08),
    ).panel


@pytest.fixture(scope="module")
def gauntlet(panel: Panel) -> Gauntlet:
    return Gauntlet(panel, n_folds=5)


# --------------------------------------------------------------------------
# purged folds
# --------------------------------------------------------------------------


def test_folds_start_after_warmup() -> None:
    folds = purged_folds(2000, warmup=250, horizon=1, n_folds=4)
    assert folds[0].start >= 250


def test_folds_are_separated_by_the_full_overlap() -> None:
    """A windowed signal reaches back warmup days and a forward return reaches
    ahead horizon days, so that is exactly the gap that has to be cut."""
    warmup, horizon = 250, 5
    folds = purged_folds(3000, warmup=warmup, horizon=horizon, n_folds=4)
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.start - earlier.stop == warmup + horizon


def test_folds_do_not_overlap() -> None:
    folds = purged_folds(3000, warmup=100, horizon=1, n_folds=5)
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.stop <= later.start


def test_folds_stay_inside_the_panel() -> None:
    folds = purged_folds(2000, warmup=250, horizon=1, n_folds=5)
    assert folds[-1].stop <= 2000


def test_fewer_folds_rather_than_useless_ones() -> None:
    """A thirty-day fold produces an IC whose standard error swamps any
    difference between folds, so comparing them measures noise against noise."""
    folds = purged_folds(1200, warmup=250, horizon=1, n_folds=5, min_fold_days=120)
    assert len(folds) < 5
    assert all(f.n_days >= 120 for f in folds)


def test_a_panel_too_short_for_any_fold_gives_none() -> None:
    assert purged_folds(300, warmup=250, horizon=1, n_folds=5) == ()


def test_impossible_arguments_are_refused() -> None:
    with pytest.raises(ValueError):
        purged_folds(0, warmup=10, horizon=1)
    with pytest.raises(ValueError):
        purged_folds(1000, warmup=10, horizon=0)


def test_regimes_split_on_realised_volatility() -> None:
    """A date range is a poor proxy: quiet years contain violent quarters."""
    calm, rough = regime_split([1.0, 5.0, 2.0, 8.0, 3.0, 9.0])
    assert sorted(calm) == [0, 2, 4]
    assert sorted(rough) == [1, 3, 5]


def test_regimes_ignore_undefined_days() -> None:
    calm, rough = regime_split([1.0, float("nan"), 9.0])
    assert 1 not in calm and 1 not in rough


# --------------------------------------------------------------------------
# nothing is frozen
# --------------------------------------------------------------------------


def test_every_threshold_starts_unset() -> None:
    assert set(GauntletThresholds().unset()) == set(THRESHOLD_FOR_TEST.values())


def test_a_verdict_is_refused_before_calibration(gauntlet: Gauntlet) -> None:
    result = gauntlet.run(from_string(MOMENTUM))
    assert not result.judged
    with pytest.raises(NotCalibrated):
        result.passed


def test_requiring_completeness_names_what_is_missing() -> None:
    with pytest.raises(NotCalibrated, match="min_abs_ic"):
        GauntletThresholds().require_complete()


def test_partial_thresholds_are_still_incomplete() -> None:
    partial = GauntletThresholds(min_abs_ic=0.01)
    assert "min_abs_ic" not in partial.unset()
    with pytest.raises(NotCalibrated):
        partial.require_complete()


def test_every_test_is_measured_even_with_no_thresholds(gauntlet: Gauntlet) -> None:
    result = gauntlet.run(from_string(MOMENTUM))
    assert {o.name for o in result.outcomes} == set(TEST_DIRECTIONS)
    assert all(o.threshold is None and o.passed is None for o in result.outcomes)


def test_a_full_set_of_thresholds_yields_a_verdict(gauntlet: Gauntlet, panel: Panel) -> None:
    thresholds = GauntletThresholds(
        min_abs_ic=0.001,
        min_fold_sign_agreement=0.5,
        min_fold_ic_fraction=0.0,
        max_delay_cliff=10.0,
        min_jitter_plateau=0.0,
        max_jitter_sign_flips=0,
        min_regime_sign_agreement=0.5,
        min_breadth=-5.0,
    )
    judged = Gauntlet(panel, thresholds=thresholds, n_folds=5)
    result = judged.run(from_string(MOMENTUM))
    assert result.judged
    assert result.passed


# --------------------------------------------------------------------------
# what the tests measure
# --------------------------------------------------------------------------


def test_a_planted_signal_beats_a_noise_signal(gauntlet: Gauntlet) -> None:
    strong = gauntlet.run(from_string(MOMENTUM)).measured
    weak = gauntlet.run(from_string("ts_mean(volume, 20)")).measured
    assert abs(strong["ic_magnitude"]) > abs(weak["ic_magnitude"])
    assert strong["fold_sign"] >= weak["fold_sign"]
    assert strong["breadth"] > weak["breadth"]


def test_the_sign_test_rewards_agreement_across_folds(gauntlet: Gauntlet) -> None:
    """The test that only exists because sign was decoupled from identity.
    While the hash carried the sign, asking whether the sign held across folds
    was asking whether the candidate was itself."""
    measured = gauntlet.run(from_string(MOMENTUM)).measured
    assert measured["fold_sign"] == pytest.approx(1.0)


def test_the_sign_test_is_a_fraction_of_folds(gauntlet: Gauntlet) -> None:
    for text in (MOMENTUM, "ts_mean(volume, 20)", "rank(returns)"):
        value = gauntlet.run(from_string(text)).measured["fold_sign"]
        assert 0.0 <= value <= 1.0


def test_breadth_falls_when_one_stretch_carries_everything(panel: Panel) -> None:
    """A signal earning everything on twenty days is an event study."""
    gauntlet = Gauntlet(panel, n_folds=5)
    spiky = gauntlet.run(from_string("ts_argmax(volume, 250)")).measured["breadth"]
    broad = gauntlet.run(from_string(MOMENTUM)).measured["breadth"]
    assert broad > spiky


# --------------------------------------------------------------------------
# delay is a shape, not a level
# --------------------------------------------------------------------------


def test_a_smooth_decay_is_not_a_cliff(gauntlet: Gauntlet) -> None:
    """The first decrement is about the size of the second, so the ratio sits
    near one rather than blowing up."""
    statistic = gauntlet.run(from_string(MOMENTUM)).measured["delay_shape"]
    assert 0.0 <= statistic < 5.0


def test_the_delay_statistic_carries_no_units_of_horizon(panel: Panel) -> None:
    """A monthly alpha and a fast one both pass, which is the whole point:
    failing a three-day signal for losing most of its IC to a one-day delay is
    failing it for being what it is."""
    gauntlet = Gauntlet(panel, n_folds=5)
    slow = gauntlet.run(from_string(MOMENTUM)).measured["delay_shape"]
    fast = gauntlet.run(from_string("div(ts_delay(close, 1), ts_delay(close, 5))"))
    assert np.isfinite(slow)
    assert np.isfinite(fast.measured["delay_shape"])


def test_the_delay_detail_records_every_step(gauntlet: Gauntlet) -> None:
    outcome = next(
        o for o in gauntlet.run(from_string(MOMENTUM)).outcomes if o.name == "delay_shape"
    )
    assert set(outcome.detail) == {"delay_0", "delay_1", "delay_2"}


# --------------------------------------------------------------------------
# jitter is a plateau, not a level
# --------------------------------------------------------------------------


def test_jitter_probes_land_inside_twenty_percent() -> None:
    """The ladder cannot express this: the neighbours of 20 are 10 and 60."""
    tree = from_string("ts_mean(close, 20)")
    windows = sorted({probe.nodes()[0].params[0] for probe in _jittered(tree)})
    assert windows
    assert all(16 <= w <= 24 for w in windows), windows


def test_every_jitter_neighbour_is_a_probe() -> None:
    """A probe is a sensitivity instrument, never a hypothesis, and never
    reaches the ledger."""
    tree = from_string("ts_mean(ts_std(close, 60), 20)")
    for neighbour in _jittered(tree):
        assert any(node.probe for node in neighbour.nodes())


def test_jitter_moves_one_parameter_at_a_time() -> None:
    """Moving several at once measures a direction in parameter space rather
    than sensitivity to each."""
    tree = from_string("ts_mean(ts_std(close, 60), 20)")
    original = [node.params for node in tree.nodes()]
    for neighbour in _jittered(tree):
        changed = sum(
            1
            for before, after in zip(original, [n.params for n in neighbour.nodes()], strict=True)
            if before != after
        )
        assert changed == 1


def test_a_tree_with_no_parameters_has_no_neighbourhood() -> None:
    assert list(_jittered(from_string("rank(close)"))) == []


def test_the_plateau_statistic_is_a_fraction_of_the_centre(gauntlet: Gauntlet) -> None:
    """Not 'IC exceeds tau at every setting', which would fail every real alpha
    whose IC sits near tau."""
    measured = gauntlet.run(from_string(MOMENTUM)).measured
    assert measured["jitter_plateau"] > 0.5
    assert measured["jitter_sign"] == 0.0


def test_a_planted_signal_has_a_flat_neighbourhood(gauntlet: Gauntlet) -> None:
    outcome = next(
        o for o in gauntlet.run(from_string(MOMENTUM)).outcomes
        if o.name == "jitter_plateau"
    )
    assert outcome.detail["n_neighbours"] > 0
    assert outcome.detail["flips"] == 0.0


# --------------------------------------------------------------------------
# binding
# --------------------------------------------------------------------------


def test_a_gauntlet_refuses_a_calibration_from_another_panel(panel: Panel) -> None:
    from alpha.screen.null import CalibrationMismatch, calibrate

    other = generate(
        np.random.default_rng(4242), SyntheticSpec(n_days=600, n_instruments=120)
    ).panel
    calibration = calibrate(
        other, np.random.default_rng(7), n_trials=200, min_scored_days=200
    )
    with pytest.raises(CalibrationMismatch):
        Gauntlet(panel, calibration=calibration)


def test_thresholds_take_tau_from_the_null() -> None:
    from alpha.screen.null import calibrate

    small = generate(
        np.random.default_rng(11), SyntheticSpec(n_days=600, n_instruments=120)
    ).panel
    calibration = calibrate(
        small, np.random.default_rng(13), n_trials=200, min_scored_days=200
    )
    thresholds = GauntletThresholds.from_calibration(calibration)
    assert thresholds.min_abs_ic == calibration.tau
    assert "min_abs_ic" not in thresholds.unset()


def test_the_thresholds_reach_the_run_config() -> None:
    config = GauntletThresholds(min_abs_ic=0.01).as_config()
    assert config["gauntlet_min_abs_ic"] == 0.01
    assert config["gauntlet_min_breadth"] is None
