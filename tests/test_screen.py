"""Tests for the screen and its metrics.

The load-bearing property is the alignment. A signal is scored against the
next day's return, never the same day's, and the test for that does not
inspect an index: it feeds in a signal that is exactly today's return and
asserts the IC is near zero. If the alignment were off by one, that signal
would score an IC of 1.0 and every number in the system would be wrong in a
direction nobody would question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.data.synthetic import SyntheticSpec, generate
from alpha.data.types import DTYPE, FIELDS, Panel
from alpha.expr.ast import from_string, random_tree
from alpha.expr.evaluator import evaluate, rank_rows
from alpha.registry.db import KillReason, Registry, Verdict
from alpha.screen.metrics import (
    coverage,
    daily_rank_ic,
    dispersion,
    forward_returns,
    rank_forward,
    summarise,
    turnover,
)
from alpha.screen.screen import Screen, ScreenConfig

NAN = np.nan


def make_panel(**planes: list[list[float]]) -> Panel:
    first = next(iter(planes.values()))
    n_days, n_inst = len(first), len(first[0])
    fields = {name: np.full((n_days, n_inst), np.nan, dtype=DTYPE) for name in FIELDS}
    for name, values in planes.items():
        fields[name] = np.array(values, dtype=DTYPE)
    fields["is_listed"] = np.ones((n_days, n_inst), dtype=DTYPE)
    return Panel(
        dates=pd.bdate_range("2020-01-01", periods=n_days, name="date"),
        instruments=np.array([f"A{i}" for i in range(n_inst)], dtype=object),
        fields=fields,
    )


@pytest.fixture(scope="module")
def panel() -> Panel:
    return generate(
        np.random.default_rng(20260918),
        SyntheticSpec(n_days=900, n_instruments=150, momentum_strength=0.15),
    ).panel


@pytest.fixture()
def registry() -> Registry:
    reg = Registry.open(":memory:")
    yield reg
    reg.close()


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------


def test_forward_returns_hold_the_next_days_return() -> None:
    panel = make_panel(returns=[[0.1], [0.2], [0.3], [0.4]])
    got = forward_returns(panel, 1)[:, 0]
    np.testing.assert_allclose(got[:3], [0.2, 0.3, 0.4])
    assert np.isnan(got[3])


def test_a_longer_horizon_compounds() -> None:
    panel = make_panel(returns=[[0.1], [0.1], [0.1], [0.1]])
    got = forward_returns(panel, 2)[:, 0]
    np.testing.assert_allclose(got[:2], [0.21, 0.21], rtol=1e-6)
    assert np.isnan(got[2]) and np.isnan(got[3])


def test_a_zero_horizon_is_refused() -> None:
    panel = make_panel(returns=[[0.1], [0.2]])
    with pytest.raises(ValueError):
        forward_returns(panel, 0)


def test_todays_return_does_not_predict_itself(panel: Panel) -> None:
    """The look-ahead test that matters. If the alignment were off by one this
    signal would score an IC of 1.0 instead of roughly nothing."""
    ranked_forward = rank_forward(panel, 1)
    signal = evaluate(from_string("returns"), panel)
    metrics, _ = summarise(signal.values, signal.warmup, panel, ranked_forward)
    assert abs(metrics.ic) < 0.1


def test_a_signal_built_from_tomorrows_return_would_score_perfectly(panel: Panel) -> None:
    """The control for the test above: the machinery can detect a perfect
    predictor, so a near-zero IC means alignment rather than a broken metric."""
    cheating = np.asarray(forward_returns(panel, 1), dtype=DTYPE)
    ranked_forward = rank_forward(panel, 1)
    metrics, _ = summarise(cheating, 0, panel, ranked_forward)
    assert metrics.ic > 0.99


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_rank_ic_of_a_perfect_ordering_is_one() -> None:
    n = 40
    signal = rank_rows(np.arange(n, dtype=DTYPE)[None, :])
    assert daily_rank_ic(signal, signal)[0] == pytest.approx(1.0, abs=1e-6)


def test_rank_ic_of_a_reversed_ordering_is_minus_one() -> None:
    n = 40
    signal = rank_rows(np.arange(n, dtype=DTYPE)[None, :])
    assert daily_rank_ic(signal, -signal)[0] == pytest.approx(-1.0, abs=1e-6)


def test_a_thin_day_is_not_scored() -> None:
    """A correlation over a handful of names is mostly an artefact of which
    names happened to be listed."""
    row = np.full((1, 40), np.nan, dtype=DTYPE)
    row[0, :5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    ranked = rank_rows(row)
    assert np.isnan(daily_rank_ic(ranked, ranked)[0])


def test_turnover_of_an_unchanging_signal_is_zero() -> None:
    ranked = rank_rows(np.tile(np.arange(30, dtype=DTYPE), (10, 1)))
    assert turnover(ranked) == pytest.approx(0.0, abs=1e-6)


def test_turnover_of_a_complete_reversal_is_one() -> None:
    """Half the sum of absolute weight changes, so a full flip is 1.0."""
    forward = np.arange(30, dtype=DTYPE)
    values = np.stack([forward, forward[::-1]])
    assert turnover(rank_rows(values)) == pytest.approx(1.0, abs=1e-6)


def test_coverage_is_measured_against_listed_cells() -> None:
    """Dividing by the shape of the array would score every signal as poor,
    because a panel is mostly empty at its corners."""
    panel = make_panel(close=[[1.0, 2.0], [3.0, 4.0]])
    values = np.array([[1.0, NAN], [2.0, 3.0]], dtype=DTYPE)
    assert coverage(values, 0, panel) == pytest.approx(0.75)


def test_coverage_ignores_warmup_rows() -> None:
    panel = make_panel(close=[[1.0, 2.0], [3.0, 4.0]])
    values = np.array([[NAN, NAN], [2.0, 3.0]], dtype=DTYPE)
    assert coverage(values, 1, panel) == pytest.approx(1.0)


def test_dispersion_of_a_flat_signal_is_zero() -> None:
    flat = np.zeros((5, 40), dtype=DTYPE)
    assert dispersion(flat) == pytest.approx(0.0, abs=1e-6)


def test_dispersion_of_a_spread_signal_is_positive() -> None:
    spread = rank_rows(np.tile(np.arange(40, dtype=DTYPE), (5, 1)))
    assert dispersion(spread) > 0.2


# --------------------------------------------------------------------------
# the screen
# --------------------------------------------------------------------------


def make_screen(panel: Panel, registry: Registry, **overrides) -> Screen:
    config = ScreenConfig(**overrides)
    run = registry.start_run(config.as_config(), data_snapshot_id="test")
    return Screen(panel, registry, run.id, config)


def test_a_good_signal_passes_and_is_logged(panel: Panel, registry: Registry) -> None:
    screen = make_screen(panel, registry)
    outcome = screen.screen(from_string("div(ts_delay(close, 20), ts_delay(close, 250))"))
    assert outcome.verdict is Verdict.PASSED
    assert outcome.logged
    assert outcome.metrics is not None and outcome.metrics.ic > 0.0
    assert registry.trial_count(screen.run_id) == 1


def test_a_monotone_respelling_collapses_without_a_row(panel: Panel, registry: Registry) -> None:
    """x and zscore(x) are one hypothesis. Two rows would inflate the N behind
    the Deflated Sharpe with a trial that had no independent chance."""
    screen = make_screen(panel, registry)
    base = "div(ts_delay(close, 20), ts_delay(close, 250))"
    first = screen.screen(from_string(base))
    second = screen.screen(from_string(f"zscore({base})"))

    assert first.logged and not second.logged
    assert second.kill_reason is KillReason.DUPLICATE
    assert first.value_hash == second.value_hash
    assert registry.trial_count(screen.run_id) == 1
    assert screen.report.semantic_collapses == 1


def test_a_structural_duplicate_never_reaches_the_panel(panel: Panel, registry: Registry) -> None:
    screen = make_screen(panel, registry)
    tree = from_string("ts_mean(close, 20)")
    screen.screen(tree)
    again = screen.screen(tree)
    assert not again.logged
    assert again.kill_reason is KillReason.DUPLICATE
    assert screen.report.structural_duplicates == 1
    assert registry.trial_count(screen.run_id) == 1


def test_the_root_rank_is_elided(panel: Panel, registry: Registry) -> None:
    screen = make_screen(panel, registry)
    screen.screen(from_string("rank(ts_mean(close, 20))"))
    assert screen.report.elided_nodes == 1


def test_a_flat_signal_is_killed_as_degenerate(panel: Panel, registry: Registry) -> None:
    screen = make_screen(panel, registry)
    outcome = screen.screen(from_string("sign(abs(close))"))
    assert outcome.verdict is Verdict.KILLED
    assert outcome.kill_reason is KillReason.DEGENERATE


def test_a_signal_with_no_ic_is_killed(panel: Panel, registry: Registry) -> None:
    screen = make_screen(panel, registry, min_abs_ic=0.99)
    outcome = screen.screen(from_string("ts_mean(close, 20)"))
    assert outcome.kill_reason is KillReason.LOW_IC


def test_a_churning_signal_is_killed(panel: Panel, registry: Registry) -> None:
    screen = make_screen(panel, registry, max_turnover=0.0001, min_abs_ic=0.0, min_ic_ir=0.0)
    outcome = screen.screen(from_string("returns"))
    assert outcome.kill_reason is KillReason.HIGH_TURNOVER


def test_a_sparse_signal_is_killed_on_coverage(panel: Panel, registry: Registry) -> None:
    screen = make_screen(panel, registry, min_coverage=0.999)
    outcome = screen.screen(from_string("ts_mean(close, 250)"))
    assert outcome.kill_reason is KillReason.LOW_COVERAGE


def test_a_kill_still_earns_a_row(panel: Panel, registry: Registry) -> None:
    """It looked at the data, so it was a trial."""
    screen = make_screen(panel, registry, min_abs_ic=0.99)
    outcome = screen.screen(from_string("ts_mean(close, 20)"))
    assert outcome.logged
    assert registry.trial_count(screen.run_id) == 1


def test_the_ledger_holds_one_row_per_distinct_hypothesis(
    panel: Panel, registry: Registry
) -> None:
    screen = make_screen(panel, registry)
    rng = np.random.default_rng(5)
    for _ in range(60):
        screen.screen(random_tree(rng, 4))

    rows = registry.trial_count(screen.run_id)
    assert rows == screen.report.logged
    assert rows == screen.report.seen - (
        screen.report.structural_duplicates + screen.report.semantic_collapses
    )


def test_metrics_reach_the_ledger(panel: Panel, registry: Registry) -> None:
    screen = make_screen(panel, registry)
    screen.screen(from_string("div(ts_delay(close, 20), ts_delay(close, 250))"))
    row = registry._conn.execute(
        "SELECT ic, ic_ir, turnover, value_hash FROM trials"
    ).fetchone()
    assert row["ic"] is not None
    assert row["ic_ir"] is not None
    assert row["turnover"] is not None
    assert len(row["value_hash"]) == 32


def test_a_nan_metric_is_stored_as_null(panel: Panel, registry: Registry) -> None:
    """NaN is not a measurement, and a NULL column says so rather than
    letting a later average quietly swallow it."""
    screen = make_screen(panel, registry)
    screen.screen(from_string("sign(abs(close))"))
    row = registry._conn.execute("SELECT ic FROM trials").fetchone()
    assert row["ic"] is None


def test_the_screen_config_reaches_the_run_config(panel: Panel, registry: Registry) -> None:
    """Thresholds decide which candidates survive, so two runs with different
    values are not comparable and the ledger has to say which was used."""
    screen = make_screen(panel, registry, min_abs_ic=0.033)
    stored = registry.run(screen.run_id).config
    assert stored["screen_min_abs_ic"] == 0.033
    assert stored["screen_horizon"] == 1


def test_screening_is_reproducible(panel: Panel, registry: Registry) -> None:
    first = make_screen(panel, registry)
    second = make_screen(panel, registry)
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(25):
        a = first.screen(random_tree(rng_a, 4))
        b = second.screen(random_tree(rng_b, 4))
        assert a.expr_hash == b.expr_hash
        assert a.verdict is b.verdict
        assert a.value_hash == b.value_hash
