"""Tests for evaluation.

Every operator is checked against a hand-computed result on a small panel,
because a test that recomputes the implementation cannot catch a wrong
implementation. Beyond that, three properties carry the weight:

Nothing is emitted before warmup. Rows a node declares invalid must hold
nothing, or everything downstream treats them as data.

Prefix evaluation is bitwise identical. Evaluating on the first k rows must
equal evaluating on everything and taking the first k, exactly, not to a
tolerance.

NaN propagates and is never filled, and no operator produces an infinity.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alpha.data.types import DTYPE, FIELDS, Panel
from alpha.expr import grammar
from alpha.expr.ast import Node, from_string, random_tree
from alpha.expr.evaluator import EvaluationError, evaluate, evaluate_batch

NAN = np.nan


def make_panel(**planes: list[list[float]]) -> Panel:
    """A panel from literal columns, with every other field left as NaN."""
    first = next(iter(planes.values()))
    n_days, n_inst = len(first), len(first[0])
    fields: dict[str, np.ndarray] = {}
    for name in FIELDS:
        if name in planes:
            fields[name] = np.array(planes[name], dtype=DTYPE)
        else:
            fields[name] = np.full((n_days, n_inst), np.nan, dtype=DTYPE)
    fields["is_listed"] = np.ones((n_days, n_inst), dtype=DTYPE)
    return Panel(
        dates=pd.bdate_range("2020-01-01", periods=n_days, name="date"),
        instruments=np.array([f"A{i}" for i in range(n_inst)], dtype=object),
        fields=fields,
    )


def column(panel: Panel, text: str, j: int = 0) -> np.ndarray:
    return evaluate(from_string(text), panel).values[:, j]


def assert_close(got: np.ndarray, expected: list[float]) -> None:
    np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=2e-6, atol=1e-6)


# --------------------------------------------------------------------------
# elementwise operators
# --------------------------------------------------------------------------

SIMPLE = {"close": [[4.0], [1.0], [9.0], [NAN], [16.0]], "open": [[2.0], [2.0], [3.0], [1.0], [0.0]]}


def test_add() -> None:
    assert_close(column(make_panel(**SIMPLE), "add(close, open)"), [6, 3, 12, NAN, 16])


def test_sub() -> None:
    assert_close(column(make_panel(**SIMPLE), "sub(close, open)"), [2, -1, 6, NAN, 16])


def test_mul() -> None:
    assert_close(column(make_panel(**SIMPLE), "mul(close, open)"), [8, 2, 27, NAN, 0])


def test_div_guards_a_vanishing_denominator() -> None:
    """The last row divides by zero and must give NaN, not an infinity."""
    got = column(make_panel(**SIMPLE), "div(close, open)")
    assert_close(got[:4], [2.0, 0.5, 3.0, NAN])
    assert np.isnan(got[4])


def test_div_guard_uses_the_grammar_epsilon() -> None:
    tiny = grammar.DIV_EPS / 2
    panel = make_panel(close=[[1.0], [1.0]], open=[[tiny], [1.0]])
    got = column(panel, "div(close, open)")
    assert np.isnan(got[0])
    assert got[1] == pytest.approx(1.0)


def test_log_of_a_non_positive_value_is_nan_not_negative_infinity() -> None:
    panel = make_panel(close=[[math.e], [0.0], [-3.0], [1.0]])
    got = column(panel, "log(close)")
    assert_close(got[[0, 3]], [1.0, 0.0])
    assert np.isnan(got[1]) and np.isnan(got[2])


def test_abs() -> None:
    panel = make_panel(close=[[-2.0], [3.0], [NAN]])
    assert_close(column(panel, "abs(close)"), [2, 3, NAN])


def test_sign_keeps_nan() -> None:
    panel = make_panel(close=[[-2.0], [0.0], [5.0], [NAN]])
    assert_close(column(panel, "sign(close)"), [-1, 0, 1, NAN])


def test_no_result_carries_an_infinity() -> None:
    panel = make_panel(close=[[1.0], [0.0], [-1.0], [1e30]], open=[[0.0], [0.0], [0.0], [1e-30]])
    for text in ("div(close, open)", "log(close)", "mul(close, close)"):
        assert not np.isinf(column(panel, text)).any(), text


def test_an_overflow_becomes_nan_and_is_counted() -> None:
    """float32 tops out near 3.4e38 and a nested product reaches it. An
    infinity would survive every downstream operator and surface only as a
    nonsense IC, so it is converted, and counted so the conversion is not
    invisible."""
    panel = make_panel(close=[[1.0], [1e30]])
    result = evaluate(from_string("mul(close, close)"), panel)
    assert result.non_finite == 1
    assert np.isnan(result.values[1, 0])
    assert result.values[0, 0] == pytest.approx(1.0)


def test_an_ordinary_result_counts_no_overflow() -> None:
    panel = make_panel(close=[[2.0], [3.0]])
    assert evaluate(from_string("mul(close, close)"), panel).non_finite == 0


# --------------------------------------------------------------------------
# point lookups
# --------------------------------------------------------------------------


def test_ts_delay_reads_d_rows_back() -> None:
    panel = make_panel(close=[[1.0], [2.0], [3.0], [4.0], [5.0]])
    assert_close(column(panel, "ts_delay(close, 2)"), [NAN, NAN, 1, 2, 3])


def test_delta_is_the_difference_to_d_rows_back() -> None:
    panel = make_panel(close=[[1.0], [2.0], [4.0], [8.0], [16.0]])
    assert_close(column(panel, "delta(close, 2)"), [NAN, NAN, 3, 6, 12])


def test_a_point_lookup_needs_both_endpoints() -> None:
    """Exempt from min_periods: what lies between the endpoints is irrelevant,
    and either endpoint missing gives NaN."""
    panel = make_panel(close=[[1.0], [NAN], [NAN], [8.0], [16.0]])
    assert_close(column(panel, "delta(close, 3)"), [NAN, NAN, NAN, 7, NAN])


# --------------------------------------------------------------------------
# rolling kernels
# --------------------------------------------------------------------------


def test_ts_mean_over_a_full_window() -> None:
    panel = make_panel(close=[[1.0], [2.0], [3.0], [4.0], [5.0]])
    assert_close(column(panel, "ts_mean(close, 3)"), [NAN, NAN, 2, 3, 4])


def test_ts_mean_excludes_missing_observations() -> None:
    """Rows 2 and 3 hold two of three, which clears ceil(0.8 * 3) = 3? No:
    the floor is 3, so they are NaN, and row 4 has all three."""
    assert grammar.min_periods(grammar.get("ts_mean"), 3) == 3
    panel = make_panel(close=[[1.0], [NAN], [3.0], [5.0], [7.0]])
    got = column(panel, "ts_mean(close, 3)")
    assert np.isnan(got[2]) and np.isnan(got[3])
    assert got[4] == pytest.approx(5.0)


def test_ts_std_matches_the_sample_standard_deviation() -> None:
    panel = make_panel(close=[[1.0], [2.0], [3.0], [4.0], [8.0]])
    got = column(panel, "ts_std(close, 3)")
    assert_close(got[2:], [1.0, 1.0, float(np.std([3.0, 4.0, 8.0], ddof=1))])


def test_ts_std_of_a_constant_series_is_zero_not_nan() -> None:
    """The clamp must return zero, which is the answer, rather than a NaN
    manufactured by cancellation."""
    panel = make_panel(close=[[2.0]] * 8)
    got = column(panel, "ts_std(close, 3)")
    assert_close(got[2:], [0.0] * 6)


def test_ts_min_and_ts_max() -> None:
    panel = make_panel(close=[[3.0], [1.0], [4.0], [1.0], [5.0], [9.0]])
    assert_close(column(panel, "ts_min(close, 3)"), [NAN, NAN, 1, 1, 1, 1])
    assert_close(column(panel, "ts_max(close, 3)"), [NAN, NAN, 4, 4, 5, 9])


def test_ts_argmax_counts_rows_not_observations() -> None:
    """A maximum three rows back is 3 whether or not the rows between traded,
    or the same number would mean different elapsed times in different cells."""
    #   ceil(0.8 * 5) = 4, so the window needs four of its five rows present
    assert grammar.min_periods(grammar.get("ts_argmax"), 5) == 4
    panel = make_panel(close=[[9.0], [NAN], [1.0], [2.0], [3.0]])
    got = column(panel, "ts_argmax(close, 5)")
    assert got[4] == pytest.approx(4.0)


def test_ts_argmax_is_zero_when_today_is_the_maximum() -> None:
    panel = make_panel(close=[[1.0], [2.0], [3.0], [9.0]])
    assert_close(column(panel, "ts_argmax(close, 3)"), [NAN, NAN, 0, 0])


def test_decay_linear_weights_the_newest_observation_most() -> None:
    #   weights 3, 2, 1 on rows t, t-1, t-2 over [1, 2, 3]
    #   (3*3 + 2*2 + 1*1) / 6 = 14 / 6
    panel = make_panel(close=[[1.0], [2.0], [3.0]])
    assert_close(column(panel, "decay_linear(close, 3)")[2:], [14.0 / 6.0])


def test_decay_linear_renormalises_over_present_observations() -> None:
    """Otherwise a missing observation shrinks the result in proportion to how
    much data is absent, and missingness becomes a liquidity factor."""
    #   rows [2, NaN, 3] with weights 1, 2, 3: (1*2 + 3*3) / (1 + 3) = 11 / 4
    panel = make_panel(close=[[2.0], [NAN], [3.0]])
    assert grammar.min_periods(grammar.get("decay_linear"), 3) == 3
    # the floor of 3 rejects it, so widen the window to see the renormalisation
    panel = make_panel(close=[[2.0], [NAN], [3.0], [4.0], [5.0]])
    got = column(panel, "decay_linear(close, 5)")
    expected = (1 * 2.0 + 3 * 3.0 + 4 * 4.0 + 5 * 5.0) / (1 + 3 + 4 + 5)
    assert got[4] == pytest.approx(expected, rel=1e-6)


def test_correlation_of_a_series_with_itself_is_one() -> None:
    panel = make_panel(close=[[1.0], [3.0], [2.0], [7.0], [4.0], [9.0]], open=[[1.0], [3.0], [2.0], [7.0], [4.0], [9.0]])
    got = column(panel, "correlation(close, open, 5)")
    assert got[5] == pytest.approx(1.0, abs=1e-6)


def test_correlation_of_a_perfect_inverse_is_minus_one() -> None:
    values = [1.0, 3.0, 2.0, 7.0, 4.0, 9.0]
    panel = make_panel(
        close=[[v] for v in values], open=[[-v] for v in values]
    )
    got = column(panel, "correlation(close, open, 5)")
    assert got[5] == pytest.approx(-1.0, abs=1e-6)


def test_correlation_with_a_constant_leg_is_nan() -> None:
    """Zero variance means there is no correlation to report, and zero would
    claim there was one and that it was none."""
    panel = make_panel(
        close=[[1.0], [3.0], [2.0], [7.0], [4.0], [9.0]], open=[[5.0]] * 6
    )
    assert np.isnan(column(panel, "correlation(close, open, 5)")[5])


def test_ts_rank_normalises_by_the_present_count() -> None:
    """Dividing by d would cap a degraded window at m/d and damp the signal
    for exactly the halted names."""
    #   window of 5 rows holding 4 present values, today's is the largest:
    #   less = 3, equal = 1, m = 4  ->  (3 + 1) / 4 = 1.0
    panel = make_panel(close=[[1.0], [2.0], [NAN], [3.0], [9.0]])
    assert grammar.min_periods(grammar.get("ts_rank"), 5) == 4
    assert column(panel, "ts_rank(close, 5)")[4] == pytest.approx(1.0)


def test_ts_rank_averages_ties() -> None:
    #   [5, 5, 5, 5, 5]: less = 0, equal = 5, m = 5 -> (0 + 3) / 5 = 0.6
    panel = make_panel(close=[[5.0]] * 5)
    assert column(panel, "ts_rank(close, 5)")[4] == pytest.approx(0.6)


def test_ts_rank_of_the_smallest_value_is_positive() -> None:
    panel = make_panel(close=[[5.0], [4.0], [3.0], [2.0], [1.0]])
    got = column(panel, "ts_rank(close, 5)")[4]
    assert got == pytest.approx(0.2)


# --------------------------------------------------------------------------
# cross-sectional operators
# --------------------------------------------------------------------------


def test_rank_is_centred_and_sums_to_zero() -> None:
    #   four names: positions 0..3 -> (p + 0.5) / 4 - 0.5
    panel = make_panel(close=[[10.0, 20.0, 30.0, 40.0]])
    got = evaluate(from_string("rank(close)"), panel).values[0]
    assert_close(got, [-0.375, -0.125, 0.125, 0.375])
    assert float(got.sum()) == pytest.approx(0.0, abs=1e-6)


def test_rank_excludes_nan_from_the_population() -> None:
    #   three present names, so positions 0..2 over n = 3
    panel = make_panel(close=[[10.0, NAN, 30.0, 40.0]])
    got = evaluate(from_string("rank(close)"), panel).values[0]
    assert np.isnan(got[1])
    assert_close(got[[0, 2, 3]], [-1 / 3, 0.0, 1 / 3])


def test_rank_averages_ties() -> None:
    """Breaking ties by instrument order would turn the instrument axis into
    a signal, and sign and ts_argmax produce heavily tied output."""
    panel = make_panel(close=[[5.0, 5.0, 5.0, 9.0]])
    got = evaluate(from_string("rank(close)"), panel).values[0]
    assert got[0] == pytest.approx(got[1]) == pytest.approx(got[2])
    assert got[3] > got[0]
    assert float(got.sum()) == pytest.approx(0.0, abs=1e-6)


def test_rank_of_a_single_name_is_zero() -> None:
    panel = make_panel(close=[[7.0, NAN, NAN]])
    got = evaluate(from_string("rank(close)"), panel).values[0]
    assert got[0] == pytest.approx(0.0)


def test_zscore_is_centred_and_scaled() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    panel = make_panel(close=[values])
    got = evaluate(from_string("zscore(close)"), panel).values[0]
    expected = (np.array(values) - np.mean(values)) / np.std(values, ddof=1)
    assert_close(got, list(expected))


def test_zscore_of_a_day_with_no_dispersion_is_nan() -> None:
    """Zero would be a fabricated neutral score that aggregates would average
    in as though it meant something."""
    panel = make_panel(close=[[3.0, 3.0, 3.0]])
    assert np.isnan(evaluate(from_string("zscore(close)"), panel).values[0]).all()


def test_zscore_of_an_empty_day_is_nan_without_warning() -> None:
    panel = make_panel(close=[[NAN, NAN]])
    assert np.isnan(evaluate(from_string("zscore(close)"), panel).values[0]).all()


def test_demean_by_removes_the_group_mean() -> None:
    #   group 0: [1, 2, 3, 4, 5] mean 3 -> [-2, -1, 0, 1, 2]
    #   group 1: [10, 20, 30, 40, 50] mean 30 -> [-20, -10, 0, 10, 20]
    panel = make_panel(
        close=[[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]],
        sector=[[0.0] * 5 + [1.0] * 5],
    )
    got = evaluate(from_string("demean_by(close, sector)"), panel).values[0]
    assert_close(got, [-2, -1, 0, 1, 2, -20, -10, 0, 10, 20])


def test_a_null_group_label_gives_nan_with_no_residual_bucket() -> None:
    """Pooling the unclassified would demean a company against an arbitrary
    set that shares nothing except that the vendor failed to label them."""
    panel = make_panel(
        close=[[1.0, 2.0, 3.0, 4.0, 5.0, 99.0, 98.0]],
        sector=[[0.0] * 5 + [NAN, NAN]],
    )
    got = evaluate(from_string("demean_by(close, sector)"), panel).values[0]
    assert_close(got[:5], [-2, -1, 0, 1, 2])
    assert np.isnan(got[5]) and np.isnan(got[6])


def test_a_group_below_the_floor_gives_nan() -> None:
    """Demeaning two names returns half their difference and nothing else,
    which is an arithmetic identity rather than a measurement."""
    assert grammar.MIN_GROUP_SIZE == 5
    panel = make_panel(
        close=[[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]],
        sector=[[0.0] * 5 + [1.0, 1.0]],
    )
    got = evaluate(from_string("demean_by(close, sector)"), panel).values[0]
    assert_close(got[:5], [-2, -1, 0, 1, 2])
    assert np.isnan(got[5]) and np.isnan(got[6])


def test_rank_within_ranks_inside_the_group() -> None:
    #   five names per group, so positions 0..4 -> (p + 0.5) / 5 - 0.5
    expected = [-0.4, -0.2, 0.0, 0.2, 0.4]
    panel = make_panel(
        close=[[1.0, 2.0, 3.0, 4.0, 5.0, 50.0, 40.0, 30.0, 20.0, 10.0]],
        sector=[[0.0] * 5 + [1.0] * 5],
    )
    got = evaluate(from_string("rank_within(close, sector)"), panel).values[0]
    assert_close(got[:5], expected)
    assert_close(got[5:], expected[::-1])


def test_rank_within_a_group_below_the_floor_is_nan() -> None:
    """A centred rank over two names is always plus or minus 0.25 whatever the
    values are, and a constant is worse than noise downstream."""
    panel = make_panel(
        close=[[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]],
        sector=[[0.0] * 5 + [1.0, 1.0]],
    )
    got = evaluate(from_string("rank_within(close, sector)"), panel).values[0]
    assert np.isnan(got[5]) and np.isnan(got[6])
    assert np.isfinite(got[:5]).all()


def test_rank_within_averages_ties() -> None:
    #   [5, 5, 5, 1, 9]: the three fives share positions 1, 2, 3 -> average 2
    panel = make_panel(
        close=[[5.0, 5.0, 5.0, 1.0, 9.0]],
        sector=[[0.0] * 5],
    )
    got = evaluate(from_string("rank_within(close, sector)"), panel).values[0]
    assert got[0] == pytest.approx(got[1]) == pytest.approx(got[2])
    assert got[0] == pytest.approx((2.0 + 0.5) / 5.0 - 0.5)
    assert got[3] < got[0] < got[4]


def test_rank_within_handles_a_null_label() -> None:
    panel = make_panel(
        close=[[1.0, 2.0, 3.0, 4.0, 5.0, 10.0]],
        sector=[[0.0] * 5 + [NAN]],
    )
    got = evaluate(from_string("rank_within(close, sector)"), panel).values[0]
    assert np.isnan(got[5])
    assert np.isfinite(got[:5]).all()


def test_the_group_floor_kills_the_variance_clamp_at_source() -> None:
    """The clamp used to fire constantly because a two-member sector produced
    the same rank every day, giving a constant series whose true variance is
    zero."""
    rng = np.random.default_rng(2)
    values = rng.normal(size=(60, 7)).astype(np.float32)
    sectors = np.tile(np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.float32), (60, 1))
    panel = make_panel(close=values.tolist(), sector=sectors.tolist())
    result = evaluate(from_string("ts_std(rank_within(close, sector), 10)"), panel)
    assert result.clamped == 0
    assert np.isnan(result.values[:, 5]).all()
    assert np.isnan(result.values[:, 6]).all()


# --------------------------------------------------------------------------
# warmup
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel() -> Panel:
    from alpha.data.synthetic import SyntheticSpec, generate

    return generate(np.random.default_rng(20260918), SyntheticSpec(n_days=400, n_instruments=50)).panel


def test_nothing_is_emitted_before_warmup(panel: Panel) -> None:
    rng = np.random.default_rng(5)
    for _ in range(200):
        tree = random_tree(rng, 4)
        result = evaluate(tree, panel)
        assert np.isnan(result.values[: result.warmup]).all(), str(tree)


def test_the_warmup_reported_is_the_trees_own(panel: Panel) -> None:
    tree = from_string("ts_mean(ts_mean(close, 20), 60)")
    assert evaluate(tree, panel).warmup == tree.warmup == 78


def test_min_periods_cannot_emit_before_warmup(panel: Panel) -> None:
    """The interaction that a plain window-position guard gets wrong: the
    inner mean is NaN over its own warmup, so the outer window clears the 80
    per cent floor several rows before the composed warmup."""
    tree = from_string("ts_mean(ts_mean(close, 20), 60)")
    result = evaluate(tree, panel)
    assert np.isnan(result.values[:78]).all()
    assert np.isfinite(result.values[78:]).any()


def test_a_forward_reading_result_is_rejected(panel: Panel) -> None:
    """The assertion has to fire, or it is decoration."""
    import alpha.expr.evaluator as module

    tree = from_string("ts_mean(close, 20)")
    values = np.zeros((panel.n_days, panel.n_instruments), dtype=DTYPE)
    with pytest.raises(EvaluationError, match="read forward"):
        module._assert_warmup_is_empty(tree, values)


# --------------------------------------------------------------------------
# prefix equality
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "ts_mean(close, 20)",
        "ts_std(close, 60)",
        "ts_rank(volume, 20)",
        "ts_argmax(close, 20)",
        "decay_linear(close, 20)",
        "correlation(close, volume, 20)",
        "div(ts_delay(close, 20), ts_delay(close, 250))",
        "rank(demean_by(ts_mean(returns, 10), sector))",
    ],
)
def test_prefix_evaluation_is_bitwise_identical(panel: Panel, text: str) -> None:
    """Not to a tolerance. Prefix evaluation runs an identical sequence of
    operations, so any difference at all is a forward read or kernel state
    crossing the boundary, and a tolerance would mask exactly the small leaks
    that survive to production."""
    tree = from_string(text)
    full = evaluate(tree, panel).values
    for k in (120, 250, 399):
        prefix = evaluate(tree, panel.head(k)).values
        assert np.array_equal(prefix, full[:k], equal_nan=True), f"{text} at k={k}"


def test_prefix_equality_across_random_trees(panel: Panel) -> None:
    rng = np.random.default_rng(9)
    for _ in range(60):
        tree = random_tree(rng, 4)
        full = evaluate(tree, panel).values
        k = 275
        prefix = evaluate(tree, panel.head(k)).values
        assert np.array_equal(prefix, full[:k], equal_nan=True), str(tree)


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------


def test_a_clean_window_reports_no_degradation() -> None:
    panel = make_panel(close=[[float(i)] for i in range(40)])
    result = evaluate(from_string("ts_mean(close, 10)"), panel)
    assert result.max_degradation == 0.0


def test_gaps_raise_the_degradation() -> None:
    values = [[float(i)] for i in range(40)]
    for i in range(0, 40, 7):
        values[i] = [NAN]
    panel = make_panel(close=values)
    result = evaluate(from_string("ts_mean(close, 10)"), panel)
    assert result.max_degradation > 0.9


def test_degradation_is_one_entry_per_node() -> None:
    tree = from_string("ts_mean(ts_mean(close, 20), 60)")
    panel = make_panel(close=[[float(i)] for i in range(200)])
    result = evaluate(tree, panel)
    assert len(result.degradation) == tree.size == len(result.node_ops)
    assert result.node_ops == ("ts_mean", "ts_mean", "close")


def test_an_inner_node_degradation_is_visible_at_the_root() -> None:
    """The whole reason for per-node counters. A root-level fill plane would
    report a full window here, because the outer mean reads the inner mean's
    non-NaN output."""
    values = [[float(i)] for i in range(300)]
    for i in range(0, 300, 9):
        values[i] = [NAN]
    panel = make_panel(close=values)
    result = evaluate(from_string("ts_mean(ts_mean(close, 20), 60)"), panel)
    index, op, worst = result.worst_node()
    assert op == "ts_mean"
    assert worst > 0.5
    assert result.max_degradation == worst


def test_an_unwindowed_node_reports_no_degradation() -> None:
    panel = make_panel(close=[[1.0, 2.0], [3.0, 4.0]])
    result = evaluate(from_string("rank(close)"), panel)
    assert result.degradation == (0.0, 0.0)


def test_a_point_lookup_reports_no_degradation() -> None:
    """delta is exempt from the count rule, so it has nothing to degrade."""
    panel = make_panel(close=[[1.0], [NAN], [3.0], [4.0]])
    result = evaluate(from_string("delta(close, 2)"), panel)
    assert result.max_degradation == 0.0


def test_debug_fill_returns_planes_only_for_windowed_nodes() -> None:
    tree = from_string("ts_mean(rank(close), 10)")
    panel = make_panel(close=[[float(i), float(i * 2)] for i in range(40)])
    result = evaluate(tree, panel, debug_fill=True)
    assert set(result.fill) == {0}
    assert result.fill[0].shape == panel.shape
    assert not evaluate(tree, panel).fill


def test_debug_fill_agrees_with_the_scalar_counter() -> None:
    values = [[float(i)] for i in range(60)]
    for i in range(0, 60, 5):
        values[i] = [NAN]
    panel = make_panel(close=values)
    tree = from_string("ts_mean(close, 10)")
    result = evaluate(tree, panel, debug_fill=True)
    plane = result.fill[0]
    partial = np.isfinite(plane) & (plane < 1.0)
    reported = np.isfinite(result.values)
    assert result.max_degradation == pytest.approx(
        (partial & reported).sum() / reported.sum()
    )


# --------------------------------------------------------------------------
# caching and batching
# --------------------------------------------------------------------------


def test_a_repeated_subtree_is_computed_once(panel: Panel) -> None:
    """The cache key is the structural hash, so the same subtree appearing
    twice is one evaluation."""
    import alpha.expr.evaluator as module

    calls = 0
    original = module._apply

    def counting(node, children):
        nonlocal calls
        calls += 1
        return original(node, children)

    module._apply = counting
    try:
        evaluate(from_string("sub(ts_mean(close, 20), ts_mean(close, 20))"), panel)
    finally:
        module._apply = original
    assert calls == 2  # one ts_mean, one sub


def test_commutative_spellings_share_a_cache_entry(panel: Panel) -> None:
    first, second = from_string("add(close, volume)"), from_string("add(volume, close)")
    assert first.structural_hash() == second.structural_hash()
    results = evaluate_batch((first, second), panel)
    assert np.array_equal(results[0].values, results[1].values, equal_nan=True)


def test_a_batch_matches_evaluating_one_at_a_time(panel: Panel) -> None:
    rng = np.random.default_rng(13)
    trees = [random_tree(rng, 4) for _ in range(20)]
    batched = evaluate_batch(trees, panel)
    for tree, result in zip(trees, batched, strict=True):
        alone = evaluate(tree, panel)
        assert np.array_equal(alone.values, result.values, equal_nan=True)
        assert alone.degradation == result.degradation


def test_every_operator_has_an_implementation() -> None:
    """Checked at import too, but stated here so the failure names the test."""
    import alpha.expr.evaluator as module

    module._assert_complete()


def test_values_are_always_float32(panel: Panel) -> None:
    rng = np.random.default_rng(17)
    for _ in range(50):
        assert evaluate(random_tree(rng, 4), panel).values.dtype == DTYPE


def test_the_panel_is_never_modified(panel: Panel) -> None:
    before = {name: np.array(plane) for name, plane in panel.fields.items()}
    rng = np.random.default_rng(21)
    for _ in range(30):
        evaluate(random_tree(rng, 4), panel)
    for name, plane in before.items():
        assert np.array_equal(panel[name], plane, equal_nan=True)


# --------------------------------------------------------------------------
# semantic identity
# --------------------------------------------------------------------------


def hash_of(text: str, panel: Panel) -> str:
    from alpha.expr.evaluator import rank_rows, value_hash

    return value_hash(rank_rows(evaluate(from_string(text), panel).values))


@pytest.mark.parametrize(
    "text",
    [
        "rank(delta(close, 20))",
        "zscore(delta(close, 20))",
        "rank(rank(delta(close, 20)))",
        "rank(zscore(delta(close, 20)))",
    ],
)
def test_a_monotone_respelling_is_the_same_hypothesis(panel: Panel, text: str) -> None:
    """Spearman IC is invariant under a strictly monotone per-day transform,
    so these produce the same IC, deciles and turnover. One row, not four."""
    assert hash_of(text, panel) == hash_of("delta(close, 20)", panel)


def test_sign_is_not_the_same_hypothesis(panel: Panel) -> None:
    """Monotone but not strictly: coarsening to three levels changes the
    ranks, so it earns its own row."""
    assert hash_of("sign(delta(close, 20))", panel) != hash_of("delta(close, 20)", panel)


def test_a_different_window_is_a_different_hypothesis(panel: Panel) -> None:
    assert hash_of("delta(close, 20)", panel) != hash_of("delta(close, 60)", panel)


def test_a_reversed_signal_does_not_collide(panel: Panel) -> None:
    """A decreasing transform reverses the ranks rather than preserving them.
    Whether a signal and its negation are one hypothesis is a separate
    question about canonicalising sign."""
    assert hash_of("sub(close, open)", panel) != hash_of("sub(open, close)", panel)


def test_the_hash_is_stable_across_calls(panel: Panel) -> None:
    assert hash_of("ts_mean(close, 20)", panel) == hash_of("ts_mean(close, 20)", panel)


def test_the_hash_sees_the_missingness_pattern() -> None:
    """Two signals with the same order but different coverage are different,
    because deciles and turnover both depend on which cells are defined."""
    from alpha.expr.evaluator import rank_rows, value_hash

    full = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=DTYPE)
    holed = full.copy()
    holed[0, 2] = NAN
    assert value_hash(rank_rows(full)) != value_hash(rank_rows(holed))


def test_a_negative_nan_hashes_the_same_as_a_positive_one() -> None:
    """A division can produce the negative bit pattern, and two identical
    signals must not differ by the sign bit of a NaN."""
    from alpha.expr.evaluator import value_hash

    positive = np.array([[1.0, np.nan]], dtype=DTYPE)
    negative = np.array([[1.0, np.float32(-np.nan)]], dtype=DTYPE)
    assert value_hash(positive) == value_hash(negative)


# --------------------------------------------------------------------------
# root elision
# --------------------------------------------------------------------------


def test_eliding_the_root_rank_changes_nothing_the_screen_measures(panel: Panel) -> None:
    from alpha.expr.ast import strip_elidable_root
    from alpha.expr.evaluator import rank_rows, value_hash

    tree = from_string("rank(ts_mean(close, 20))")
    stripped, removed = strip_elidable_root(tree)
    assert removed == 1 and str(stripped) == "ts_mean(close, 20)"
    assert stripped.warmup == tree.warmup

    full = rank_rows(evaluate(tree, panel).values)
    elided = rank_rows(evaluate(stripped, panel).values)
    assert value_hash(full) == value_hash(elided)
    assert np.array_equal(full, elided, equal_nan=True)


def test_elision_strips_a_whole_chain(panel: Panel) -> None:
    from alpha.expr.ast import strip_elidable_root

    stripped, removed = strip_elidable_root(from_string("rank(rank(rank(close)))"))
    assert removed == 3 and str(stripped) == "close"


def test_elision_leaves_a_rank_that_is_not_at_the_root(panel: Panel) -> None:
    """ts_mean(rank(x), 20) averages ranks, which is not the rank of an
    average, so the inner rank is load-bearing."""
    from alpha.expr.ast import strip_elidable_root

    tree = from_string("ts_mean(rank(close), 20)")
    stripped, removed = strip_elidable_root(tree)
    assert removed == 0 and stripped == tree


def test_zscore_is_not_elidable(panel: Panel) -> None:
    """It preserves order but turns a zero-dispersion day into an all-NaN
    day, and the screen depends on which cells are defined."""
    from alpha.expr.ast import strip_elidable_root

    tree = from_string("zscore(close)")
    stripped, removed = strip_elidable_root(tree)
    assert removed == 0 and stripped == tree


def test_log_is_not_elidable() -> None:
    """It preserves order but drops every non-positive cell."""
    assert not grammar.get("log").elidable_at_root
    assert grammar.get("rank").elidable_at_root
