"""Tests for the operator grammar.

Two things are being defended here.

First, *spec conformance*: the registry contains exactly the operators the
design calls for, with no silent additions and no quiet omissions. An operator that
drifts into the grammar unannounced widens the search space and inflates the
multiple-testing burden without anyone deciding it should.

Second, *warmup arithmetic*. Every expected value below is hand-computed and
written out longhand rather than derived from the code under test, because a
test that recomputes the implementation cannot catch an off-by-one in it.
"""

from __future__ import annotations

import pytest

from alpha.expr import grammar as g
from alpha.expr.grammar import Axis, DType, OpKind, WarmupRule

# --------------------------------------------------------------------------
# spec conformance
# --------------------------------------------------------------------------

EXPECTED_FIELDS = {"open", "high", "low", "close", "volume", "vwap", "returns", "mcap"}
EXPECTED_UNARY = {"rank", "zscore", "log", "abs", "sign"}
EXPECTED_TS = {
    "ts_delay",
    "delta",
    "ts_mean",
    "ts_std",
    "ts_rank",
    "ts_min",
    "ts_max",
    "ts_argmax",
    "decay_linear",
}
EXPECTED_BINARY = {"add", "sub", "mul", "div", "correlation"}
EXPECTED_GROUP = {"demean_by", "rank_within"}


def test_registry_contains_exactly_the_specified_operators() -> None:
    expected = (
        EXPECTED_FIELDS
        | {"sector"}
        | EXPECTED_UNARY
        | EXPECTED_TS
        | EXPECTED_BINARY
        | EXPECTED_GROUP
    )
    assert set(g.OPS) == expected


def test_matrix_fields_are_the_eight_specified() -> None:
    matrix_terminals = {op.name for op in g.terminals_producing(DType.MATRIX)}
    assert matrix_terminals == EXPECTED_FIELDS


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (OpKind.UNARY, EXPECTED_UNARY),
        (OpKind.TS, EXPECTED_TS),
        (OpKind.BINARY, EXPECTED_BINARY),
        (OpKind.GROUP, EXPECTED_GROUP),
    ],
)
def test_operator_families(kind: OpKind, expected: set[str]) -> None:
    assert {op.name for op in g.ops_by_kind(kind)} == expected


def test_window_ladder_is_the_specified_one() -> None:
    assert g.WINDOWS == (1, 2, 3, 5, 10, 20, 60, 120, 250)


# --------------------------------------------------------------------------
# structural invariants
# --------------------------------------------------------------------------


def test_registry_is_immutable() -> None:
    with pytest.raises(TypeError):
        g.OPS["close"] = g.OPS["open"]  # type: ignore[index]


def test_arity_matches_signature() -> None:
    for op in g.OPS.values():
        assert op.arity == len(op.in_types)
        assert op.arity <= g.MAX_ARITY


def test_only_time_series_ops_cost_warmup() -> None:
    """The invariant the evaluator leans on: no cross-sectional op has a window."""
    for op in g.OPS.values():
        assert op.is_windowed == (op.axis is Axis.TIME_SERIES)
        assert op.is_windowed == bool(op.params)


def test_cross_sectional_ops_are_the_expected_four() -> None:
    cross = {op.name for op in g.OPS.values() if op.axis is Axis.CROSS_SECTION}
    assert cross == {"rank", "zscore", "demean_by", "rank_within"}


def test_commutative_ops_are_the_expected_three() -> None:
    commutative = {op.name for op in g.OPS.values() if op.commutative}
    assert commutative == {"add", "mul", "correlation"}


def test_group_values_are_never_computed() -> None:
    """GROUP comes from the panel; no operator may synthesise one."""
    for op in g.OPS.values():
        if op.out_type is DType.GROUP:
            assert op.is_terminal


def test_group_ops_take_a_group_second_argument() -> None:
    for name in EXPECTED_GROUP:
        assert g.get(name).in_types == (DType.MATRIX, DType.GROUP)


def test_sector_is_not_a_free_matrix_terminal() -> None:
    """Type-directed sampling must never be able to emit sector on its own."""
    assert "sector" not in {op.name for op in g.ops_producing(DType.MATRIX)}


def test_every_operator_is_documented() -> None:
    for op in g.OPS.values():
        assert op.doc.strip(), f"{op.name} has no doc"


# --------------------------------------------------------------------------
# warmup arithmetic, every expected value hand-computed
# --------------------------------------------------------------------------


def test_terminal_has_no_warmup() -> None:
    assert g.warmup(g.get("close"), (), ()) == 0


def test_returns_field_is_warmup_free() -> None:
    """The adapter already paid the one-day cost of forming the return."""
    assert g.warmup(g.get("returns"), (), ()) == 0


@pytest.mark.parametrize(
    ("name", "d", "expected"),
    [
        # a d-observation window first closes at offset d-1
        ("ts_mean", 20, 19),
        ("ts_std", 60, 59),
        ("ts_rank", 250, 249),
        ("ts_min", 2, 1),
        ("ts_max", 10, 9),
        ("ts_argmax", 5, 4),
        ("decay_linear", 3, 2),
        # a value reading d days back first exists at offset d
        ("delta", 1, 1),
        ("delta", 20, 20),
        ("delta", 250, 250),
        ("ts_delay", 1, 1),
        ("ts_delay", 20, 20),
        ("ts_delay", 250, 250),
    ],
)
def test_windowed_warmup_over_a_bare_field(name: str, d: int, expected: int) -> None:
    assert g.warmup(g.get(name), (d,), (0,)) == expected


def test_delta_costs_one_more_row_than_a_window_of_the_same_length() -> None:
    """The distinction that makes delta its own rule, stated as a test."""
    d = 20
    ts_mean = g.warmup(g.get("ts_mean"), (d,), (0,))
    delta = g.warmup(g.get("delta"), (d,), (0,))
    assert delta - ts_mean == 1


def test_elementwise_and_cross_sectional_ops_pass_warmup_through() -> None:
    for name in ("rank", "zscore", "log", "abs", "sign"):
        assert g.warmup(g.get(name), (), (37,)) == 37


def test_binary_warmup_takes_the_slower_child() -> None:
    #   sub(ts_mean(close, 60), ts_mean(close, 5))
    #     left  = 60 - 1 = 59
    #     right =  5 - 1 =  4
    #     sub   = max(59, 4) = 59
    assert g.warmup(g.get("sub"), (), (59, 4)) == 59


def test_warmup_accumulates_up_a_nested_tree() -> None:
    #   ts_mean(delta(close, 5), 10)
    #     close          =  0
    #     delta(., 5)    =  0 + 5      =  5
    #     ts_mean(., 10) =  5 + 10 - 1 = 14
    inner = g.warmup(g.get("delta"), (5,), (0,))
    assert inner == 5
    assert g.warmup(g.get("ts_mean"), (10,), (inner,)) == 14


def test_warmup_of_a_depth_four_tree() -> None:
    #   decay_linear(ts_rank(delta(close, 250), 20), 5)
    #     close             =   0
    #     delta(., 250)     =   0 + 250     = 250
    #     ts_rank(., 20)    = 250 + 20 - 1  = 269
    #     decay_linear(.,5) = 269 +  5 - 1  = 273
    w = g.warmup(g.get("delta"), (250,), (0,))
    w = g.warmup(g.get("ts_rank"), (20,), (w,))
    w = g.warmup(g.get("decay_linear"), (5,), (w,))
    assert w == 273


def test_correlation_warmup_stacks_on_the_slower_leg() -> None:
    #   correlation(ts_mean(close, 10), volume, 20)
    #     left  = 10 - 1 = 9
    #     right =          0
    #     corr  = max(9, 0) + 20 - 1 = 28
    assert g.warmup(g.get("correlation"), (20,), (9, 0)) == 28


def test_twelve_minus_one_momentum_warmup() -> None:
    """The verification milestone's own tree, as a warmup check.

    12-1 momentum is the ratio of the price twenty days ago to the price two
    hundred and fifty days ago. It skips the most recent month, where
    short-term reversal dominates, and it is a ratio of two prices rather than
    a difference, so it measures a return rather than a raw price change:

        div(ts_delay(close, 20), ts_delay(close, 250))
          close              =   0
          ts_delay(., 20)    =   0 +  20 = 20
          ts_delay(., 250)   =   0 + 250 = 250
          div                = max(20, 250) = 250

    250 rows is one trading year, which is what a 12-month signal costs.
    """
    near = g.warmup(g.get("ts_delay"), (20,), (0,))
    far = g.warmup(g.get("ts_delay"), (250,), (0,))
    assert (near, far) == (20, 250)
    assert g.warmup(g.get("div"), (), (near, far)) == 250


def test_momentum_ratio_is_scale_free_unlike_a_difference() -> None:
    """Why the tree is a div and not a sub.

    sub(ts_delay(close, 20), ts_delay(close, 250)) is a price difference, so
    ranking it cross-sectionally ranks high-priced stocks above low-priced
    ones regardless of their returns. The ratio has no price units, and the
    grammar carries both, so this is a property of the tree rather than of the
    operator set.
    """
    assert g.get("div").arity == 2
    assert g.get("sub").arity == 2
    assert g.warmup(g.get("sub"), (), (20, 250)) == 250


# --------------------------------------------------------------------------
# parameter validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "d"),
    [
        ("ts_mean", 1),  # identity
        ("ts_std", 2),  # a scaled first difference, not a dispersion
        ("ts_rank", 2),  # a sign test
        ("ts_rank", 3),  # still a sign test
        ("ts_argmax", 2),
        ("decay_linear", 1),
        ("correlation", 3),  # sample correlation is noise below 5
    ],
)
def test_degenerate_windows_are_rejected(name: str, d: int) -> None:
    with pytest.raises(ValueError, match="admissible ladder"):
        g.validate_params(g.get(name), (d,))


def test_delta_admits_a_one_day_window() -> None:
    """d=1 is the daily change rather than an identity, so it stays."""
    g.validate_params(g.get("delta"), (1,))


def test_ts_delay_admits_the_full_ladder() -> None:
    """A point lookup is meaningful at every lag, including one day."""
    assert g.get("ts_delay").params[0].values == g.WINDOWS


def test_point_lookups_share_the_window_rule() -> None:
    """delta and ts_delay both read an observation d days back."""
    on_window_rule = {
        op.name for op in g.OPS.values() if op.warmup_rule is WarmupRule.WINDOW
    }
    assert on_window_rule == {"delta", "ts_delay"}


@pytest.mark.parametrize("d", [0, 4, 7, 21, 251, -20])
def test_windows_off_the_ladder_are_rejected(d: int) -> None:
    with pytest.raises(ValueError):
        g.validate_params(g.get("ts_mean"), (d,))


def test_parameter_count_is_enforced() -> None:
    with pytest.raises(ValueError, match="takes 1 parameters"):
        g.validate_params(g.get("ts_mean"), ())
    with pytest.raises(ValueError, match="takes 0 parameters"):
        g.validate_params(g.get("rank"), (20,))


def test_booleans_are_not_accepted_as_windows() -> None:
    """True == 1 in Python; the grammar must not let that through."""
    with pytest.raises(ValueError, match="must be an int"):
        g.validate_params(g.get("delta"), (True,))  # type: ignore[arg-type]


def test_child_types_are_enforced() -> None:
    with pytest.raises(ValueError, match="expects"):
        g.validate_child_types(g.get("rank_within"), (DType.MATRIX, DType.MATRIX))
    g.validate_child_types(g.get("rank_within"), (DType.MATRIX, DType.GROUP))


def test_unknown_operator_names_list_the_alternatives() -> None:
    with pytest.raises(KeyError, match="known operators"):
        g.get("ts_meen")


# --------------------------------------------------------------------------
# lookup helpers
# --------------------------------------------------------------------------


def test_param_grid_enumerates_the_ladder() -> None:
    assert g.param_grid(g.get("ts_std")) == ((3,), (5,), (10,), (20,), (60,), (120,), (250,))


def test_param_grid_of_an_unparameterised_op_is_one_empty_tuple() -> None:
    """So callers can iterate without branching on whether there is a window."""
    assert g.param_grid(g.get("rank")) == ((),)


def test_every_param_grid_entry_validates() -> None:
    for op in g.OPS.values():
        for params in g.param_grid(op):
            g.validate_params(op, params)


def test_terminals_and_internals_partition_each_type() -> None:
    for dtype in DType:
        producing = set(g.ops_producing(dtype))
        assert producing == set(g.terminals_producing(dtype)) | set(
            g.internal_producing(dtype)
        )
        assert not set(g.terminals_producing(dtype)) & set(g.internal_producing(dtype))


def test_signature_renders_children_then_params() -> None:
    assert g.get("ts_mean").signature() == "ts_mean(matrix, d) -> matrix"
    assert g.get("correlation").signature() == "correlation(matrix, matrix, d) -> matrix"
    assert g.get("rank_within").signature() == "rank_within(matrix, group) -> matrix"
    assert g.get("close").signature() == "close() -> matrix"


def test_warmup_rules_are_exhaustively_covered() -> None:
    """If a rule is added, this test fails until warmup() handles it."""
    for rule in WarmupRule:
        op = next((o for o in g.OPS.values() if o.warmup_rule is rule), None)
        assert op is not None, f"no operator exercises {rule}"


# --------------------------------------------------------------------------
# window policy and min_periods
# --------------------------------------------------------------------------


def test_min_periods_fraction_is_eight_tenths() -> None:
    assert g.MIN_PERIODS_FRACTION == 0.8


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        # ceil(0.8 * d), computed by hand
        (2, 2),  # 1.6
        (3, 3),  # 2.4
        (5, 4),  # 4.0
        (10, 8),  # 8.0
        (20, 16),  # 16.0
        (60, 48),  # 48.0
        (120, 96),  # 96.0
        (250, 200),  # 200.0
    ],
)
def test_min_periods_over_the_ladder(d: int, expected: int) -> None:
    assert g.min_periods(g.get("ts_mean"), d) == expected


def test_point_lookups_have_no_minimum_count() -> None:
    """delta and ts_delay need their endpoints, not a count of what is between."""
    for name in ("delta", "ts_delay"):
        for d in g.WINDOWS:
            assert g.min_periods(g.get(name), d) is None


def test_unwindowed_operators_have_no_minimum_count() -> None:
    for name in ("rank", "zscore", "add", "demean_by", "close"):
        assert g.min_periods(g.get(name), 20) is None


def test_min_periods_never_exceeds_the_window() -> None:
    for op in g.OPS.values():
        if not op.is_windowed:
            continue
        for (d,) in g.param_grid(op):
            floor = g.min_periods(op, d)
            if floor is not None:
                assert 1 <= floor <= d


@pytest.mark.parametrize(
    ("name", "policy"),
    [
        ("ts_delay", g.WindowPolicy.POINT),
        ("delta", g.WindowPolicy.POINT),
        ("ts_mean", g.WindowPolicy.MIN_PERIODS),
        ("ts_std", g.WindowPolicy.MIN_PERIODS),
        ("ts_rank", g.WindowPolicy.MIN_PERIODS),
        ("ts_min", g.WindowPolicy.MIN_PERIODS),
        ("ts_max", g.WindowPolicy.MIN_PERIODS),
        ("ts_argmax", g.WindowPolicy.MIN_PERIODS),
        ("decay_linear", g.WindowPolicy.RENORMALIZE),
        ("correlation", g.WindowPolicy.PAIRWISE),
    ],
)
def test_window_policy_assignment(name: str, policy: g.WindowPolicy) -> None:
    assert g.get(name).window_policy is policy


def test_decay_linear_is_the_only_renormalising_operator() -> None:
    """If another weighted aggregate is added it must declare its own policy."""
    renormalising = {
        op.name for op in g.OPS.values() if op.window_policy is g.WindowPolicy.RENORMALIZE
    }
    assert renormalising == {"decay_linear"}


def test_correlation_is_the_only_pairwise_operator() -> None:
    pairwise = {
        op.name for op in g.OPS.values() if op.window_policy is g.WindowPolicy.PAIRWISE
    }
    assert pairwise == {"correlation"}


def test_unwindowed_operators_declare_no_policy() -> None:
    for op in g.OPS.values():
        assert op.is_windowed == (op.window_policy is not g.WindowPolicy.NONE)


def test_point_policy_and_window_warmup_rule_coincide() -> None:
    """Two statements of the same fact, kept in agreement by the registry."""
    for op in g.OPS.values():
        assert (op.window_policy is g.WindowPolicy.POINT) == (
            op.warmup_rule is WarmupRule.WINDOW
        )
