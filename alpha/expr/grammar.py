"""Operator grammar: the typed vocabulary every expression tree is built from.

This module is data, not behaviour. It declares what an operator is: arity,
input and output types, which axis it acts on, its legal parameter values, and
how much warmup it costs. It says nothing about how to compute it. ``ast.py``
reads it to generate and mutate well-typed trees; ``evaluator.py`` reads it to
dispatch and to accumulate warmup. There is exactly one registry, ``OPS``, and
it is immutable.

Axes
----
Every operator acts on exactly one axis of the (n_days x n_instruments) panel:

``Axis.NONE``           elementwise; shape-preserving, no coupling
``Axis.CROSS_SECTION``  per day, across instruments (rank, zscore, demean_by)
``Axis.TIME_SERIES``    per instrument, across time (every ts_* op, delta,
                        decay_linear, correlation)

Only ``TIME_SERIES`` ops look backwards, so only they cost warmup. The axis is
therefore a field on every operator, not a comment.

Warmup
------
``warmup(node)`` is the number of leading rows that are *not valid data*. Row
index ``warmup`` is the first row computed entirely from in-window
observations; rows ``[0, warmup)`` must be reported as invalid, never returned
as values. Warmup accumulates up the tree, because a child's warmup is the
floor for its parent's. That is why it is declared here per operator rather
than hardcoded in the evaluator.

Two rules cover every windowed operator:

``WarmupRule.WINDOW_MINUS_1``  a window of ``d`` observations first closes at
                              offset ``d - 1`` (ts_mean, ts_std, ts_rank,
                              ts_min, ts_max, ts_argmax, decay_linear,
                              correlation)
``WarmupRule.WINDOW``          a value that reads an observation ``d`` days
                              back first exists at offset ``d``
                              (delta, ts_delay)

Getting this off by one is the single largest look-ahead risk in the system.
It is declared once, here, and tested directly.

Why warmup is not relaxed to match min_periods
----------------------------------------------
``min_periods`` tolerates a window holding 80 per cent of its observations, so
it is tempting to let warmup end at ``ceil(0.8 * d)`` too and recover a fifth
of the history. That would be wrong, because the two answer different
questions.

``min_periods`` is about gaps in the middle of a window: the instrument
existed, it simply did not trade on some of those days, and the surrounding
observations still describe it. warmup is about absence of history at the
start: there is no earlier data because there was no earlier instrument.

Relaxing warmup would let a company that listed mid-panel start emitting once
80 per cent of its first window had accumulated, while a company listed
throughout waits for a full one. Every windowed operator would then carry a
listing-age term, recently listed names would differ systematically from
mature ones for a reason that has nothing to do with their returns, and the
search would find that difference and call it a factor. It is the same shape
as the decay_linear renormalisation bug, arriving through the time axis
instead of the weight vector.

The cost of the conservative choice is ``ceil(0.2 * d)`` rows at the start of
each instrument's life, which is a few weeks even at the longest window.

Normalisation
-------------
Cross-sectional ranks are centred on zero:

    rank = (position + 0.5) / n - 0.5,   range (-0.5, 0.5)

where ``position`` is the 0-based ascending position of the value among the
day's non-NaN entries, ties take the average position, and ``n`` is that
day's non-NaN count. The output sums to zero across the day, so a ranked
signal is already a zero-net-exposure score and needs no separate demeaning
before weighting.

Centring makes ``rank`` and ``zscore`` interchangeable along one dimension
only: robustness. Both produce a centred cross-sectional score, but ``rank``
bounds the influence of a value in the tail of the distribution while
``zscore`` lets that value set the scale for the whole day. Swapping one for
the other is a robustness choice, not a different hypothesis, and the
generator should price the pair as one idea rather than two.

``ts_rank`` is deliberately not centred. It is a within-instrument feature
that feeds into further operators, not a cross-sectional score, and centring
it would imply a neutrality across instruments that it does not have.

NaN policy
----------
NaN propagates. No operator fills, interpolates or drops. An instrument that
is not listed, is halted, or has no observation is NaN, and any expression
touching it is NaN on that cell. That is the correct answer, because there was
nothing to trade. Cross-sectional ops exclude NaN from the day's
population rather than treating it as a value; ``Domain`` records the two
places where a defined input still yields NaN.

A null group label is NaN, not a residual bucket: demean_by and rank_within
give NaN for any instrument the vendor has not classified that day.

Inside a rolling window, missing observations are excluded rather than
poisoning the whole window, subject to a floor of
``ceil(MIN_PERIODS_FRACTION * d)`` present values. ``WindowPolicy`` records
how each windowed operator does that, because the answer is not the same for
all of them: point lookups need their endpoints, decay_linear must renormalise
its weights over what is present, and correlation counts pairwise-complete
days.

Normalise by what is present, never by the window length
--------------------------------------------------------
Any operator that divides by a count must divide by ``m``, the number of
present observations in the window, and not by ``d``. This is one rule with
two instances so far, ``ts_rank`` and ``decay_linear``, and it is the same
mistake both times.

Dividing by ``d`` compresses a degraded window toward the middle of the
operator's output range: a ts_rank over 40 present observations in a 250-day
window would report at most 0.16 however extreme the value was, and a
decay_linear would report a fraction of its true magnitude. The compression is
not random. Missing observations concentrate in halted, thinly traded and
recently listed names, so the signal would be systematically damped for
exactly one group of instruments, and the search would discover that group as
a factor. A liquidity factor manufactured by an arithmetic slip is worse than
no factor at all, because it is real in the backtest and absent in the market.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import ceil
from types import MappingProxyType
from typing import Final, Mapping

# --------------------------------------------------------------------------
# type system
# --------------------------------------------------------------------------


class DType(Enum):
    """Value type flowing along an edge of the tree."""

    MATRIX = "matrix"
    """(n_days x n_instruments) float32 plane. Almost everything."""

    GROUP = "group"
    """(n_days x n_instruments) integer plane of group labels, e.g. sector.

    Never produced by an operator and never sampled as a free terminal: it
    exists only to be consumed as the second argument of a group op.
    """


class OpKind(Enum):
    """Coarse family, used for grouping in docs and sampling policy."""

    FIELD = "field"
    UNARY = "unary"
    TS = "ts"
    BINARY = "binary"
    GROUP = "group"


class Axis(Enum):
    """Which axis of the panel the operator couples along."""

    NONE = "none"
    CROSS_SECTION = "cross_section"
    TIME_SERIES = "time_series"


class WarmupRule(Enum):
    """How an operator adds to the warmup of its children."""

    NONE = "none"
    WINDOW_MINUS_1 = "window_minus_1"
    WINDOW = "window"


class Domain(Enum):
    """Inputs that are numerically defined but mathematically inadmissible."""

    ANY = "any"

    POSITIVE = "positive"
    """x <= 0 -> NaN (log)."""

    NONZERO_DENOM = "nonzero_denom"
    """|denominator| <= DIV_EPS -> NaN (div)."""


class WindowPolicy(Enum):
    """How a windowed operator handles missing observations inside its window.

    Every windowed operator needs an answer, and the answers are not the same,
    so the answer is declared per operator rather than assumed by the
    evaluator.
    """

    NONE = "none"
    """Not windowed. No policy applies."""

    POINT = "point"
    """A point lookup, exempt from the minimum-count rule.

    delta and ts_delay read specific observations rather than aggregating a
    window, so a count of present values in between is meaningless. If either
    endpoint is NaN the result is NaN, whatever lies between them.
    """

    MIN_PERIODS = "min_periods"
    """Standard rolling aggregate over present observations.

    The window must hold at least ``min_periods(op, d)`` non-NaN values, and
    the aggregate is computed over the values that are present.
    """

    RENORMALIZE = "renormalize"
    """A weighted aggregate that must rescale its weights over what is present.

    decay_linear only. If missing observations were treated as zero weight
    without renormalising, the result would be scaled down in proportion to
    how much data an instrument is missing, which turns missingness into a
    stealth liquidity factor: illiquid, frequently halted names would carry
    systematically smaller signal magnitudes for no reason anyone intended.
    """

    PAIRWISE = "pairwise"
    """A two-input aggregate counted on pairwise-complete observations.

    correlation only. A day counts toward the window only if both inputs are
    present on it, and the moments are accumulated over exactly those days.
    """


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

WINDOWS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10, 20, 60, 120, 250)
"""The only lookback lengths the grammar admits.

A coarse, roughly log-spaced ladder rather than every integer: adjacent
windows produce near-identical signals, so a dense ladder multiplies the
multiple-testing burden without adding hypotheses. 250 is about one trading
year; 20 about one month.
"""

DIV_EPS: Final[float] = 1e-10
"""Denominators with |y| <= DIV_EPS evaluate to NaN rather than +/-inf.

Single source of truth: the evaluator imports this rather than picking its own.
"""

MIN_PERIODS_FRACTION: Final[float] = 0.8
"""Fraction of a window that must be present for a rolling result to exist.

``min_periods = ceil(MIN_PERIODS_FRACTION * d)``.

A strict reading of "NaN propagates" would require the whole window, which
means one halted day voids a 250-day mean. That is not propagation so much as
amplification: a single missing observation destroys a year of otherwise
usable signal, and the names it happens to are exactly the ones a liquidity
screen would already treat with suspicion.

Excluding missing values from a rolling window is the same operation that
cross-sectional ops already perform on the day's population, and it introduces
no look-ahead: the window still ends at t. The floor is what keeps it honest,
by refusing a result computed from a handful of scattered observations.

This constant changes every signal in the system, so runs with different
values are not comparable. Layer 0 records it in ``runs.config_json``
alongside the git SHA.
"""

MIN_GROUP_SIZE: Final[int] = 5
"""Instruments a group must hold before a within-group statistic means anything.

demean_by and rank_within give NaN for any group smaller than this on a given
day. It is the cross-sectional counterpart of ``min_periods``, and it exists
for the same reason: a statistic computed from one or two observations is not
a weak measurement, it is an arithmetic identity wearing the costume of one.

A centred rank over two names is always exactly plus or minus 0.25 whatever
the values are. Over one name it is exactly zero. Demeaning a group of two
returns plus or minus half their difference. None of those carry information
about the instruments, and all of them are perfectly stable through time,
which is worse than noise: a constant series has zero variance, so anything
downstream measuring dispersion sees a degenerate input and the search sees a
signal that never changes.

Five is comfortably below any real GICS sector on a US panel, so on real data
this never binds. It binds on synthetic panels and on thin universes, which is
exactly where a silent arithmetic identity would otherwise be mistaken for a
result.
"""

MAX_ARITY: Final[int] = 2


def windows_at_least(minimum: int) -> tuple[int, ...]:
    """The ladder, restricted to windows a given operator can actually use."""
    return tuple(d for d in WINDOWS if d >= minimum)


# --------------------------------------------------------------------------
# operator records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """A scalar, non-tree operator parameter and its legal values.

    Parameters are drawn from a closed set, never a range: the search space
    stays enumerable and every trial is reproducible from its string form.
    """

    name: str
    values: tuple[int, ...]
    doc: str = ""

    def __contains__(self, value: int) -> bool:
        return value in self.values


@dataclass(frozen=True, slots=True)
class OpSpec:
    """Everything the engine needs to know about one operator.

    Frozen and hashable so specs can be used as dict keys and embedded in
    frozen ``Node`` instances without defensive copying.
    """

    name: str
    kind: OpKind
    in_types: tuple[DType, ...]
    out_type: DType
    axis: Axis
    params: tuple[ParamSpec, ...] = ()
    warmup_rule: WarmupRule = WarmupRule.NONE
    window_policy: WindowPolicy = WindowPolicy.NONE
    commutative: bool = False
    elidable_at_root: bool = False
    """Removing this operator from the root of a tree changes nothing that a
    rank-based screen measures.

    Requires two properties, not one. The operator must preserve the
    cross-sectional order within every day, and it must preserve the NaN
    pattern exactly. Rank IC, decile membership and turnover are all
    invariant under a strictly monotone per-day transform, but every one of
    them depends on which cells are defined.

    ``rank`` qualifies. ``zscore`` preserves order but turns a day with no
    dispersion into an all-NaN day, and ``log`` preserves order but drops
    every non-positive cell, so neither is exactly elidable even though both
    are monotone where they are defined.
    """

    domain: Domain = Domain.ANY
    weight: float = 1.0
    """Relative sampling weight for random tree generation. Tuning knob only;
    it carries no semantics."""
    doc: str = ""

    @property
    def arity(self) -> int:
        """Number of child subtrees (parameters are not children)."""
        return len(self.in_types)

    @property
    def is_terminal(self) -> bool:
        return self.arity == 0

    @property
    def is_windowed(self) -> bool:
        return self.warmup_rule is not WarmupRule.NONE

    def signature(self) -> str:
        """Human-readable signature, e.g. ``ts_mean(matrix, d) -> matrix``."""
        args = [t.value for t in self.in_types] + [p.name for p in self.params]
        return f"{self.name}({', '.join(args)}) -> {self.out_type.value}"


# --------------------------------------------------------------------------
# fields (terminals)
# --------------------------------------------------------------------------
#
# Price and volume fields are adjusted panels supplied by Layer 1; the
# grammar takes them as given. ``returns`` is the one-day simple return
# already aligned to its own date, so it carries no warmup here. The cost of
# forming it was paid by the adapter. Sampling weights lean toward the fields
# that carry tradable information (returns, close, volume) rather than the
# intraday extremes, which are mostly useful inside a ratio.

_FIELD_OPS: Final[tuple[OpSpec, ...]] = (
    OpSpec(
        name="open",
        kind=OpKind.FIELD,
        in_types=(),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        weight=0.8,
        doc="Adjusted opening price.",
    ),
    OpSpec(
        name="high",
        kind=OpKind.FIELD,
        in_types=(),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        weight=0.8,
        doc="Adjusted session high.",
    ),
    OpSpec(
        name="low",
        kind=OpKind.FIELD,
        in_types=(),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        weight=0.8,
        doc="Adjusted session low.",
    ),
    OpSpec(
        name="close",
        kind=OpKind.FIELD,
        in_types=(),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        weight=1.5,
        doc="Adjusted closing price.",
    ),
    OpSpec(
        name="volume",
        kind=OpKind.FIELD,
        in_types=(),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        weight=1.2,
        doc="Share volume, adjusted consistently with price.",
    ),
    OpSpec(
        name="vwap",
        kind=OpKind.FIELD,
        in_types=(),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        weight=1.0,
        doc="Volume-weighted average price for the session.",
    ),
    OpSpec(
        name="returns",
        kind=OpKind.FIELD,
        in_types=(),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        weight=1.5,
        doc="One-day simple return, as-of its own date. Warmup-free: the "
        "adapter already paid the one-day cost of forming it.",
    ),
    OpSpec(
        name="mcap",
        kind=OpKind.FIELD,
        in_types=(),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        weight=1.0,
        doc="Market capitalisation, as-of the panel date, never restated.",
    ),
)

_GROUP_FIELD_OPS: Final[tuple[OpSpec, ...]] = (
    OpSpec(
        name="sector",
        kind=OpKind.FIELD,
        in_types=(),
        out_type=DType.GROUP,
        axis=Axis.NONE,
        weight=1.0,
        doc="Integer sector label per instrument per day, as-of the panel "
        "date. Reclassifications apply from the date they happened, not "
        "retroactively. Consumed only by group ops; never sampled freely, "
        "because type-directed sampling asks for MATRIX producers.",
    ),
)


# --------------------------------------------------------------------------
# unary operators
# --------------------------------------------------------------------------
#
# ``rank`` and ``zscore`` are the normalisers that make a raw quantity
# comparable across instruments on a given day; they are cross-sectional and
# cost no warmup. ``log``, ``abs`` and ``sign`` are elementwise shape-fixers.

_UNARY_OPS: Final[tuple[OpSpec, ...]] = (
    OpSpec(
        name="rank",
        kind=OpKind.UNARY,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.CROSS_SECTION,
        elidable_at_root=True,
        weight=1.5,
        doc="Per-day cross-sectional rank across instruments, centred on "
        "zero: (position + 0.5) / n - 0.5, range (-0.5, 0.5). n is the day's "
        "non-NaN count, NaN cells are excluded from the population and stay "
        "NaN, and ties take the average position. The result sums to zero "
        "across the day, so it is already a zero-net-exposure score. Differs "
        "from zscore in robustness only, not in what it measures.",
    ),
    OpSpec(
        name="zscore",
        kind=OpKind.UNARY,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.CROSS_SECTION,
        weight=1.2,
        doc="Per-day (x - mean) / std across instruments, NaN excluded from "
        "both moments. A day with zero cross-sectional dispersion is NaN, "
        "not zero, because there is no information to normalise.",
    ),
    OpSpec(
        name="log",
        kind=OpKind.UNARY,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        domain=Domain.POSITIVE,
        weight=0.8,
        doc="Natural log. x <= 0 is NaN, not -inf: a non-positive price or "
        "volume is bad data, and propagating NaN says so.",
    ),
    OpSpec(
        name="abs",
        kind=OpKind.UNARY,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        weight=0.6,
        doc="Absolute value. Turns a signed spread into a magnitude.",
    ),
    OpSpec(
        name="sign",
        kind=OpKind.UNARY,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        weight=0.6,
        doc="-1 / 0 / +1, NaN preserved. Discards magnitude, keeps direction.",
    ),
)


# --------------------------------------------------------------------------
# time-series operators
# --------------------------------------------------------------------------
#
# These are the only operators that look backwards, and therefore the only
# ones that cost warmup.
#
# Warmup is *structural*: it is a count of row indices, derived from window
# lengths alone, and it is identical for every instrument. It is not a
# statement about data availability. An instrument that IPOs midway through
# the panel is NaN before its listing date; that is the NaN policy's job, and
# it is per-instrument. Conflating the two is how look-ahead gets in. Warmup
# says "this row could not have been computed yet". NaN says "there was
# nothing to compute it from".
#
# Per-operator window floors exist because the ladder's short end is
# degenerate for some ops: ts_mean(x, 1) is the identity, ts_std(x, 2) is
# |x_t - x_{t-1}| / sqrt(2) dressed up as a dispersion estimate, and
# ts_rank over three or four points is a sign test. Excluding them costs
# nothing and removes candidates that would burn trials rediscovering an
# operator the grammar already has.

_WINDOW_DOC: Final[str] = "Lookback length in trading days."


def _window(minimum: int) -> ParamSpec:
    """Window parameter restricted to the part of the ladder that is useful."""
    return ParamSpec(name="d", values=windows_at_least(minimum), doc=_WINDOW_DOC)


_TS_OPS: Final[tuple[OpSpec, ...]] = (
    OpSpec(
        name="ts_delay",
        kind=OpKind.TS,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.TIME_SERIES,
        params=(ParamSpec("d", WINDOWS, _WINDOW_DOC),),
        warmup_rule=WarmupRule.WINDOW,
        window_policy=WindowPolicy.POINT,
        weight=1.5,
        doc="x[t-d]. A point lookup, not a window: it reads one observation "
        "and copies it forward. On the WINDOW rule, because the value at "
        "offset d is the first that has a d-days-ago observation to read. "
        "Without this operator there is no way to express a signal that skips "
        "a period, which is what separates momentum from short-term reversal.",
    ),
    OpSpec(
        name="delta",
        kind=OpKind.TS,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.TIME_SERIES,
        params=(ParamSpec("d", WINDOWS, _WINDOW_DOC),),
        warmup_rule=WarmupRule.WINDOW,
        window_policy=WindowPolicy.POINT,
        weight=1.5,
        doc="x[t] - x[t-d]. On the WINDOW rule with ts_delay: a lag-d "
        "difference first exists at offset d, not d-1. d=1 is admissible "
        "here, because it is the daily change rather than an identity.",
    ),
    OpSpec(
        name="ts_mean",
        kind=OpKind.TS,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.TIME_SERIES,
        params=(_window(2),),
        warmup_rule=WarmupRule.WINDOW_MINUS_1,
        window_policy=WindowPolicy.MIN_PERIODS,
        weight=1.5,
        doc="Rolling mean over d days. d=1 excluded: it is the identity.",
    ),
    OpSpec(
        name="ts_std",
        kind=OpKind.TS,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.TIME_SERIES,
        params=(_window(3),),
        warmup_rule=WarmupRule.WINDOW_MINUS_1,
        window_policy=WindowPolicy.MIN_PERIODS,
        weight=1.2,
        doc="Rolling sample standard deviation (ddof=1) over d days. d<3 "
        "excluded: a two-point 'dispersion' is a scaled first difference.",
    ),
    OpSpec(
        name="ts_rank",
        kind=OpKind.TS,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.TIME_SERIES,
        params=(_window(5),),
        warmup_rule=WarmupRule.WINDOW_MINUS_1,
        window_policy=WindowPolicy.MIN_PERIODS,
        weight=1.2,
        doc="Percentile rank of x[t] among the present observations of its "
        "own trailing d-day window, in (0, 1]. Normalised by the present "
        "count m, never by the window length d: dividing by d would cap a "
        "degraded window's output at m/d and damp the signal for precisely "
        "the halted and thinly traded names, manufacturing a liquidity "
        "factor. Ties take the average position. Not centred, unlike the "
        "cross-sectional rank: this is a within-instrument feature, not a "
        "cross-sectional score. Self-normalising, so it survives regime "
        "shifts in level. Floor of 5, for the same reason as correlation: "
        "over three or four points the rank carries little more than the "
        "sign of a recent change, which sign(delta(x, d)) already expresses "
        "more cheaply.",
    ),
    OpSpec(
        name="ts_min",
        kind=OpKind.TS,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.TIME_SERIES,
        params=(_window(2),),
        warmup_rule=WarmupRule.WINDOW_MINUS_1,
        window_policy=WindowPolicy.MIN_PERIODS,
        weight=1.0,
        doc="Rolling minimum over d days.",
    ),
    OpSpec(
        name="ts_max",
        kind=OpKind.TS,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.TIME_SERIES,
        params=(_window(2),),
        warmup_rule=WarmupRule.WINDOW_MINUS_1,
        window_policy=WindowPolicy.MIN_PERIODS,
        weight=1.0,
        doc="Rolling maximum over d days.",
    ),
    OpSpec(
        name="ts_argmax",
        kind=OpKind.TS,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.TIME_SERIES,
        params=(_window(3),),
        warmup_rule=WarmupRule.WINDOW_MINUS_1,
        window_policy=WindowPolicy.MIN_PERIODS,
        weight=0.8,
        doc="Days since the window maximum, 0 = today, in [0, d-1]. Encodes "
        "recency of the extreme rather than its level. The index is counted "
        "in calendar positions within the window, not in present "
        "observations: a maximum five rows back is 5 whether or not the rows "
        "between were traded. Counting present observations instead would "
        "make the same number mean different elapsed times in different "
        "cells, and the values would stop being comparable across the "
        "cross-section, which is the only way this operator is ever used.",
    ),
    OpSpec(
        name="decay_linear",
        kind=OpKind.TS,
        in_types=(DType.MATRIX,),
        out_type=DType.MATRIX,
        axis=Axis.TIME_SERIES,
        params=(_window(2),),
        warmup_rule=WarmupRule.WINDOW_MINUS_1,
        window_policy=WindowPolicy.RENORMALIZE,
        weight=1.2,
        doc="Weighted mean over d days with weights d, d-1, ..., 1. The "
        "weights are renormalised over the observations actually present, "
        "not over the full window, for the same reason ts_rank divides by m: "
        "treating a missing observation as zero weight would shrink the "
        "result in proportion to how much data an instrument is missing and "
        "turn missingness into a liquidity factor. A smoother that keeps "
        "most of its mass on recent observations. The standard turnover "
        "damper.",
    ),
)


# --------------------------------------------------------------------------
# binary operators
# --------------------------------------------------------------------------
#
# ``commutative`` is not decoration: ast.py uses it to order the children of
# commutative nodes canonically before hashing, so add(close, open) and
# add(open, close) collide on one structural hash and cost one trial instead
# of two. With an append-only ledger, duplicate trials permanently inflate the
# N behind the Deflated Sharpe, so canonicalisation is a correctness concern
# rather than a tidiness one.

_BINARY_OPS: Final[tuple[OpSpec, ...]] = (
    OpSpec(
        name="add",
        kind=OpKind.BINARY,
        in_types=(DType.MATRIX, DType.MATRIX),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        commutative=True,
        weight=1.0,
        doc="Elementwise x + y.",
    ),
    OpSpec(
        name="sub",
        kind=OpKind.BINARY,
        in_types=(DType.MATRIX, DType.MATRIX),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        weight=1.2,
        doc="Elementwise x - y. The spread former; not commutative.",
    ),
    OpSpec(
        name="mul",
        kind=OpKind.BINARY,
        in_types=(DType.MATRIX, DType.MATRIX),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        commutative=True,
        weight=1.0,
        doc="Elementwise x * y.",
    ),
    OpSpec(
        name="div",
        kind=OpKind.BINARY,
        in_types=(DType.MATRIX, DType.MATRIX),
        out_type=DType.MATRIX,
        axis=Axis.NONE,
        domain=Domain.NONZERO_DENOM,
        weight=1.2,
        doc="Elementwise x / y, with |y| <= DIV_EPS mapped to NaN rather than "
        "+/-inf. An infinity would survive every downstream op and only "
        "surface as a nonsense IC; NaN is caught by the coverage checks.",
    ),
    OpSpec(
        name="correlation",
        kind=OpKind.BINARY,
        in_types=(DType.MATRIX, DType.MATRIX),
        out_type=DType.MATRIX,
        axis=Axis.TIME_SERIES,
        params=(_window(5),),
        warmup_rule=WarmupRule.WINDOW_MINUS_1,
        window_policy=WindowPolicy.PAIRWISE,
        commutative=True,
        weight=1.0,
        doc="Rolling Pearson correlation of x and y over their own trailing "
        "d-day windows, per instrument, in [-1, 1]. Windows below 5 are "
        "excluded: the sample correlation is almost pure noise there. A "
        "window in which either series has zero variance is NaN, not zero.",
    ),
)


# --------------------------------------------------------------------------
# group operators
# --------------------------------------------------------------------------
#
# Cross-sectional, but conditioned on a label: the population for a given
# instrument on a given day is its own sector that day, not the whole market.
# These are how a signal gets stripped of the sector bet it implicitly
# carries, which is usually the difference between an alpha and a
# repackaged sector tilt.
#
# The group argument is DType.GROUP, so it type-checks: rank_within(x, close)
# is not constructible, and a generator sampling by type cannot produce it.
#
# An instrument whose group label is null on a given day is NaN on that day.
# There is no residual bucket. Pooling the unlabelled into a synthetic group
# would mean demeaning a company against an arbitrary set of unrelated
# companies that share nothing except that the vendor did not classify them,
# and the resulting neutrality would be against a group that does not exist.
# Dropping the cell says the signal is undefined there, which is true.

_GROUP_OPS: Final[tuple[OpSpec, ...]] = (
    OpSpec(
        name="demean_by",
        kind=OpKind.GROUP,
        in_types=(DType.MATRIX, DType.GROUP),
        out_type=DType.MATRIX,
        axis=Axis.CROSS_SECTION,
        weight=1.2,
        doc="Subtract the per-day, per-group mean of x, leaving the part of "
        "the signal that is not the group bet. A null group label that day "
        "gives NaN: no residual bucket. A group holding fewer than "
        "MIN_GROUP_SIZE instruments that day also gives NaN, because "
        "demeaning two names returns half their difference and nothing else.",
    ),
    OpSpec(
        name="rank_within",
        kind=OpKind.GROUP,
        in_types=(DType.MATRIX, DType.GROUP),
        out_type=DType.MATRIX,
        axis=Axis.CROSS_SECTION,
        weight=1.2,
        doc="Rank of x within its group that day, centred on zero by the "
        "same formula as rank, with n the group's non-NaN count. Sums to zero "
        "within each group, so the result is neutral to the group as well as "
        "to the market. A null group label that day gives NaN: no residual "
        "bucket, and so does a group holding fewer than MIN_GROUP_SIZE "
        "instruments, because a centred rank over two names is always plus or "
        "minus 0.25 whatever the values are.",
    ),
)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_ALL_OPS: Final[tuple[OpSpec, ...]] = (
    _FIELD_OPS + _GROUP_FIELD_OPS + _UNARY_OPS + _TS_OPS + _BINARY_OPS + _GROUP_OPS
)

OPS: Final[Mapping[str, OpSpec]] = MappingProxyType({op.name: op for op in _ALL_OPS})
"""The operator registry. Immutable, and the only one. There is no global
mutable state anywhere in this system and no runtime registration hook."""


def get(name: str) -> OpSpec:
    """Look up an operator by name.

    Raises ``KeyError`` listing the available names, because the usual cause
    is a typo in a hand-written expression string.
    """
    try:
        return OPS[name]
    except KeyError:
        raise KeyError(
            f"unknown operator {name!r}; known operators: {sorted(OPS)}"
        ) from None


def ops_by_kind(kind: OpKind) -> tuple[OpSpec, ...]:
    """Every operator in one family, in declaration order."""
    return tuple(op for op in _ALL_OPS if op.kind is kind)


def ops_producing(dtype: DType) -> tuple[OpSpec, ...]:
    """Every operator whose result is ``dtype``.

    Type-directed generation is built on this: to fill a hole of type T, draw
    from ``ops_producing(T)``. It is what keeps randomly generated trees
    well-typed by construction rather than by rejection sampling.
    """
    return tuple(op for op in _ALL_OPS if op.out_type is dtype)


def terminals_producing(dtype: DType) -> tuple[OpSpec, ...]:
    """Leaf-eligible operators of a given type, the depth-limit fallback."""
    return tuple(op for op in ops_producing(dtype) if op.is_terminal)


def internal_producing(dtype: DType) -> tuple[OpSpec, ...]:
    """Operators of a given type that take children."""
    return tuple(op for op in ops_producing(dtype) if not op.is_terminal)


def param_grid(op: OpSpec) -> tuple[tuple[int, ...], ...]:
    """Every legal parameter tuple for an operator.

    One entry, the empty tuple, for unparameterised ops, so callers iterate
    uniformly without branching on whether an operator takes a window.
    """
    return tuple(product(*(p.values for p in op.params)))


def validate_params(op: OpSpec, params: tuple[int, ...]) -> None:
    """Raise ``ValueError`` unless ``params`` is legal for ``op``.

    Called on ``Node`` construction, so an ill-formed tree cannot exist. The
    ledger never records a trial that was never a valid hypothesis.
    """
    if len(params) != len(op.params):
        raise ValueError(
            f"{op.name} takes {len(op.params)} parameters, got {len(params)}: {params!r}"
        )
    for spec, value in zip(op.params, params, strict=True):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"{op.name}.{spec.name} must be an int, got {type(value).__name__}"
            )
        if value not in spec:
            raise ValueError(
                f"{op.name}.{spec.name}={value} is outside the admissible "
                f"ladder {spec.values}"
            )


def validate_child_types(op: OpSpec, child_types: tuple[DType, ...]) -> None:
    """Raise ``ValueError`` unless the children match the operator signature."""
    if child_types != op.in_types:
        expected = tuple(t.value for t in op.in_types)
        got = tuple(t.value for t in child_types)
        raise ValueError(f"{op.name} expects {expected}, got {got}")


def warmup(op: OpSpec, params: tuple[int, ...], child_warmups: tuple[int, ...]) -> int:
    """Rows of the result that are not valid data.

    The result row at index ``warmup`` is the first one computed entirely from
    in-window observations. A parent can never be valid earlier than its
    children, hence the max; a windowed parent additionally pays for its own
    lookback.

    This function is the whole of the warmup arithmetic in the system. The
    evaluator calls it and stores what it returns. It does not re-derive offsets
    per operator, because that is exactly where an off-by-one would hide.
    """
    base = max(child_warmups) if child_warmups else 0
    if op.warmup_rule is WarmupRule.NONE:
        return base
    validate_params(op, params)
    d = params[0]
    if op.warmup_rule is WarmupRule.WINDOW_MINUS_1:
        return base + d - 1
    if op.warmup_rule is WarmupRule.WINDOW:
        return base + d
    raise AssertionError(f"unhandled warmup rule {op.warmup_rule!r}")


def min_periods(op: OpSpec, d: int) -> int | None:
    """Non-NaN observations a window must hold for its result to exist.

    ``ceil(MIN_PERIODS_FRACTION * d)`` for aggregating operators. ``None`` for
    point lookups and for operators that are not windowed at all, where a
    count of present values is not the right question: delta and ts_delay need
    their two endpoints and nothing else.

    As with ``warmup()``, this is the only place the arithmetic happens. The
    evaluator reads it rather than recomputing a fraction per operator.
    """
    if op.window_policy in (WindowPolicy.NONE, WindowPolicy.POINT):
        return None
    validate_params(op, (d,))
    return ceil(MIN_PERIODS_FRACTION * d)


# --------------------------------------------------------------------------
# import-time invariants
# --------------------------------------------------------------------------


def _validate_registry() -> None:
    """Check the registry for internal consistency once, at import.

    These are the assumptions ast.py and evaluator.py are written against. A
    violation is a bug in this file, and failing at import is cheaper than
    discovering it as a malformed tree ten thousand trials into a run.
    """
    names = [op.name for op in _ALL_OPS]
    if len(names) != len(set(names)):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise AssertionError(f"duplicate operator names: {duplicates}")

    group_terminals = terminals_producing(DType.GROUP)

    for op in _ALL_OPS:
        if op.arity > MAX_ARITY:
            raise AssertionError(f"{op.name}: arity {op.arity} exceeds MAX_ARITY")
        if op.weight <= 0:
            raise AssertionError(f"{op.name}: sampling weight must be positive")

        # windowed <=> parameterised <=> time-series. All three, or none.
        flags = (op.is_windowed, bool(op.params), op.axis is Axis.TIME_SERIES)
        if len(set(flags)) != 1:
            raise AssertionError(
                f"{op.name}: windowed={flags[0]} parameterised={flags[1]} "
                f"time_series={flags[2]} must agree. Only backward-looking "
                f"ops take a window, and only they cost warmup"
            )

        if (op.window_policy is WindowPolicy.NONE) == op.is_windowed:
            raise AssertionError(
                f"{op.name}: a windowed operator needs a window policy and an "
                f"unwindowed one must not have one"
            )

        if (op.window_policy is WindowPolicy.POINT) != (
            op.warmup_rule is WarmupRule.WINDOW
        ):
            raise AssertionError(
                f"{op.name}: point lookups are exactly the operators on the "
                f"WINDOW warmup rule, because reading an observation d days "
                f"back is what both statements describe"
            )

        if op.window_policy is WindowPolicy.PAIRWISE and op.arity != 2:
            raise AssertionError(f"{op.name}: pairwise counting needs two inputs")

        for spec in op.params:
            if not spec.values:
                raise AssertionError(f"{op.name}.{spec.name}: empty value set")
            if len(set(spec.values)) != len(spec.values):
                raise AssertionError(f"{op.name}.{spec.name}: duplicate values")
            if tuple(sorted(spec.values)) != spec.values:
                raise AssertionError(f"{op.name}.{spec.name}: values must be sorted")
            unknown = set(spec.values) - set(WINDOWS)
            if unknown:
                raise AssertionError(
                    f"{op.name}.{spec.name}: {sorted(unknown)} is off the WINDOWS ladder"
                )

        if op.is_terminal:
            if op.axis is not Axis.NONE:
                raise AssertionError(f"{op.name}: a terminal couples along no axis")
            if op.kind is not OpKind.FIELD:
                raise AssertionError(f"{op.name}: only fields may be terminals")

        if op.out_type is DType.GROUP and not op.is_terminal:
            raise AssertionError(
                f"{op.name}: GROUP values come from the panel, they are never computed"
            )

        if op.elidable_at_root:
            if op.arity != 1 or op.out_type is not DType.MATRIX:
                raise AssertionError(
                    f"{op.name}: only a unary matrix operator can be elided"
                )
            if op.axis is not Axis.CROSS_SECTION:
                raise AssertionError(
                    f"{op.name}: only a cross-sectional operator preserves the "
                    f"order a rank-based screen measures"
                )

        if op.commutative:
            if op.arity != 2:
                raise AssertionError(f"{op.name}: commutative requires arity 2")
            if op.in_types[0] is not op.in_types[1]:
                raise AssertionError(
                    f"{op.name}: commutative requires interchangeable argument types"
                )

        if DType.GROUP in op.in_types and not group_terminals:
            raise AssertionError(
                f"{op.name}: consumes a GROUP but no GROUP terminal is declared"
            )

    if not terminals_producing(DType.MATRIX):
        raise AssertionError("no MATRIX terminals: no tree could ever be generated")


_validate_registry()
