"""Bottom-up, pure-numpy evaluation of expression trees.

Returns an ``EvalResult`` carrying the values, the warmup, and a scalar
degradation counter per node.

Warmup
------
Rows before ``warmup`` are not data. They are reported, never silently
returned. In practice they come out NaN on their own, because every windowed
operator writes nothing before its window first closes, and elementwise
operators inherit that NaN from their children. The evaluator asserts that
rather than trusting it, since a row of plausible numbers where the warmup
says there should be none is the exact shape of a look-ahead bug.

Degradation
-----------
Every windowed node reports one float: the fraction of the cells it emitted
that were computed from a partial window. That is all the screen needs, and it
is roughly ten floats for a depth-4 tree instead of ten planes.

The obvious alternative, a per-cell fill plane at the root, is not merely
coarse. It is wrong. In ``ts_mean(ts_mean(close, 20), 250)`` the outer mean
reads the inner mean's output, which is non-NaN wherever the inner window
cleared its floor, so a root-level fill would report a full window while the
data underneath it was thin. Composing fill correctly would mean carrying a
plane up through every node, which costs exactly what per-subtree planes cost.
So the real choice was between cheap and wrong or expensive and right, and a
per-node scalar is a third option that is cheap and right for the question
actually being asked.

The screen kills on ``max(degradation)``, and attribution comes free: the
index of the maximum is the node responsible. The evaluator is
memory-bandwidth-bound, so every extra plane roughly halves throughput, and
paying that on a million candidates to obtain information needed for a
thousand is the wrong trade. ``debug_fill=True`` re-evaluates with full
per-cell planes for the rare survivor that needs one. Evaluation is
deterministic, so the replay is exact.

Caching
-------
Subtrees are cached by structural hash within a batch. Because commutative
arguments are ordered canonically at construction, ``add(close, open)`` and
``add(open, close)`` share a cache entry as well as a trial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Callable, Final, Mapping, Sequence

import numpy as np

from alpha.data.types import DTYPE, Panel
from alpha.expr import grammar, kernels
from alpha.expr.ast import Node
from alpha.expr.grammar import MIN_GROUP_SIZE, WindowPolicy
from alpha.logging_config import get_logger

_log = get_logger(__name__)

ZSCORE_DDOF: Final[int] = 1
"""Sample standard deviation across the cross-section, matching ts_std."""


class EvaluationError(RuntimeError):
    """A tree could not be evaluated against this panel."""


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Values, warmup and per-node degradation for one tree.

    ``degradation`` and ``node_ops`` are aligned with ``tree.nodes()``, which
    is a stable parents-before-children ordering, so index i in either refers
    to the same node the tree reports at index i.
    """

    values: np.ndarray
    warmup: int
    degradation: tuple[float, ...]
    node_ops: tuple[str, ...]
    clamped: int = 0
    """Times a rolling variance came out negative and was clamped to zero.
    In float64 this should be zero; anything else is worth investigating."""

    non_finite: int = 0
    """Cells that overflowed to an infinity and were replaced with NaN.

    float32 tops out around 3.4e38, and a nested product reaches it on
    plausible fields. An infinity would survive every downstream operator and
    surface only as a nonsense IC, so it is converted, and counted rather than
    ignored: a NaN from an overflow means something different from a NaN
    meaning there was nothing to trade.

    The conversion happens at the root, once per tree, and that placement has
    a consequence worth stating rather than leaving implied. An infinity
    formed inside a windowed operator is read by that operator's window for
    the next d rows, so one overflowed cell contaminates up to d output cells
    before the root ever sees it, and the root then converts all of them. The
    count therefore measures contaminated output, not the number of
    overflows.

    That is accepted rather than overlooked. Overflow needs a product of two
    values near 1e19 on fields that do not reach it, the inflated NaN count
    kills the candidate on coverage regardless, and guarding every node would
    cost a pass over 30MB per node in a design that is already
    memory-bandwidth-bound. If the count ever fires routinely, the right
    answer is a guard inside the operators that can overflow, not a wider net
    at the root."""

    fill: Mapping[int, np.ndarray] = field(default_factory=dict)
    """Per-cell window fill by node index, only when ``debug_fill=True``."""

    @property
    def max_degradation(self) -> float:
        """What the screen kills on."""
        return max(self.degradation, default=0.0)

    def worst_node(self) -> tuple[int, str, float]:
        """Index, operator and degradation of the worst offender.

        Attribution to a specific operator, which is what makes a degradation
        kill actionable rather than just a rejection.
        """
        if not self.degradation:
            return (0, "", 0.0)
        index = int(np.argmax(self.degradation))
        return (index, self.node_ops[index], self.degradation[index])


# --------------------------------------------------------------------------
# semantic identity
# --------------------------------------------------------------------------


def value_hash(ranked: np.ndarray) -> str:
    """Hash of a signal's cross-sectional ranks, not of its raw values.

    Two expressions with the same hash are the same hypothesis, and the
    registry should hold one row for them rather than two.

    Spearman IC is invariant under any strictly monotone per-day transform, so
    ``x``, ``rank(x)``, ``zscore(x)`` and ``log(x)`` all rank identically and
    all produce the same IC, the same deciles and the same turnover. They are
    one idea spelled four ways. Hashing raw values would record four trials;
    hashing ranks records one.

    That is the cheapest reduction in the multiple-testing burden available,
    because N enters the Deflated Sharpe through ``sqrt(2 ln N)``: nothing
    else removes trials at zero cost to the search.

    ``sign(x)`` deliberately does not collide. It is monotone but not
    strictly so, and coarsening a continuous signal to three levels changes
    the ranks, the deciles and the IC. It is a different hypothesis and gets
    its own row.

    A strictly decreasing transform also does not collide, because it reverses
    the ranks rather than preserving them. Whether ``x`` and its negation
    should count as one hypothesis is a separate question, and collapsing them
    would mean canonicalising the sign of every signal, which is a decision
    about what a signal means rather than a hashing detail.

    The caller passes ranks that it already needed. The screen ranks each
    candidate once for the IC and reuses that array here, so the hash is free
    rather than a second pass over the panel.
    """
    array = np.ascontiguousarray(ranked, dtype=DTYPE)
    # NaN has many bit patterns and a division can produce the negative one,
    # so the missingness pattern is canonicalised before hashing. Otherwise
    # two identical signals could differ in the sign bit of a NaN.
    missing = np.isnan(array)
    if missing.any():
        array = np.where(missing, np.float32(np.nan), array)
    digest = sha256()
    digest.update(f"{array.shape[0]}x{array.shape[1]}|".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()[:32]


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def evaluate(tree: Node, panel: Panel, *, debug_fill: bool = False) -> EvalResult:
    """Evaluate one tree against a panel."""
    return evaluate_batch((tree,), panel, debug_fill=debug_fill)[0]


def evaluate_batch(
    trees: Sequence[Node], panel: Panel, *, debug_fill: bool = False
) -> tuple[EvalResult, ...]:
    """Evaluate several trees, sharing one subtree cache.

    The cache is local to the call. It is not an attribute of anything, so two
    batches cannot silently share state and a batch cannot grow without bound
    across a run.
    """
    cache: dict[str, np.ndarray] = {}
    local: dict[str, tuple[float, int]] = {}
    return tuple(_evaluate_one(tree, panel, cache, local, debug_fill) for tree in trees)


def _evaluate_one(
    tree: Node,
    panel: Panel,
    cache: dict[str, np.ndarray],
    local: dict[str, tuple[float, int]],
    debug_fill: bool,
) -> EvalResult:
    fill: dict[int, np.ndarray] = {}
    values = _compute(tree, panel, cache, local)

    nodes = tree.nodes()
    degradation = tuple(local.get(node.structural_hash(), (0.0, 0))[0] for node in nodes)
    clamped = sum(local.get(node.structural_hash(), (0.0, 0))[1] for node in nodes)

    if debug_fill:
        for index, node in enumerate(nodes):
            policy = node.spec.window_policy
            if policy in (WindowPolicy.NONE, WindowPolicy.POINT):
                continue
            child = cache[node.children[0].structural_hash()]
            fill[index] = kernels.window_fill(
                _contiguous(child), node.params[0], node.warmup
            )

    if clamped:
        _log.warning(
            "rolling variance clamped at zero",
            extra={"expr": str(tree), "clamped_cells": clamped},
        )

    values, non_finite = _finite_only(values)
    if non_finite:
        _log.warning(
            "overflow to infinity replaced with NaN",
            extra={"expr": str(tree), "cells": non_finite},
        )

    _assert_warmup_is_empty(tree, values)
    return EvalResult(
        values=values,
        warmup=tree.warmup,
        degradation=degradation,
        node_ops=tuple(node.op for node in nodes),
        clamped=clamped,
        non_finite=non_finite,
        fill=fill,
    )


def _finite_only(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace infinities with NaN at the root, and say how many there were.

    Done once per tree rather than once per node. Guarding every operator
    would cost a pass over 30MB at every node in a design that is already
    memory-bandwidth-bound, and an infinity that appears mid-tree either
    survives to the root or turns into a NaN on the way, so the root catches
    what matters.
    """
    infinite = np.isinf(values)
    count = int(np.count_nonzero(infinite))
    if count:
        values = values.copy()
        values[infinite] = np.nan
    return values, count


def _assert_warmup_is_empty(tree: Node, values: np.ndarray) -> None:
    """Rows before warmup must hold nothing.

    Every windowed operator writes nothing before its window first closes and
    elementwise operators inherit that NaN, so this should hold by
    construction. It is asserted because a row of plausible numbers where the
    warmup says there should be none is precisely what a look-ahead bug looks
    like, and the assertion costs one pass over a slice that is usually small.
    """
    head = values[: tree.warmup]
    if head.size and not np.isnan(head).all():
        rows = np.flatnonzero(~np.isnan(head).all(axis=1))
        raise EvaluationError(
            f"{tree} has warmup {tree.warmup} but produced values at rows "
            f"{rows[:5].tolist()}. Those rows could not have been computed "
            f"from in-window data, so something read forward."
        )


# --------------------------------------------------------------------------
# recursion
# --------------------------------------------------------------------------


def _compute(
    node: Node,
    panel: Panel,
    cache: dict[str, np.ndarray],
    local: dict[str, tuple[float, int]],
) -> np.ndarray:
    """Bottom up: children first, then the node, memoised by structural hash."""
    key = node.structural_hash()
    hit = cache.get(key)
    if hit is not None:
        return hit

    if node.is_terminal:
        values = _field(node, panel)
    else:
        children = [_compute(child, panel, cache, local) for child in node.children]
        values, degradation, clamped = _apply(node, children)
        local[key] = (degradation, clamped)

    cache[key] = values
    return values


def _field(node: Node, panel: Panel) -> np.ndarray:
    try:
        return panel[node.op]
    except KeyError as exc:
        raise EvaluationError(f"panel has no field for terminal {node.op!r}") from exc


def _apply(node: Node, children: list[np.ndarray]) -> tuple[np.ndarray, float, int]:
    """Dispatch one operator. Returns values, local degradation and clamps."""
    spec = node.spec
    if spec.window_policy not in (WindowPolicy.NONE, WindowPolicy.POINT):
        return _windowed(node, children)

    handler = _NUMPY_OPS.get(node.op)
    if handler is None:
        raise EvaluationError(f"no implementation for operator {node.op!r}")
    return handler(node, children), 0.0, 0


def _windowed(node: Node, children: list[np.ndarray]) -> tuple[np.ndarray, float, int]:
    """Run a kernel and turn its counters into one degradation fraction."""
    d = node.params[0]
    floor = grammar.min_periods(node.spec, d)
    assert floor is not None  # guaranteed by the window policy
    # The node's warmup, not the row where this window first closes. See the
    # kernels module: min_periods alone would let a node emit into rows its
    # own warmup declares invalid.
    first_row = node.warmup

    if node.op == "ts_rank":
        # The only kernel that scans its window, so it wants each instrument's
        # history contiguous rather than each day's cross-section.
        transposed = np.ascontiguousarray(children[0].T)
        out, degraded, considered, clamped = kernels.ts_rank(
            transposed, d, floor, first_row
        )
        values = np.ascontiguousarray(out.T)
    elif node.op == "correlation":
        out, degraded, considered, clamped = kernels.correlation(
            _contiguous(children[0]), _contiguous(children[1]), d, floor, first_row
        )
        values = out
    else:
        kernel = _KERNELS[node.op]
        out, degraded, considered, clamped = kernel(
            _contiguous(children[0]), d, floor, first_row
        )
        values = out

    fraction = float(degraded) / considered if considered else 0.0
    return values, fraction, int(clamped)


def _contiguous(array: np.ndarray) -> np.ndarray:
    """Kernels want C-contiguous float32. Panel planes already are."""
    if array.dtype == DTYPE and array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array, dtype=DTYPE)


_KERNELS: Final[Mapping[str, Callable]] = {
    "ts_mean": kernels.ts_mean,
    "ts_std": kernels.ts_std,
    "ts_min": kernels.ts_min,
    "ts_max": kernels.ts_max,
    "ts_argmax": kernels.ts_argmax,
    "decay_linear": kernels.decay_linear,
}


# --------------------------------------------------------------------------
# elementwise
# --------------------------------------------------------------------------


def _add(node: Node, c: list[np.ndarray]) -> np.ndarray:
    # Overflow is handled once at the root, where an infinity becomes a
    # counted NaN, so it does not need to warn at every node on the way.
    with np.errstate(over="ignore", invalid="ignore"):
        return (c[0] + c[1]).astype(DTYPE, copy=False)


def _sub(node: Node, c: list[np.ndarray]) -> np.ndarray:
    # Overflow is handled once at the root, where an infinity becomes a
    # counted NaN, so it does not need to warn at every node on the way.
    with np.errstate(over="ignore", invalid="ignore"):
        return (c[0] - c[1]).astype(DTYPE, copy=False)


def _mul(node: Node, c: list[np.ndarray]) -> np.ndarray:
    # Overflow is handled once at the root, where an infinity becomes a
    # counted NaN, so it does not need to warn at every node on the way.
    with np.errstate(over="ignore", invalid="ignore"):
        return (c[0] * c[1]).astype(DTYPE, copy=False)


def _div(node: Node, c: list[np.ndarray]) -> np.ndarray:
    """x / y, with a vanishing denominator giving NaN rather than an infinity.

    An infinity survives every downstream operator and surfaces only as a
    nonsense IC. A NaN is caught by the coverage checks.
    """
    numerator, denominator = c
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / np.where(
            np.abs(denominator) <= grammar.DIV_EPS, np.nan, denominator
        )
    return out.astype(DTYPE, copy=False)


def _natural_log(node: Node, c: list[np.ndarray]) -> np.ndarray:
    """Natural log, with a non-positive input giving NaN rather than -inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.where(c[0] > 0.0, c[0], np.nan))
    return out.astype(DTYPE, copy=False)


def _abs(node: Node, c: list[np.ndarray]) -> np.ndarray:
    return np.abs(c[0]).astype(DTYPE, copy=False)


def _sign(node: Node, c: list[np.ndarray]) -> np.ndarray:
    return np.sign(c[0]).astype(DTYPE, copy=False)


# --------------------------------------------------------------------------
# point lookups
# --------------------------------------------------------------------------


def _ts_delay(node: Node, c: list[np.ndarray]) -> np.ndarray:
    """x[t - d]. The first d rows have nothing to read."""
    d = node.params[0]
    out = np.full_like(c[0], np.nan, dtype=DTYPE)
    if d < out.shape[0]:
        out[d:] = c[0][:-d]
    return out


def _delta(node: Node, c: list[np.ndarray]) -> np.ndarray:
    """x[t] - x[t - d]. Either endpoint missing gives NaN, whatever lies
    between them: this is a point lookup, not a window."""
    d = node.params[0]
    out = np.full_like(c[0], np.nan, dtype=DTYPE)
    if d < out.shape[0]:
        out[d:] = c[0][d:] - c[0][:-d]
    return out


# --------------------------------------------------------------------------
# cross-sectional
# --------------------------------------------------------------------------


def _rank(node: Node, c: list[np.ndarray]) -> np.ndarray:
    return rank_rows(c[0])


def _average_positions(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-day average ordinal position of each value, and the day's count.

    Position is 0-based among the row's non-NaN entries, and a run of equal
    values shares the average of the positions it spans. NaN cells come back
    NaN and are excluded from the count.

    Ties are averaged rather than broken by order of appearance. That is not
    a nicety: sign and ts_argmax produce heavily tied output, volume ties on
    round numbers, and low-priced names tie on price. Breaking ties by
    instrument index would make the result depend on column order, which is
    an arbitrary property of the panel, and the search would find it.

    Shared by rank and rank_within so the tie handling is written once. Kept
    in float32, because positions run to a few thousand and are exact there,
    and every full-size temporary on a real panel is 30MB of traffic.
    """
    n_days, n_inst = x.shape
    counts = np.count_nonzero(~np.isnan(x), axis=1).astype(np.float32)

    order = np.argsort(x, axis=1, kind="stable")
    sorted_values = np.take_along_axis(x, order, axis=1)
    positions = np.arange(n_inst, dtype=np.float32)

    # A run boundary is where the sorted value changes. NaN never equals
    # itself, so trailing NaNs each form their own run and fall out below.
    changed = np.empty((n_days, n_inst), dtype=bool)
    changed[:, 0] = True
    np.not_equal(sorted_values[:, 1:], sorted_values[:, :-1], out=changed[:, 1:])

    run_start = np.where(changed, positions, np.float32(-1.0))
    np.maximum.accumulate(run_start, axis=1, out=run_start)

    ends = np.empty((n_days, n_inst), dtype=bool)
    ends[:, -1] = True
    ends[:, :-1] = changed[:, 1:]
    # Reversed accumulation on a contiguous copy. Accumulating through a
    # negative-stride view is markedly slower than paying for the copy.
    run_end = np.ascontiguousarray(
        np.where(ends, positions, np.float32(n_inst))[:, ::-1]
    )
    np.minimum.accumulate(run_end, axis=1, out=run_end)
    run_end = run_end[:, ::-1]

    average = run_start
    average += run_end
    average *= np.float32(0.5)

    scattered = np.empty((n_days, n_inst), dtype=DTYPE)
    np.put_along_axis(scattered, order, average, axis=1)
    scattered[np.isnan(x)] = np.nan
    return scattered, counts


def rank_rows(x: np.ndarray) -> np.ndarray:
    """Per-day centred rank, ties averaged, NaN excluded from the population.

    ``(position + 0.5) / n - 0.5`` over the day's non-NaN entries, so the row
    sums to zero and a ranked signal is already a zero-net-exposure score.
    """
    positions, counts = _average_positions(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        positions += np.float32(0.5)
        positions /= np.where(counts > 0, counts, np.float32(1.0))[:, None]
        positions -= np.float32(0.5)
    positions[counts == 0] = np.nan
    return positions


def _zscore(node: Node, c: list[np.ndarray]) -> np.ndarray:
    """Per-day (x - mean) / std across instruments.

    A day with no cross-sectional dispersion is NaN, not zero. There is no
    information to normalise, and zero would be a fabricated neutral score
    that later aggregates would average in as though it meant something.
    """
    x = c[0]
    # Moments come from nan-aware sums rather than nanmean and nanstd, which
    # warn on a day where nothing is listed, and an empty day is an ordinary
    # case in a panel with listings rather than an anomaly. The two-pass form
    # also cannot produce a negative variance, so unlike the rolling kernels
    # there is no clamp to reason about here.
    #
    # Temporaries are reused in place. Each full-size array is 30MB on a real
    # panel, and this operator is called once per cross-sectional node, so an
    # extra one is an extra pass over main memory rather than an extra name.
    missing = np.isnan(x)
    counts = np.count_nonzero(~missing, axis=1, keepdims=True).astype(np.float64)

    work = np.where(missing, 0.0, x).astype(np.float64)
    mean = work.sum(axis=1, keepdims=True) / np.maximum(counts, 1.0)

    np.subtract(x, mean, out=work, where=~missing)
    work[missing] = 0.0
    variance = np.einsum("ij,ij->i", work, work)[:, None] / np.maximum(
        counts - ZSCORE_DDOF, 1.0
    )
    std = np.sqrt(variance)

    usable = (counts > ZSCORE_DDOF) & (std > 0.0)
    np.divide(work, np.where(usable, std, 1.0), out=work)
    out = work.astype(DTYPE, copy=False)
    out[missing] = np.nan
    out[~np.broadcast_to(usable, x.shape)] = np.nan
    return out


def _group_keys(x: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Flat (row, group) keys for the cells both planes define.

    A null group label is excluded rather than pooled. Pooling the
    unclassified into a synthetic group would demean a company against an
    arbitrary set that shares nothing except that the vendor failed to label
    them, producing neutrality against a group that does not exist.
    """
    valid = ~np.isnan(x) & ~np.isnan(groups)
    if not valid.any():
        return valid, np.zeros(0, dtype=np.int64), 0
    labels = groups[valid].astype(np.int64)
    span = int(labels.max()) + 1
    rows = np.broadcast_to(
        np.arange(x.shape[0], dtype=np.int64)[:, None], x.shape
    )[valid]
    return valid, rows * span + labels, span


def _demean_by(node: Node, c: list[np.ndarray]) -> np.ndarray:
    """Subtract the per-day, per-group mean, leaving what is not the group bet."""
    x, groups = c
    out = np.full(x.shape, np.nan, dtype=DTYPE)
    valid, keys, span = _group_keys(x, groups)
    if span == 0:
        return out

    size = x.shape[0] * span
    totals = np.bincount(keys, weights=x[valid].astype(np.float64), minlength=size)
    counts = np.bincount(keys, minlength=size)
    with np.errstate(invalid="ignore"):
        means = np.where(
            counts >= MIN_GROUP_SIZE, totals / np.maximum(counts, 1), np.nan
        )
    out[valid] = (x[valid] - means[keys]).astype(DTYPE, copy=False)
    return out


def _rank_within(node: Node, c: list[np.ndarray]) -> np.ndarray:
    """Centred rank within the instrument's own group that day.

    Two per-row passes rather than one lexicographic sort over every defined
    cell. The old form sorted seven and a half million (key, value) pairs
    globally and cost seconds; this sorts rows, which is what numpy is fast
    at, and the group structure is recovered arithmetically.

    Rank the row once to get each value's position, map that to (0, 1), and
    add the integer group label. The result lies strictly inside
    ``(g, g + 1)``, so ordering the composite orders by group first and by
    value within a group, and equal values inside a group keep an equal
    composite so they tie again. Ranking that composite gives each cell its
    position in a row laid out group by group, and subtracting where its own
    group starts leaves its position within the group.

    A group below ``MIN_GROUP_SIZE`` is NaN: a centred rank over two names is
    always plus or minus 0.25 whatever the values are.
    """
    x, groups = c
    out = np.full(x.shape, np.nan, dtype=DTYPE)
    valid, keys, span = _group_keys(x, groups)
    if span == 0:
        return out

    within_row, counts = _average_positions(np.where(valid, x, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = (within_row + np.float32(0.5)) / np.where(
            counts > 0, counts, np.float32(1.0)
        )[:, None]

    composite = np.where(valid, groups + fraction, np.nan).astype(DTYPE, copy=False)
    across_row, _ = _average_positions(composite)

    # Where each group's block starts in that row, and how big it is.
    size = x.shape[0] * span
    per_group = np.bincount(keys, minlength=size).reshape(x.shape[0], span)
    block_start = np.cumsum(per_group, axis=1) - per_group

    labels = groups[valid].astype(np.int64)
    rows = np.broadcast_to(
        np.arange(x.shape[0], dtype=np.int64)[:, None], x.shape
    )[valid]
    starts = block_start[rows, labels].astype(np.float32)
    sizes = per_group[rows, labels].astype(np.float32)

    within_group = across_row[valid] - starts
    with np.errstate(invalid="ignore", divide="ignore"):
        ranked = np.where(
            sizes >= MIN_GROUP_SIZE,
            (within_group + np.float32(0.5)) / sizes - np.float32(0.5),
            np.nan,
        )
    out[valid] = ranked.astype(DTYPE, copy=False)
    return out


_NUMPY_OPS: Final[Mapping[str, Callable[[Node, list[np.ndarray]], np.ndarray]]] = {
    "add": _add,
    "sub": _sub,
    "mul": _mul,
    "div": _div,
    "log": _natural_log,
    "abs": _abs,
    "sign": _sign,
    "ts_delay": _ts_delay,
    "delta": _delta,
    "rank": _rank,
    "zscore": _zscore,
    "demean_by": _demean_by,
    "rank_within": _rank_within,
}


def _assert_complete() -> None:
    """Every non-terminal operator has an implementation.

    Checked at import, because the alternative is discovering a missing
    operator when the generator happens to draw it, which could be an hour
    into a run.
    """
    missing = []
    for name, spec in grammar.OPS.items():
        if spec.is_terminal:
            continue
        windowed = spec.window_policy not in (WindowPolicy.NONE, WindowPolicy.POINT)
        if windowed:
            if name not in _KERNELS and name not in ("ts_rank", "correlation"):
                missing.append(name)
        elif name not in _NUMPY_OPS:
            missing.append(name)
    if missing:
        raise AssertionError(f"operators with no implementation: {sorted(missing)}")


_assert_complete()
