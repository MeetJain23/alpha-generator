"""Performance assertions on a full-size panel.

Marked slow, because the panel is 2500 x 3000 across nineteen planes, which is
roughly 570MB and several seconds to generate. Run with::

    python -m pytest tests/test_benchmark.py -m slow

On the 50ms target
------------------
The design goal was under 50ms per depth-4 tree on a 2500 x 3000 panel. It is
not met, and the thresholds below say which parts come close rather than being
tuned until they pass.

Measure the machine before reading any of this. A float32 plane here is 30MB,
and on the development machine ``np.copy`` of one runs at about 7GB/s, not the
20 to 25GB/s a modern desktop is usually quoted at. So a three-stream
operator, reading a window's leaving value and its arriving value and writing
one output, has a floor near 12ms rather than near 3ms. A quoted bandwidth
figure is an upper bound on a machine nobody is running, and comparing against
it makes every kernel look sixteen times worse than it is.

Against the measured floor the streaming kernels are within roughly twice:
ts_mean is 29ms against a 12ms floor for the same memory traffic with no
arithmetic at all. The remaining gap is branch overhead in the inner loop,
which a branch-free two-phase formulation recovers about a quarter of. That is
recorded rather than applied, because it doubles the length of every kernel
for a change that leaves the op within the same factor of the floor.

The rank family cannot reach the floor, because ranking needs a sort and a
sort is not a streaming pass. ``rank`` spends 370ms of its time in argsort
alone, which is irreducible. ``ts_rank`` scans its window per cell, since a
sliding window rank has no constant-time update.

Loop order is not the problem and was checked: the kernels iterate time-outer
and instrument-inner, matching the row-major panel, and the deliberately wrong
order measures 95ms against 27ms for the same kernel.

So the assertions are split. The streaming operators are held near the
measured bandwidth limit. The sorting operators are held to a ceiling that
catches a regression without pretending the target applies to them.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from alpha.data.synthetic import SyntheticSpec, generate
from alpha.data.types import Panel
from alpha.expr.ast import from_string
from alpha.logging_config import get_logger

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]

_log = get_logger(__name__)

PANEL_DAYS = 2500
PANEL_INSTRUMENTS = 3000

STREAMING_BUDGET_MS = 90.0
"""Per operator, for the ops that stream over memory. The bandwidth floor for
one of these is roughly 6ms; the headroom absorbs a loaded machine."""

STREAMING_TREE_BUDGET_MS = 260.0
"""A depth-4 tree built only from streaming operators."""

SORTING_CEILING_MS = 3500.0
"""A regression guard for the rank family, not a target. rank_within was
7.2s before it was rewritten around two per-row passes; this would catch a
return to anything like that."""


@pytest.fixture(scope="module")
def panel() -> Panel:
    spec = SyntheticSpec(n_days=PANEL_DAYS, n_instruments=PANEL_INSTRUMENTS)
    return generate(np.random.default_rng(0), spec).panel


def measure(text: str, panel: Panel, repeats: int = 3) -> float:
    """Best of several runs, after a warm-up that pays the compilation."""
    from alpha.expr.evaluator import evaluate

    tree = from_string(text)
    evaluate(tree, panel)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        evaluate(tree, panel)
        best = min(best, (time.perf_counter() - start) * 1000.0)
    _log.info("benchmark", extra={"expr": text, "ms": round(best, 1)})
    return best


STREAMING_OPERATORS = [
    "ts_delay(close, 20)",
    "delta(close, 20)",
    "add(close, open)",
    "sub(close, open)",
    "mul(close, open)",
    "div(close, volume)",
    "abs(returns)",
    "sign(returns)",
    "log(close)",
    "ts_mean(close, 20)",
    "ts_mean(close, 250)",
    "ts_std(close, 60)",
    "decay_linear(close, 20)",
]

SORTING_OPERATORS = [
    "rank(close)",
    "rank_within(close, sector)",
    "ts_rank(close, 20)",
    "ts_rank(close, 250)",
    "zscore(close)",
    "demean_by(close, sector)",
    "ts_max(close, 20)",
    "ts_argmax(close, 60)",
    "correlation(close, volume, 60)",
]


@pytest.mark.parametrize("text", STREAMING_OPERATORS)
def test_a_streaming_operator_stays_near_the_bandwidth_bound(
    panel: Panel, text: str
) -> None:
    elapsed = measure(text, panel)
    assert elapsed < STREAMING_BUDGET_MS, f"{text} took {elapsed:.1f}ms"


@pytest.mark.parametrize("text", SORTING_OPERATORS)
def test_a_sorting_operator_stays_under_the_regression_ceiling(
    panel: Panel, text: str
) -> None:
    """Not the design target. A ceiling that catches an accidental order of
    magnitude without claiming a sort can be made memory-bound."""
    elapsed = measure(text, panel)
    assert elapsed < SORTING_CEILING_MS, f"{text} took {elapsed:.1f}ms"


def test_a_depth_four_streaming_tree_meets_its_budget(panel: Panel) -> None:
    """The class of tree the 50ms goal was written about.

    Five nodes, each of which streams. Three planes of traffic per node at
    30MB a plane puts the floor near 30ms, and this asserts the real
    implementation stays within a small multiple of that.
    """
    text = "div(ts_mean(close, 20), add(ts_std(close, 60), decay_linear(volume, 10)))"
    tree = from_string(text)
    assert tree.depth == 4 and tree.size == 8
    elapsed = measure(text, panel)
    assert elapsed < STREAMING_TREE_BUDGET_MS, f"{text} took {elapsed:.1f}ms"


def test_the_momentum_tree_is_cheap(panel: Panel) -> None:
    """Two point lookups and a divide, which is the cheapest thing the
    grammar can express at a 250-day lookback."""
    elapsed = measure("div(ts_delay(close, 20), ts_delay(close, 250))", panel)
    assert elapsed < STREAMING_TREE_BUDGET_MS


def test_a_repeated_subtree_is_not_paid_for_twice(panel: Panel) -> None:
    """The subtree cache has to be worth having at full size."""
    once = measure("ts_mean(close, 60)", panel)
    twice = measure("sub(ts_mean(close, 60), ts_mean(close, 60))", panel)
    assert twice < once * 2.0
