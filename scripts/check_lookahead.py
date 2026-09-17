"""Falsify the no-look-ahead claim across 500 random trees.

For each tree, evaluating on ``panel.head(k)`` must give values identical to
evaluating on the full panel and slicing to ``[:k]``. Identical meaning equal
bit for bit, not close.

Why bitwise
-----------
Prefix evaluation runs the same operations in the same order on the same
inputs. Every kernel is a single forward pass whose state at row t depends
only on rows at or before t, so there is no reordering, no different
accumulation path, and nothing that could legitimately produce a different
last bit. Any difference at all is therefore a forward read, or kernel state
crossing the prefix boundary.

A tolerance would hide exactly the leaks worth finding. A signal that reads
one day ahead moves a value by far less than a relative 1e-6 on most cells,
and it is the leak that survives to production precisely because it is small
enough to look like rounding. ``np.allclose`` would pass it every time.

The second check is that nothing appears before warmup. A tree that reports a
warmup of 250 and returns numbers at row 200 has not necessarily read
forward, but it has returned values in rows it declares invalid, and
everything downstream would treat them as data.

Usage
-----
    python scripts/check_lookahead.py
    python scripts/check_lookahead.py --trees 2000 --seed 7
    python scripts/check_lookahead.py --root path/to/parquet --start 2010-01-01
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from alpha.data.synthetic import SyntheticSpec, generate
from alpha.data.types import Panel
from alpha.data.us_adapter import USAdapter
from alpha.expr.ast import Node, random_tree
from alpha.expr.evaluator import evaluate
from alpha.logging_config import configure


@dataclass(frozen=True, slots=True)
class Failure:
    """One tree that did not survive the check."""

    tree: str
    prefix_length: int
    reason: str
    first_row: int
    first_column: int


def check_tree(tree: Node, panel: Panel, cuts: tuple[int, ...]) -> list[Failure]:
    """Compare full evaluation against evaluation on each prefix."""
    failures: list[Failure] = []
    full = evaluate(tree, panel)

    head = full.values[: full.warmup]
    if head.size and not np.isnan(head).all():
        rows = np.flatnonzero(~np.isnan(head).all(axis=1))
        failures.append(
            Failure(str(tree), 0, "values before warmup", int(rows[0]), -1)
        )

    for k in cuts:
        if k <= 0 or k > panel.n_days:
            continue
        prefix = evaluate(tree, panel.head(k)).values
        if np.array_equal(prefix, full.values[:k], equal_nan=True):
            continue
        differs = ~(
            (prefix == full.values[:k])
            | (np.isnan(prefix) & np.isnan(full.values[:k]))
        )
        row, column = (int(a[0]) for a in np.nonzero(differs))
        failures.append(
            Failure(str(tree), k, "prefix differs from the full panel", row, column)
        )
    return failures


def build_panel(args: argparse.Namespace) -> Panel:
    if args.root:
        return USAdapter(args.root).panel(args.start, args.end)
    spec = SyntheticSpec(n_days=args.days, n_instruments=args.instruments)
    return generate(np.random.default_rng(args.seed), spec).panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--days", type=int, default=600)
    parser.add_argument("--instruments", type=int, default=120)
    parser.add_argument("--root", default=None, help="parquet root, instead of synthetic")
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2100-01-01")
    args = parser.parse_args()

    configure()
    panel = build_panel(args)
    # Cuts land on both sides of the longest window, so a tree with a 250-day
    # lookback is checked where it produces nothing and where it produces
    # everything.
    cuts = tuple(
        sorted({panel.n_days // 4, panel.n_days // 2, panel.n_days - 1, 251, 260})
    )
    print(f"panel {panel.n_days} x {panel.n_instruments}, cuts at {cuts}")
    print(f"checking {args.trees} random trees of depth {args.depth}\n")

    rng = np.random.default_rng(args.seed)
    failures: list[Failure] = []
    evaluated = 0
    for index in range(args.trees):
        tree = random_tree(rng, args.depth)
        failures.extend(check_tree(tree, panel, cuts))
        evaluated += 1
        if (index + 1) % 50 == 0:
            print(f"  {index + 1:5d} trees, {len(failures)} failures")

    print()
    if not failures:
        print(f"PASS: {evaluated} trees, every prefix bitwise identical")
        return 0

    print(f"FAIL: {len(failures)} failures across {evaluated} trees\n")
    for failure in failures[:20]:
        print(f"  {failure.reason} at k={failure.prefix_length}")
        print(f"    {failure.tree}")
        print(f"    first at row {failure.first_row}, column {failure.first_column}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
