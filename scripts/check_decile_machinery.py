"""Check the decile machinery against planted synthetic momentum.

This validates the portfolio harness, not the market. It plants a known 12-1
momentum effect in synthetic returns and asserts that the decile sort finds
it, and that it finds nothing when nothing is planted.

What this cannot do
-------------------
It cannot validate the data pipeline. Delisting composition, adjustment
factors and point-in-time alignment are the things the momentum milestone
exists to exercise, and synthetic data satisfies all of them by construction:
the generator applies terminal returns because it was written to, the prices
are already adjusted because they were never unadjusted, and nothing is ever
restated because there is no vendor to restate it. A harness cannot test the
invariants its own fixture is built from.

So this script answers one question, "does the sort work", and
``verify_momentum.py`` holds the open question, "is the data right".

The two controls
----------------
Both matter, and either alone would be misleading. Finding a planted effect
shows the machinery can detect signal. Finding nothing when nothing was
planted shows it is not manufacturing it. A harness that passes only the first
is one that reports a spread for any input.

Usage
-----
    python scripts/check_decile_machinery.py
    python scripts/check_decile_machinery.py --planted 0.10 --days 2000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from alpha.data.synthetic import SyntheticSpec, generate
from alpha.expr.ast import from_string
from alpha.expr.evaluator import evaluate
from alpha.logging_config import configure
from alpha.screen.portfolio import decile_returns

MOMENTUM = "div(ts_delay(close, 20), ts_delay(close, 250))"

NULL_SPREAD_CEILING = 0.15
"""Annualised D10 minus D1 tolerated when nothing was planted.

Generous, because a few dozen months of noise on a few hundred instruments
genuinely wanders. It is a check that the harness is not manufacturing a
result, not a significance test.
"""

PLANTED_SPREAD_FLOOR = 0.25
"""Annualised spread the harness must find when an effect is planted."""


def run_once(planted: float, args: argparse.Namespace) -> tuple[float, float, int]:
    spec = SyntheticSpec(
        n_days=args.days,
        n_instruments=args.instruments,
        momentum_strength=planted,
    )
    panel = generate(np.random.default_rng(args.seed), spec).panel
    result = evaluate(from_string(MOMENTUM), panel)
    deciles = decile_returns(result.values, result.warmup, panel, lag=not args.no_lag)
    return (
        deciles.spread_annualised,
        deciles.top_minus_market_annualised,
        deciles.n_months,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--instruments", type=int, default=400)
    parser.add_argument("--planted", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--no-lag", action="store_true",
                        help="earn the formation day's return, which is look-ahead")
    args = parser.parse_args()

    configure()
    print(f"tree        {MOMENTUM}")
    print(f"panel       {args.days} x {args.instruments}\n")

    null_spread, null_excess, months = run_once(0.0, args)
    print(f"control, nothing planted")
    print(f"  D10 - D1      {null_spread:+.2%} annualised over {months} months")
    print(f"  D10 - market  {null_excess:+.2%}")

    planted_spread, planted_excess, months = run_once(args.planted, args)
    print(f"\neffect planted at {args.planted} daily sigma")
    print(f"  D10 - D1      {planted_spread:+.2%} annualised over {months} months")
    print(f"  D10 - market  {planted_excess:+.2%}")

    print()
    failures = []
    if abs(null_spread) > NULL_SPREAD_CEILING:
        failures.append(
            f"the harness reports {null_spread:+.2%} on data with no effect in it"
        )
    if planted_spread < PLANTED_SPREAD_FLOOR:
        failures.append(
            f"the harness found only {planted_spread:+.2%} for a planted effect"
        )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print("PASS: the sort finds a planted effect and finds nothing without one.")
    print("This says nothing about the data pipeline. See verify_momentum.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
