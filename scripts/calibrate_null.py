"""Calibrate the null: what the screen's statistics look like with nothing there.

Run this before the gauntlet, not after. Every gauntlet test is a statement
about how unusual a number is, and "unusual" is undefined until the
distribution under no effect is known. A threshold chosen without it is a
guess, and a guess that is too loose admits noise while a guess that is too
tight discards the thing the search exists to find.

These numbers describe a dataset
--------------------------------
Run on a synthetic panel and every constant produced describes the generator,
not the market. Real returns have fat tails, volatility clustering,
cross-sectional dispersion that varies enormously across decades, and sector
block correlation that makes three thousand names behave like far fewer. None
of that is in a generated panel unless it was put there.

The calibration is therefore bound to the ``data_snapshot_id`` of the panel it
was measured on, and the screen refuses a calibration from a different
snapshot. The mismatch is made impossible rather than documented, because it
is silent: a tau from the wrong dataset still produces a pass rate.

What comes out
--------------
``tau``          the |IC| threshold, read off the null at the quantile that
                 admits the intended fraction of noise.

``sigma_SR``     the dispersion of the IC-IR under the null.

a maximum curve  ``E[max |IC|]`` and ``E[max SR]`` as a function of search
                 size, measured over disjoint batches. This replaces
                 ``sigma * sqrt(2 ln N)``, which is reported beside it as a
                 sanity check only. See docs/REPRODUCIBILITY.md.

a leak check     the mean signed IC under the null must sit at zero.

The candidate source must be the search's
-----------------------------------------
``--source`` selects it. The maximum of N statistics depends on how correlated
those N are, and an evolutionary search that breeds from survivors is far more
correlated than uniform random trees. Calibrating on the wrong source measures
the wrong maximum, in the direction that overstates the bar.

Usage
-----
    python scripts/calibrate_null.py --trials 2500
    python scripts/calibrate_null.py --trials 100000 --out runs/null.npz
    python scripts/calibrate_null.py --root path/to/parquet --trials 100000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from alpha.data.synthetic import SyntheticSpec, generate
from alpha.data.types import Panel
from alpha.data.us_adapter import USAdapter
from alpha.expr.ast import random_tree
from alpha.logging_config import configure
from alpha.screen.null import DEFAULT_BLOCK, calibrate


def build_panel(args: argparse.Namespace) -> Panel:
    if args.root:
        return USAdapter(args.root).panel(args.start, args.end)
    spec = SyntheticSpec(n_days=args.days, n_instruments=args.instruments)
    return generate(np.random.default_rng(args.seed), spec).panel


def build_source(args: argparse.Namespace):
    """The function that produces candidates. Must match the real search."""
    if args.source == "random_tree":
        depth = args.depth
        return (lambda rng: random_tree(rng, depth)), "random_tree"
    raise SystemExit(
        f"unknown candidate source {args.source!r}. The evolutionary "
        f"generator is not built yet; when it is, it must be selectable here, "
        f"because calibrating on a different source measures a different "
        f"maximum."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2500,
                        help="candidates to score. The real calibration wants 1e5.")
    parser.add_argument("--source", default="random_tree",
                        help="candidate source; must match the real search")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK,
                        help="days per label permutation block, 20 to 60")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--admit", type=float, default=0.001,
                        help="fraction of noise tau should admit")
    parser.add_argument("--min-scored-days", type=int, default=250,
                        help="admit a trial only if the screen would have")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--instruments", type=int, default=300)
    parser.add_argument("--root", default=None)
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2100-01-01")
    parser.add_argument("--out", default=None,
                        help="npz path for the full distribution. Ignored by git.")
    parser.add_argument("--no-synthetic-warning", dest="synthetic_warning",
                        action="store_false")
    args = parser.parse_args()

    configure()
    panel = build_panel(args)
    source, source_name = build_source(args)

    print(f"panel            {panel.n_days} x {panel.n_instruments}")
    print(f"block length     {args.block} days")
    print(f"horizon          {args.horizon} day")
    print(f"source           {source_name}")
    print(f"trials           {args.trials}")
    print("scoring against block-wise label-permuted forward returns")
    print()

    result = calibrate(
        panel,
        np.random.default_rng(args.seed),
        source=source,
        source_name=source_name,
        n_trials=args.trials,
        max_depth=args.depth,
        block_length=args.block,
        horizon=args.horizon,
        admit_fraction=args.admit,
        min_scored_days=args.min_scored_days,
    )

    print(f"usable trials    {result.n_trials}")
    print(f"median scored    {result.median_scored_days:.0f} days per trial")
    print(f"snapshot         {result.data_snapshot_id}")
    print(f"grammar          {result.grammar_fingerprint[:16]}")
    print()

    print("|IC| under the null")
    for name, value in result.abs_ic_quantiles.items():
        print(f"  {name:6} {value:.5f}")

    print()
    print(f"tau              {result.tau:.5f}")
    print(f"                 admits {args.admit:.3%} of noise")
    print(f"sigma_SR         {result.sigma_sr:.4f}")

    print()
    print("maximum by search size, over disjoint batches")
    print(f"  {'N':>7} {'batches':>8} {'E[max|IC|]':>11} {'p95|IC|':>9} "
          f"{'E[maxSR]':>9} {'sqrt(2lnN)':>11}")
    curve, sr = result.max_abs_ic, result.max_ic_ir
    for i, size in enumerate(curve.sizes):
        analytic = result.analytic_max_ic_ir(size)
        sr_mean = sr.mean[i] if i < len(sr.mean) else float("nan")
        print(f"  {size:7d} {curve.n_batches[i]:8d} {curve.mean[i]:11.5f} "
              f"{curve.p95[i]:9.5f} {sr_mean:9.4f} {analytic:11.4f}")
    if not curve.sizes:
        print("  (too few usable trials to measure a maximum: needs 80)")

    print()
    print("leak check")
    print(f"  mean signed IC {result.mean_ic:+.6f}")
    print(f"  standard error {result.mean_ic_standard_error:.6f}")
    print(f"  distance       {result.leak_sigmas:.2f} sigma from zero")

    if args.synthetic_warning and not args.root:
        print()
        print("NOTE: this panel is synthetic. tau and sigma_SR describe the")
        print("generator, not US equities. These numbers are bound to this")
        print("snapshot and will be refused against any other panel.")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            abs_ic=result.abs_ic,
            ic_ir=result.ic_ir,
            max_sizes=np.array(curve.sizes, dtype=np.int64),
            max_abs_ic_mean=np.array(curve.mean, dtype=np.float64),
            max_abs_ic_p95=np.array(curve.p95, dtype=np.float64),
            max_ic_ir_mean=np.array(sr.mean, dtype=np.float64),
            meta=np.array([json.dumps(dict(result.binding()))]),
        )
        print()
        print(f"full distribution written to {target}")

    print()
    if result.leaks:
        print("FAIL: the null is not centred at zero. Something upstream is")
        print("leaking, and every threshold read off this run would be wrong.")
        return 1
    print("PASS: the null is centred at zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
