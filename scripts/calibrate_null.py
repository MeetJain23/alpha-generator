"""Calibrate the null: what the screen's statistics look like with nothing there.

Run this before the gauntlet, not after. Every gauntlet test is a statement
about how unusual a number is, and "unusual" is undefined until the
distribution under no effect is known. A threshold chosen without it is a
guess, and a guess that is too loose admits noise while a guess that is too
tight discards the thing the search exists to find.

What comes out
--------------
``tau``          the |IC| threshold, read off the null at the quantile that
                 admits the intended fraction of noise. Set, not chosen.

``sigma_SR``     the empirical dispersion of the IC-IR. The analytic Deflated
                 Sharpe assumes independent trials; grammar-generated
                 candidates share subtrees, reuse fields and overlap in
                 windows, so the analytic value is wrong for this search.

a leak check     the mean signed IC under the null must sit at zero. If it
                 does not, something upstream leaks, and this is the cheapest
                 place to learn that, because here the answer is known.

The full distribution is written out, not just the summary. A tail is what a
threshold is read from, but a distribution that is the wrong shape in the
middle is the first sign the null is not a null, and that is invisible in a
maximum.

Usage
-----
    python scripts/calibrate_null.py --trials 2000
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
from alpha.logging_config import configure
from alpha.screen.null import DEFAULT_BLOCK, calibrate


def build_panel(args: argparse.Namespace) -> Panel:
    if args.root:
        return USAdapter(args.root).panel(args.start, args.end)
    spec = SyntheticSpec(n_days=args.days, n_instruments=args.instruments)
    return generate(np.random.default_rng(args.seed), spec).panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2000,
                        help="random trees to score. The real calibration wants 1e5.")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--admit", type=float, default=0.001,
                        help="fraction of noise tau should admit")
    parser.add_argument("--min-scored-days", type=int, default=250,
                        help="admit a trial only if the screen would have")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--instruments", type=int, default=300)
    parser.add_argument("--root", default=None)
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2100-01-01")
    parser.add_argument("--out", default=None,
                        help="npz path for the full distribution. Ignored by git.")
    args = parser.parse_args()

    configure()
    panel = build_panel(args)
    print(f"panel            {panel.n_days} x {panel.n_instruments}")
    print(f"block length     {args.block} days")
    print(f"trials           {args.trials} random trees of depth {args.depth}")
    print("scoring against block-wise label-permuted forward returns\n")

    result = calibrate(
        panel,
        np.random.default_rng(args.seed),
        n_trials=args.trials,
        max_depth=args.depth,
        block_length=args.block,
        horizon=args.horizon,
        admit_fraction=args.admit,
        min_scored_days=args.min_scored_days,
    )

    print(f"usable trials    {result.n_trials}")
    print(f"median scored    {result.median_scored_days:.0f} days per trial\n")

    print("|IC| under the null")
    for name, value in result.abs_ic_quantiles.items():
        print(f"  {name:6} {value:.5f}")

    print()
    print(f"tau              {result.tau:.5f}")
    print(f"                 admits {args.admit:.3%} of noise "
          f"(quantile {result.tau_quantile:.5f})")
    print(f"sigma_SR         {result.sigma_sr:.4f}  empirical, for the DSR")
    print()
    print("leak check")
    print(f"  mean signed IC {result.mean_ic:+.6f}")
    print(f"  standard error {result.mean_ic_standard_error:.6f}")
    print(f"  distance       {result.leak_sigmas:.2f} sigma from zero")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            abs_ic=result.abs_ic,
            ic_ir=result.ic_ir,
            meta=np.array(
                [
                    json.dumps(
                        {
                            "n_trials": result.n_trials,
                            "tau": result.tau,
                            "sigma_sr": result.sigma_sr,
                            "block_length": args.block,
                            "horizon": args.horizon,
                            "seed": args.seed,
                        }
                    )
                ]
            ),
        )
        print(f"\nfull distribution written to {target}")

    print()
    if result.leaks:
        print("FAIL: the null is not centred at zero. Something upstream is")
        print("leaking, and every threshold read off this run would be wrong.")
        return 1
    print("PASS: the null is centred at zero. tau and sigma_SR above are what")
    print("the gauntlet should be calibrated against.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
