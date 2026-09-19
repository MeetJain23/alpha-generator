"""Calibrate the gauntlet on both sides: size, then power.

Size is the half everybody measures. Plant nothing, run candidates through,
count how many get through. It should be near zero.

Power is the half that gets skipped, and skipping it is the reason systems
find nothing. Eight tests, each rejecting a real candidate twenty per cent of
the time, pass a real candidate ``0.8 ** 8``, which is seventeen per cent.
Five real discoveries in six are discarded, and the output is exactly what a
search with nothing to find produces: an empty pool and eight tests all
reporting that they are working correctly.

Two kinds of test, calibrated from two different places
-------------------------------------------------------
The first attempt at this set every threshold from the null, and it rejected
one hundred per cent of planted signals at every strength. The reason is worth
stating, because it is not obvious and the failure is silent.

``ic_magnitude`` is a significance test. Its null distribution is exactly what
a threshold should be read from, and that is where the size control comes
from.

The other seven are robustness tests, and every one of them is a *ratio*
normalised by the candidate's own IC: worst fold over whole sample, worst
jittered neighbour over centre, and so on. Under the null that denominator is
approximately zero, so the ratio is heavy-tailed noise whose upper quantiles
sit *above* what a real signal achieves. Measured on one panel, ``fold_ic``
had a null 90th percentile of 2.88 against 0.15 for a genuine signal.
Thresholding a ratio against a null where its denominator is noise is a
category error, and it produces a gauntlet that rejects everything while every
individual test looks defensible.

Noise candidates never reach the robustness tests anyway: the magnitude test
kills them. So the robustness thresholds are set from the distribution of the
statistic among signals known to be real, at a quantile that retains most of
them. Their job is to reject a signal that is strong but fragile, not to
reject noise.

Held out
--------
The thresholds are set on one set of planted panels and the false-rejection
rate is measured on a different set. Setting and measuring on the same panels
would report the quantile that was just chosen, not a rate.

Usage
-----
    python scripts/calibrate_gauntlet.py
    python scripts/calibrate_gauntlet.py --retain 0.95 --replicates 10
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from alpha.data.synthetic import SyntheticSpec, generate
from alpha.data.types import Panel
from alpha.expr.ast import Node, from_string, random_tree
from alpha.gauntlet.gauntlet import (
    TEST_DIRECTIONS,
    THRESHOLD_FOR_TEST,
    Gauntlet,
    GauntletThresholds,
)
from alpha.logging_config import configure

MAGNITUDE = "ic_magnitude"

TRUE_SIGNALS: tuple[str, ...] = (
    "div(ts_delay(close, 20), ts_delay(close, 250))",
    "div(ts_delay(close, 20), ts_delay(close, 120))",
    "div(ts_delay(close, 10), ts_delay(close, 250))",
    "rank(div(ts_delay(close, 20), ts_delay(close, 250)))",
)
"""Trees that carry the planted effect.

More than one, because a threshold set from a single expression describes that
expression rather than the class of things the gauntlet has to let through.
"""


def build_panel(args: argparse.Namespace, seed: int, strength: float) -> Panel:
    spec = SyntheticSpec(
        n_days=args.days,
        n_instruments=args.instruments,
        momentum_strength=strength,
    )
    return generate(np.random.default_rng(seed), spec).panel


def statistics_of(gauntlet: Gauntlet, tree: Node) -> dict[str, float] | None:
    try:
        return dict(gauntlet.run(tree).measured)
    except Exception:
        return None


def collect_null(args: argparse.Namespace) -> list[dict[str, float]]:
    """Random candidates on panels with nothing planted."""
    out: list[dict[str, float]] = []
    for replicate in range(args.null_panels):
        panel = build_panel(args, args.seed + 1000 + replicate, 0.0)
        gauntlet = Gauntlet(panel, n_folds=args.folds)
        rng = np.random.default_rng(args.seed + 5000 + replicate)
        for _ in range(args.null_trees):
            sample = statistics_of(gauntlet, random_tree(rng, args.depth))
            if sample is not None:
                out.append(sample)
    return out


def collect_planted(
    args: argparse.Namespace, strength: float, seeds: range
) -> list[dict[str, float]]:
    """Known-true signals on panels with the effect planted."""
    trees = [from_string(text) for text in TRUE_SIGNALS]
    out: list[dict[str, float]] = []
    for seed in seeds:
        panel = build_panel(args, seed, strength)
        gauntlet = Gauntlet(panel, n_folds=args.folds)
        for tree in trees:
            sample = statistics_of(gauntlet, tree)
            if sample is not None:
                out.append(sample)
    return out


def column(samples: list[dict[str, float]], test: str) -> np.ndarray:
    values = np.array([s.get(test, np.nan) for s in samples], dtype=np.float64)
    if test == MAGNITUDE:
        values = np.abs(values)
    return values


def propose(
    null: list[dict[str, float]],
    planted: list[dict[str, float]],
    *,
    null_admit: float,
    retain: float,
) -> GauntletThresholds:
    """Magnitude from the null. Robustness from signals known to be real."""
    values: dict[str, float] = {}

    magnitude = column(null, MAGNITUDE)
    magnitude = magnitude[np.isfinite(magnitude)]
    values[THRESHOLD_FOR_TEST[MAGNITUDE]] = float(
        np.quantile(magnitude, 1.0 - null_admit)
    )

    for test, above in TEST_DIRECTIONS.items():
        if test == MAGNITUDE:
            continue
        stats = column(planted, test)
        stats = stats[np.isfinite(stats)]
        if stats.size == 0:
            continue
        quantile = 1.0 - retain if above else retain
        values[THRESHOLD_FOR_TEST[test]] = float(np.quantile(stats, quantile))

    # A sign flip inside the neighbourhood is not a matter of degree.
    values["max_jitter_sign_flips"] = 0.0
    return GauntletThresholds(**values)  # type: ignore[arg-type]


def verdicts(
    samples: list[dict[str, float]], thresholds: GauntletThresholds
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for test, above in TEST_DIRECTIONS.items():
        threshold = getattr(thresholds, THRESHOLD_FOR_TEST[test])
        stats = column(samples, test)
        finite = np.isfinite(stats)
        passed = np.zeros(stats.size, dtype=bool)
        if above:
            passed[finite] = stats[finite] >= threshold
        else:
            passed[finite] = stats[finite] <= threshold
        out[test] = passed
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1400)
    parser.add_argument("--instruments", type=int, default=180)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--null-panels", type=int, default=2)
    parser.add_argument("--null-trees", type=int, default=120)
    parser.add_argument("--replicates", type=int, default=6,
                        help="planted panels for setting thresholds")
    parser.add_argument("--holdout", type=int, default=6,
                        help="further planted panels, used only for measuring")
    parser.add_argument("--calibrate-at", type=float, default=0.02,
                        help="planted strength the thresholds are set from; use the "
                             "weakest IC the budget is stated for, because several "
                             "of these statistics are not scale-free in signal "
                             "strength and a threshold fitted at a strong IC does "
                             "not transfer down")
    parser.add_argument("--strengths", type=float, nargs="*",
                        default=[0.01, 0.02, 0.04, 0.08])
    parser.add_argument("--null-admit", type=float, default=0.01,
                        help="null candidates the magnitude test may admit")
    parser.add_argument("--retain", type=float, default=0.98,
                        help="real signals each robustness test must keep")
    parser.add_argument("--budget", type=float, default=0.30,
                        help="joint false-rejection rate allowed at IC ~0.02")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    configure()
    print("SIZE: nothing planted")
    print(f"  {args.null_panels} panels x {args.null_trees} random candidates")
    null = collect_null(args)
    print(f"  {len(null)} usable samples")

    print()
    print(f"setting robustness thresholds on planted strength "
          f"{args.calibrate_at}, {args.replicates} panels")
    fitting = collect_planted(
        args, args.calibrate_at, range(args.seed, args.seed + args.replicates)
    )
    print(f"  {len(fitting)} samples from {len(TRUE_SIGNALS)} known-true signals")

    thresholds = propose(
        null, fitting, null_admit=args.null_admit, retain=args.retain
    )
    print()
    print("proposed thresholds")
    print(f"  {'test':16} {'':2} {'value':>9}   source")
    for test in TEST_DIRECTIONS:
        name = THRESHOLD_FOR_TEST[test]
        arrow = ">=" if TEST_DIRECTIONS[test] else "<="
        source = (
            f"null, admits {args.null_admit:.0%}"
            if test == MAGNITUDE
            else ("no flips allowed" if test == "jitter_sign"
                  else f"real signals, retains {args.retain:.0%}")
        )
        print(f"  {test:16} {arrow} {getattr(thresholds, name):9.4f}   {source}")

    null_pass = verdicts(null, thresholds)
    joint_null = np.ones(len(null), dtype=bool)
    print()
    print("false acceptance")
    for test, passed in null_pass.items():
        joint_null &= passed
        print(f"  {test:16} {passed.mean():7.1%}")
    far = float(joint_null.mean())
    print(f"  {'JOINT':16} {far:7.2%}   <- should be near zero")

    print()
    print(f"POWER: measured on {args.holdout} held-out panels per strength")
    print(f"  {'strength':>9} {'achieved IC':>12} {'joint FRR':>10}  per-test rejection")

    holdout_start = args.seed + 100_000
    worst_near_two = None
    for strength in args.strengths:
        samples = collect_planted(
            args, strength, range(holdout_start, holdout_start + args.holdout)
        )
        if not samples:
            continue
        passes = verdicts(samples, thresholds)
        joint = np.ones(len(samples), dtype=bool)
        rejects = []
        for test, passed in passes.items():
            joint &= passed
            if passed.mean() < 1.0:
                rejects.append(f"{test} {1.0 - passed.mean():.0%}")
        frr = 1.0 - float(joint.mean())
        mean_ic = float(np.nanmean(column(samples, MAGNITUDE)))
        print(f"  {strength:9.3f} {mean_ic:12.4f} {frr:10.1%}  "
              f"{', '.join(rejects) if rejects else 'none'}")
        if 0.012 <= mean_ic <= 0.035:
            worst_near_two = frr if worst_near_two is None else max(worst_near_two, frr)

    print()
    problems = []
    if far > 0.02:
        problems.append(f"false acceptance is {far:.2%}, not near zero")
    if worst_near_two is None:
        problems.append("no strength landed near IC 0.02, so power there is unmeasured")
    elif worst_near_two > args.budget:
        problems.append(
            f"joint false rejection at IC ~0.02 is {worst_near_two:.0%}, above "
            f"the {args.budget:.0%} budget: the gauntlet is discarding real "
            f"discoveries and would report an empty pool"
        )

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print("PASS: size near zero, power inside the budget at IC ~0.02.")
    print()
    print("Measured on synthetic data and bound to it. Every number here has")
    print("to be remeasured on the real panel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
