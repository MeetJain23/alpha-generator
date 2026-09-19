"""Calibrate the gauntlet, and work out the search budget it implies.

Reports, in order:

  SIZE      how often a null candidate gets through, with an interval, and
            the conditional rate given it already cleared the screen
  BUDGET    expected null survivors as a curve over search size, and the N at
            which that count crosses 0.1 and 1.0
  REDUNDANCY  pairwise agreement between the tests, because two tests that
            reject the same candidates bill one piece of evidence twice
  POWER     false-rejection rate over a curve of achieved IC, against plants
            shaped like real effects rather than idealised ones

The budget is the point of it. A search is limited by its false-acceptance
rate, not by compute: at five null survivors per million candidates, a search
of a million produces a pool that is mostly noise however fast the evaluator
is. Nobody computes this, and it is the number that decides what the generator
has to be.

Intervals, not point estimates
------------------------------
A rate of 0.5 per cent measured on two hundred candidates has a 95 per cent
Wilson interval of roughly 0.06 to 2.7 per cent, which is a factor of forty in
the rate and a factor of forty in the budget. Every rate below is printed with
its interval, and the budget as a range.

The conditional rate is measured at a series of magnitude cuts rather than
only at tau. Candidates clearing tau are one in a thousand, so conditioning
there leaves a handful of samples and an interval too wide to use. Reporting
the conditional across cuts shows whether it is stable, which is what makes
reading it at tau defensible.

Plants that look like real effects
----------------------------------
An idealised plant is stationary, universe-wide and constant, which is exactly
the shape that fold stability, breadth and regime agreement reward. Measuring
power against it flatters the gauntlet. The shapes here decay over the sample,
restrict themselves to the illiquid half of the universe, and appear only in
high-volatility regimes, which are three ordinary properties of real effects
and three ways a real alpha gets thrown out.

Usage
-----
    python scripts/calibrate_gauntlet.py
    python scripts/calibrate_gauntlet.py --null-trees 2000 --days 900
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from alpha.data.synthetic import SyntheticSpec, generate
from alpha.data.types import Panel
from alpha.expr.ast import Node, from_string, random_tree
from alpha.gauntlet.budget import (
    SurvivalBudget,
    redundant_pairs,
    rejection_correlation,
    wilson,
)
from alpha.gauntlet.gauntlet import (
    TEST_DIRECTIONS,
    THRESHOLD_FOR_TEST,
    Gauntlet,
    GauntletThresholds,
)
from alpha.logging_config import configure

MAGNITUDE = "ic_magnitude"
ROBUSTNESS = tuple(name for name in TEST_DIRECTIONS if name != MAGNITUDE)

TRUE_SIGNALS: tuple[str, ...] = (
    "div(ts_delay(close, 20), ts_delay(close, 250))",
    "div(ts_delay(close, 20), ts_delay(close, 120))",
    "div(ts_delay(close, 10), ts_delay(close, 250))",
    "rank(div(ts_delay(close, 20), ts_delay(close, 250)))",
)

PLANT_SHAPES: dict[str, dict[str, float | bool]] = {
    "idealised": {},
    "decaying": {"momentum_decay_to": 0.2},
    "illiquid half": {"momentum_universe_fraction": 0.5},
    "high-vol only": {"momentum_regime_only": True},
}

BUDGET_SIZES: tuple[int, ...] = (100, 1_000, 10_000, 100_000, 1_000_000)

MIN_CONDITIONAL_SAMPLES: int = 20
"""Null candidates a cut needs before its conditional rate is worth reading."""

ADVERSARIAL_STRENGTH: float = 0.12
"""Strength for the non-idealised plants.

Compared at equal strength, an adversarial plant simply achieves a lower IC,
so it fails the magnitude test and the comparison says nothing about the
robustness tests. The point is to ask whether a real-shaped effect of
comparable STRENGTH IN IC survives, so these are planted hard enough to clear
tau and the achieved IC is printed beside each one.
"""


def holdout_seeds(args) -> range:
    """Seeds never used to fit a threshold."""
    return range(args.seed + 100_000, args.seed + 100_000 + args.holdout)


def build_panel(args, seed: int, strength: float, **shape) -> Panel:
    spec = SyntheticSpec(
        n_days=args.days,
        n_instruments=args.instruments,
        momentum_strength=strength,
        vol_regime_strength=args.vol_regime,
        **shape,
    )
    return generate(np.random.default_rng(seed), spec).panel


def statistics_of(gauntlet: Gauntlet, tree: Node) -> dict[str, float] | None:
    try:
        return dict(gauntlet.run(tree).measured)
    except Exception:
        return None


def collect_null(args) -> list[dict[str, float]]:
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


def collect_planted(args, strength: float, seeds, **shape) -> list[dict[str, float]]:
    trees = [from_string(text) for text in TRUE_SIGNALS]
    out: list[dict[str, float]] = []
    for seed in seeds:
        panel = build_panel(args, seed, strength, **shape)
        gauntlet = Gauntlet(panel, n_folds=args.folds)
        for tree in trees:
            sample = statistics_of(gauntlet, tree)
            if sample is not None:
                out.append(sample)
    return out


def column(samples, test: str) -> np.ndarray:
    values = np.array([s.get(test, np.nan) for s in samples], dtype=np.float64)
    return np.abs(values) if test == MAGNITUDE else values


def propose(null, planted, *, null_admit: float, retain: float) -> GauntletThresholds:
    values: dict[str, float] = {}
    magnitude = column(null, MAGNITUDE)
    magnitude = magnitude[np.isfinite(magnitude)]
    values[THRESHOLD_FOR_TEST[MAGNITUDE]] = float(
        np.quantile(magnitude, 1.0 - null_admit)
    )
    for test in ROBUSTNESS:
        stats = column(planted, test)
        stats = stats[np.isfinite(stats)]
        if stats.size == 0:
            continue
        quantile = 1.0 - retain if TEST_DIRECTIONS[test] else retain
        values[THRESHOLD_FOR_TEST[test]] = float(np.quantile(stats, quantile))
    values["max_jitter_sign_flips"] = 0.0
    return GauntletThresholds(**values)  # type: ignore[arg-type]


def verdicts(samples, thresholds: GauntletThresholds) -> dict[str, np.ndarray]:
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


def robustness_pass(passes: dict[str, np.ndarray]) -> np.ndarray:
    joint = np.ones(len(next(iter(passes.values()))), dtype=bool)
    for test in ROBUSTNESS:
        joint &= passes[test]
    return joint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=900)
    parser.add_argument("--instruments", type=int, default=120)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--vol-regime", type=float, default=0.4)
    parser.add_argument("--null-panels", type=int, default=3)
    parser.add_argument("--null-trees", type=int, default=700)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--holdout", type=int, default=5)
    parser.add_argument("--calibrate-at", type=float, default=0.02)
    parser.add_argument("--strengths", type=float, nargs="*",
                        default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08])
    parser.add_argument("--null-admit", type=float, default=0.001)
    parser.add_argument("--retain", type=float, default=0.98)
    parser.add_argument("--budget", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    configure()
    print(f"panel {args.days} x {args.instruments}, vol regime {args.vol_regime}")
    print()
    print("SIZE")
    null = collect_null(args)
    print(f"  {len(null)} null candidates through the full gauntlet")

    fitting = collect_planted(
        args, args.calibrate_at, range(args.seed, args.seed + args.replicates)
    )
    thresholds = propose(
        null, fitting, null_admit=args.null_admit, retain=args.retain
    )
    tau = thresholds.min_abs_ic

    print()
    print("  proposed thresholds")
    for test in TEST_DIRECTIONS:
        arrow = ">=" if TEST_DIRECTIONS[test] else "<="
        source = "null" if test == MAGNITUDE else (
            "fixed" if test == "jitter_sign" else f"real signals, retain {args.retain:.0%}"
        )
        value = getattr(thresholds, THRESHOLD_FOR_TEST[test])
        print(f"    {test:16} {arrow} {value:9.4f}   {source}")

    null_pass = verdicts(null, thresholds)
    magnitude = column(null, MAGNITUDE)
    screen_hits = int(null_pass[MAGNITUDE].sum())
    screen_rate = wilson(screen_hits, len(null))

    print()
    print(f"  P(pass screen | null)      {screen_rate}")

    print()
    print("  P(pass robustness | |IC| above a cut), by cut")
    print(f"    {'cut':>9} {'n':>6}  rate")
    finite = magnitude[np.isfinite(magnitude)]
    robust = robustness_pass(null_pass)
    usable: list[tuple[str, float, object]] = []
    for label, cut in (
        ("p50", float(np.quantile(finite, 0.50))),
        ("p90", float(np.quantile(finite, 0.90))),
        ("p99", float(np.quantile(finite, 0.99))),
        ("tau", float(tau)),
    ):
        selected = np.isfinite(magnitude) & (magnitude >= cut)
        rate = wilson(int(robust[selected].sum()), int(selected.sum()))
        marker = "  <- at tau" if label == "tau" else ""
        print(f"    {label:>3} {cut:5.4f} {int(selected.sum()):6d}  {rate}{marker}")
        if selected.sum() >= MIN_CONDITIONAL_SAMPLES:
            usable.append((label, cut, rate))

    # The conditional cannot be measured at tau by brute force: one null
    # candidate in a thousand clears it, so the sample there is a handful and
    # its interval spans everything. What the table shows instead is that the
    # conditional RISES with the cut, which is not an accident: every
    # robustness statistic is a ratio with the candidate own IC in the
    # denominator, so a null candidate that cleared a high bar by luck has a
    # large denominator and looks stable.
    #
    # So the highest cut with a usable sample is taken as a LOWER BOUND on the
    # conditional at tau, and the budget that follows is an UPPER BOUND on N.
    # Extrapolating the trend would be guessing; ignoring it and using the
    # empty sample at tau would be worse, because 0/3 reads as zero risk.
    conditional_label, conditional_cut, conditional = usable[-1]
    print()
    print(f"  conditional taken at {conditional_label} (cut {conditional_cut:.4f}), "
          f"the highest cut with at least {MIN_CONDITIONAL_SAMPLES} samples.")
    print("  It rises with the cut, so this is a LOWER bound on the rate at tau,")
    print("  and every budget below is an UPPER bound on N.")

    budget = SurvivalBudget(screen=screen_rate, gauntlet_given_screen=conditional)
    point, low, high = budget.survival

    print()
    print("BUDGET")
    print(f"  P(survive | null)          {point:.3e}  [{low:.3e}, {high:.3e}]")
    print()
    print(f"  {'N':>10}  expected null survivors [interval]")
    for size, mid, lo, hi in budget.curve(BUDGET_SIZES):
        print(f"  {size:10,d}  {mid:10.3f}  [{lo:.3f}, {hi:.3f}]")
    print()
    for level in (0.1, 1.0):
        worst, mid, best = budget.budget_at(level)
        print(f"  expected survivors = {level:<4} at N = {mid:,.0f}"
              f"   [{worst:,.0f} .. {best:,.0f}]")
    worst_budget, _, _ = budget.budget_at(1.0)
    print()
    print(f"  SEARCH BUDGET (worst case, one expected null survivor): "
          f"N = {worst_budget:,.0f}")
    needed = int(20 / max(screen_rate.estimate, 1e-9))
    print()
    print(f"  Pinning the conditional at tau to within 2x needs about 20 events")
    print(f"  above tau, which at a {screen_rate.estimate:.3%} screen rate means")
    print(f"  roughly {needed:,} gauntlet runs. That is why it is bounded rather")
    print(f"  than measured.")

    print()
    print("REDUNDANCY, pooled over every planted panel")
    pooled: list[dict[str, float]] = list(fitting)
    for strength in args.strengths:
        pooled.extend(collect_planted(args, strength, holdout_seeds(args)))
    for shape in PLANT_SHAPES.values():
        pooled.extend(
            collect_planted(args, ADVERSARIAL_STRENGTH, holdout_seeds(args), **shape)
        )
    print(f"  {len(pooled)} samples, so the rejections are numerous enough to")
    print("  correlate. Measured on the fitting set alone there were too few")
    print("  rejection events for a correlation to mean anything.")
    fit_pass = verdicts(pooled, thresholds)
    rejects = {name: (~fit_pass[name]).astype(float) for name in TEST_DIRECTIONS}
    names, matrix = rejection_correlation(rejects)
    header = "    " + "".join(f"{n[:7]:>8}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = "".join(
            "     nan" if not np.isfinite(matrix[i, j]) else f"{matrix[i, j]:8.2f}"
            for j in range(len(names))
        )
        print(f"  {name[:9]:>9}{row}")
    pairs = redundant_pairs(names, matrix, threshold=0.9)
    print()
    if pairs:
        for left, right, value in pairs:
            print(f"  REDUNDANT: {left} and {right} agree at {value:+.2f}; drop one")
    else:
        print("  no pair agrees above 0.9")

    print()
    adversarial_failures: list[tuple[str, float, float, str]] = []
    print(f"POWER, held out, {args.holdout} panels per point")
    print(f"  {'shape':14} {'strength':>8} {'IC':>8} {'joint FRR':>10}  worst tests")
    holdout = holdout_seeds(args)
    near_budget: list[float] = []

    for strength in args.strengths:
        samples = collect_planted(args, strength, holdout)
        if not samples:
            continue
        passes = verdicts(samples, thresholds)
        joint = np.ones(len(samples), dtype=bool)
        for test in TEST_DIRECTIONS:
            joint &= passes[test]
        frr = 1.0 - float(joint.mean())
        ic = float(np.nanmean(column(samples, MAGNITUDE)))
        worst = sorted(
            ((1.0 - passes[t].mean(), t) for t in TEST_DIRECTIONS), reverse=True
        )[:2]
        detail = ", ".join(f"{t} {r:.0%}" for r, t in worst if r > 0) or "none"
        flag = "  (below tau: the screen rejects these)" if ic < tau else ""
        print(f"  {'idealised':14} {strength:8.3f} {ic:8.4f} {frr:10.1%}  {detail}{flag}")
        if ic >= tau:
            near_budget.append(frr)

    print()
    for name, shape in PLANT_SHAPES.items():
        if name == "idealised":
            continue
        samples = collect_planted(args, ADVERSARIAL_STRENGTH, holdout, **shape)
        if not samples:
            continue
        passes = verdicts(samples, thresholds)
        joint = np.ones(len(samples), dtype=bool)
        for test in TEST_DIRECTIONS:
            joint &= passes[test]
        frr = 1.0 - float(joint.mean())
        ic = float(np.nanmean(column(samples, MAGNITUDE)))
        worst = sorted(
            ((1.0 - passes[t].mean(), t) for t in TEST_DIRECTIONS), reverse=True
        )[:2]
        detail = ", ".join(f"{t} {r:.0%}" for r, t in worst if r > 0) or "none"
        flag = "  (below tau)" if ic < tau else ""
        print(f"  {name:14} {ADVERSARIAL_STRENGTH:8.3f} {ic:8.4f} {frr:10.1%}  "
              f"{detail}{flag}")
        if ic >= tau and frr > args.budget:
            adversarial_failures.append((name, ic, frr, detail))

    print()
    problems = []
    for name, ic, frr, detail in adversarial_failures:
        problems.append(
            f"the {name} plant reaches IC {ic:.4f}, above tau, and is still "
            f"rejected {frr:.0%} of the time ({detail})"
        )
    if near_budget and max(near_budget) > args.budget:
        problems.append(
            f"joint false rejection reaches {max(near_budget):.0%} above tau, "
            f"over the {args.budget:.0%} budget"
        )
    if pairs:
        problems.append(
            f"{len(pairs)} test pair(s) agree above 0.9 and bill one piece of "
            f"evidence twice"
        )
    for problem in problems:
        print(f"REVIEW: {problem}")
    if not problems:
        print("Size, budget and power all inside their stated limits.")
    print()
    print("Measured on synthetic data and bound to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
