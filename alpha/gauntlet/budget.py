"""How many candidates the search may try, and how confident that number is.

The search budget is not set by compute. It is set by the false-acceptance
rate, and the arithmetic is short enough that nobody does it.

A candidate survives the pipeline under the null if it passes the screen and
then passes the gauntlet:

    P(survive | null) = P(pass screen | null)
                      x P(pass gauntlet | passed screen, null)

Expected null survivors at search size N is ``N x P(survive | null)``, and
that is the number that matters. At a screen admitting one in a thousand and a
gauntlet admitting one in two hundred of what reaches it, the product is about
five in a million, so a search of a million candidates is expected to produce
five survivors that are nothing at all. If the pool is meant to hold a handful
of real alphas, a search that size produces a pool that is mostly noise, and
no amount of compute fixes it. The budget is the N at which the expected count
crosses one.

The conditional is the one that matters
---------------------------------------
``P(pass gauntlet)`` has to be measured on candidates that passed the screen,
not on all null candidates. Those are different populations: the gauntlet's
robustness tests are ratios normalised by the candidate's own IC, and
conditioning on a large IC changes the denominator and therefore the whole
distribution of the statistic. Measuring it unconditionally is the same
category error that made the first gauntlet calibration reject everything.

The interval is wider than the estimate looks
---------------------------------------------
A false-acceptance rate of 0.5 per cent measured on two hundred candidates has
a 95 per cent interval of roughly 0.06 to 2.7 per cent. That is a factor of
forty in the rate and therefore a factor of forty in the supportable budget.
Quoting the point estimate alone turns an unknown into a decision.

So every rate here carries a Wilson interval, and the budget is reported as a
range. Wilson rather than the normal approximation because the counts are
small and the rate is near zero, which is exactly where the normal
approximation produces intervals that include negative probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np

Z_95: Final[float] = 1.959963984540054
"""Two-sided 95 per cent normal quantile."""


@dataclass(frozen=True, slots=True)
class Rate:
    """A measured rate and how little is known about it."""

    successes: int
    trials: int
    low: float
    high: float

    @property
    def estimate(self) -> float:
        return self.successes / self.trials if self.trials else float("nan")

    @property
    def width_factor(self) -> float:
        """How many times wider the upper bound is than the lower.

        The honest summary of a small-sample rate. A factor of two is a
        measurement; a factor of forty is a placeholder.
        """
        if self.low <= 0.0:
            return float("inf")
        return self.high / self.low

    def __str__(self) -> str:
        return (
            f"{self.estimate:.3%} [{self.low:.3%}, {self.high:.3%}] "
            f"({self.successes}/{self.trials})"
        )


def wilson(successes: int, trials: int, z: float = Z_95) -> Rate:
    """Wilson score interval for a binomial rate.

    Used rather than the normal approximation because the counts here are
    small and the rates near zero, which is precisely where the normal
    interval extends below zero and stops meaning anything.
    """
    if trials <= 0:
        return Rate(successes, trials, float("nan"), float("nan"))
    k, n = float(successes), float(trials)
    denominator = n + z * z
    centre = (k + 0.5 * z * z) / denominator
    spread = (z / denominator) * math.sqrt(k * (n - k) / n + 0.25 * z * z)
    # At zero successes the two terms are equal and the bound is exactly zero,
    # but in floating point they differ by a few times the machine epsilon and
    # leave dust. The exact cases are pinned so a caller can test for them.
    low = 0.0 if successes == 0 else max(0.0, centre - spread)
    high = 1.0 if successes == trials else min(1.0, centre + spread)
    return Rate(successes, trials, low, high)


@dataclass(frozen=True, slots=True)
class SurvivalBudget:
    """End-to-end false discovery, and the search size it permits."""

    screen: Rate
    gauntlet_given_screen: Rate

    @property
    def survival(self) -> tuple[float, float, float]:
        """Point estimate and interval for ``P(survive | null)``.

        The interval multiplies the two bounds, which is conservative and
        correct in direction: the two rates are measured on nested samples, so
        treating them as independent widens rather than narrows the result.
        """
        point = self.screen.estimate * self.gauntlet_given_screen.estimate
        low = self.screen.low * self.gauntlet_given_screen.low
        high = self.screen.high * self.gauntlet_given_screen.high
        return point, low, high

    def expected_survivors(self, search_size: int) -> tuple[float, float, float]:
        point, low, high = self.survival
        return search_size * point, search_size * low, search_size * high

    def budget_at(self, expected: float) -> tuple[float, float, float]:
        """Search size at which the expected null-survivor count hits a level.

        Returned worst case first, because the worst case is the one that
        constrains a decision. A high survival rate means a small budget.
        """
        point, low, high = self.survival
        worst = expected / high if high > 0 else float("inf")
        middle = expected / point if point > 0 else float("inf")
        best = expected / low if low > 0 else float("inf")
        return worst, middle, best

    def curve(self, sizes: Sequence[int]) -> tuple[tuple[int, float, float, float], ...]:
        return tuple((n, *self.expected_survivors(n)) for n in sizes)


def rejection_correlation(
    rejects: Mapping[str, np.ndarray]
) -> tuple[tuple[str, ...], np.ndarray]:
    """Pairwise correlation of which candidates each test rejects.

    Two tests that reject the same candidates are one piece of evidence billed
    twice. That is the hidden half of the compounding problem: the joint
    false-rejection rate pays for both, while the evidence they contribute is
    the same.

    The phi coefficient, which for two binary vectors is just their Pearson
    correlation. A test that never rejects, or always does, has no variance
    and its row comes back NaN rather than zero, because "no relationship" and
    "no information" are different statements.
    """
    names = tuple(rejects)
    matrix = np.full((len(names), len(names)), np.nan, dtype=np.float64)
    columns = [np.asarray(rejects[name], dtype=np.float64) for name in names]

    for i, left in enumerate(columns):
        for j, right in enumerate(columns):
            if left.std() == 0.0 or right.std() == 0.0:
                matrix[i, j] = np.nan if i != j else 1.0
                continue
            matrix[i, j] = float(np.corrcoef(left, right)[0, 1])
    return names, matrix


def redundant_pairs(
    names: Sequence[str], matrix: np.ndarray, threshold: float = 0.9
) -> tuple[tuple[str, str, float], ...]:
    """Pairs whose rejections agree closely enough that one is surplus."""
    out: list[tuple[str, str, float]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = matrix[i, j]
            if np.isfinite(value) and abs(value) >= threshold:
                out.append((names[i], names[j], float(value)))
    return tuple(sorted(out, key=lambda item: -abs(item[2])))
