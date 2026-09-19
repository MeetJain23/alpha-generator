"""The gauntlet: seven tests a candidate has to survive, and no frozen numbers.

Every threshold here starts as ``None``. The gauntlet will measure a candidate
and report every statistic with no threshold at all, and it refuses to return
a verdict until each threshold has been set from a calibration. A number
written into this file by hand would be a guess wearing the authority of code.

Both sides get calibrated
-------------------------
Size is the easy half: plant nothing, run the gauntlet, and the acceptance
rate should be near zero. Everyone measures that.

Power is the half that gets skipped, and skipping it is why systems find
nothing. Seven tests, each rejecting a real candidate twenty per cent of the
time, pass a real candidate ``0.8 ** 7``, which is twenty-one per cent. Six
real discoveries in seven are thrown away, and the output is indistinguishable
from a search that had nothing to find: an empty pool, and every test
reporting that it is working correctly.

So ``scripts/calibrate_gauntlet.py`` plants signals at IC 0.01, 0.02 and 0.03
and measures the joint false-rejection rate and the per-test breakdown. If the
joint rate at IC 0.02 is above roughly thirty per cent, the gauntlet is too
strict and the thresholds have to move.

Two tests that misbehave if written the obvious way
---------------------------------------------------
**Delay** is horizon-dependent, so a flat threshold is wrong. A
monthly-holding alpha loses almost nothing to a one-day delay; a three-day
alpha legitimately loses most of it, and failing it for that is failing it for
being what it is.

The fix here is not to bucket by horizon but to normalise by the signal's own
decay. A real signal decays smoothly, so the drop from delay 0 to 1 is about
the same as from 1 to 2. A signal contaminated by same-day information falls
off a cliff at the first step and then decays normally. The statistic is the
ratio of the first decrement to the second, which is a shape and carries no
units of horizon at all.

**Jitter** asserts a shape too, not a level. The wrong test is "IC exceeds tau
at every jittered setting", which fails every real alpha whose IC sits near
tau, because half its neighbourhood will land below by construction. What
should hold is that the neighbourhood is a plateau rather than a spike: the
IC stays a decent fraction of its centre value across plus or minus twenty per
cent, and never changes sign.

Twenty per cent of a window is off the admissible ladder, so the jitter test
builds probes. A probe is a sensitivity instrument, never a hypothesis, and
never reaches the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Final, Mapping, Sequence

import numpy as np

from alpha.data.types import DTYPE, Panel
from alpha.expr.ast import Node
from alpha.expr.evaluator import EvaluationError, evaluate, rank_rows
from alpha.gauntlet.folds import Fold, purged_folds, regime_split
from alpha.logging_config import get_logger
from alpha.screen.metrics import MIN_CROSS_SECTION, daily_rank_ic, rank_forward
from alpha.screen.null import NullCalibration

_log = get_logger(__name__)

JITTER_FRACTIONS: Final[tuple[float, ...]] = (-0.2, -0.1, 0.1, 0.2)
"""The neighbourhood a parameter is probed over. A plateau across this, not a
spike at the centre."""

DELAY_STEPS: Final[tuple[int, ...]] = (0, 1, 2)
"""Enough to measure the shape of the decay. The first decrement against the
second is what distinguishes a cliff from a decay."""


TEST_DIRECTIONS: Final[Mapping[str, bool]] = {
    "ic_magnitude": True,
    "fold_sign": True,
    "fold_ic": True,
    "delay_shape": False,
    "jitter_plateau": True,
    "jitter_sign": False,
    "regime_sign": True,
    "breadth": True,
}
"""Whether a test wants its statistic above the threshold or below it.

Needed by the calibration, which reads each threshold off the distribution of
the statistic under no effect and has to know which tail to take.
"""

THRESHOLD_FOR_TEST: Final[Mapping[str, str]] = {
    "ic_magnitude": "min_abs_ic",
    "fold_sign": "min_fold_sign_agreement",
    "fold_ic": "min_fold_ic_fraction",
    "delay_shape": "max_delay_cliff",
    "jitter_plateau": "min_jitter_plateau",
    "jitter_sign": "max_jitter_sign_flips",
    "regime_sign": "min_regime_sign_agreement",
    "breadth": "min_breadth",
}


class NotCalibrated(RuntimeError):
    """A verdict was asked for before the thresholds were measured."""


@dataclass(frozen=True, slots=True)
class GauntletThresholds:
    """Every threshold, all unset.

    ``None`` means not yet measured. Nothing here has a default, because a
    default would be a number somebody guessed that then acquired the
    authority of having been written down.
    """

    min_abs_ic: float | None = None
    """From the null calibration's tau."""

    min_fold_sign_agreement: float | None = None
    """Fraction of purged folds whose IC sign matches the whole-sample sign."""

    min_fold_ic_fraction: float | None = None
    """Worst fold's |IC| as a fraction of the whole-sample |IC|."""

    max_delay_cliff: float | None = None
    """First decrement over second. Large means a cliff at delay one."""

    min_jitter_plateau: float | None = None
    """Worst neighbour's IC as a signed fraction of the centre's."""

    max_jitter_sign_flips: int | None = None

    min_regime_sign_agreement: float | None = None
    """Both volatility regimes must agree on direction."""

    min_breadth: float | None = None
    """Fraction of the IC that survives dropping the best decile of days."""

    def unset(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "min_abs_ic",
                "min_fold_sign_agreement",
                "min_fold_ic_fraction",
                "max_delay_cliff",
                "min_jitter_plateau",
                "max_jitter_sign_flips",
                "min_regime_sign_agreement",
                "min_breadth",
            )
            if getattr(self, name) is None
        )

    def require_complete(self) -> None:
        missing = self.unset()
        if missing:
            raise NotCalibrated(
                f"these thresholds have never been measured: {list(missing)}. "
                f"Run scripts/calibrate_gauntlet.py, which measures both the "
                f"false-acceptance rate with nothing planted and the joint "
                f"false-rejection rate with signals planted at known IC. A "
                f"gauntlet with guessed thresholds discards real candidates "
                f"and reports that it is working."
            )

    @classmethod
    def from_calibration(
        cls, calibration: NullCalibration, **measured: float | int
    ) -> GauntletThresholds:
        """Take tau from the null, and the rest from a gauntlet calibration."""
        return cls(min_abs_ic=calibration.tau, **measured)  # type: ignore[arg-type]

    def as_config(self) -> dict[str, float | int | None]:
        return {f"gauntlet_{name}": getattr(self, name) for name in (
            "min_abs_ic",
            "min_fold_sign_agreement",
            "min_fold_ic_fraction",
            "max_delay_cliff",
            "min_jitter_plateau",
            "max_jitter_sign_flips",
            "min_regime_sign_agreement",
            "min_breadth",
        )}


@dataclass(frozen=True, slots=True)
class TestOutcome:
    """One test's measurement, and its verdict if there was a threshold."""

    name: str
    statistic: float
    threshold: float | None
    passed: bool | None
    detail: Mapping[str, float] = field(default_factory=dict)

    @property
    def judged(self) -> bool:
        return self.passed is not None


@dataclass(frozen=True, slots=True)
class GauntletResult:
    tree: Node
    outcomes: tuple[TestOutcome, ...]
    n_folds: int

    @property
    def measured(self) -> Mapping[str, float]:
        return {outcome.name: outcome.statistic for outcome in self.outcomes}

    @property
    def judged(self) -> bool:
        return all(outcome.judged for outcome in self.outcomes)

    @property
    def passed(self) -> bool:
        if not self.judged:
            raise NotCalibrated("a verdict was asked for before calibration")
        return all(outcome.passed for outcome in self.outcomes)

    def failures(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.outcomes if o.passed is False)


# --------------------------------------------------------------------------
# the gauntlet
# --------------------------------------------------------------------------


class Gauntlet:
    """Runs every test against one panel. Judges only if calibrated."""

    def __init__(
        self,
        panel: Panel,
        *,
        thresholds: GauntletThresholds | None = None,
        horizon: int = 1,
        n_folds: int = 5,
        calibration: NullCalibration | None = None,
        panel_snapshot: str | None = None,
    ) -> None:
        self.panel = panel
        self.thresholds = thresholds or GauntletThresholds()
        self.horizon = horizon
        self.n_folds = n_folds
        self.calibration = calibration

        if calibration is not None:
            calibration.assert_applies_to(
                panel, horizon=horizon, panel_snapshot=panel_snapshot
            )

        self._forward = {horizon: rank_forward(panel, horizon)}
        self._volatility = _daily_dispersion(panel)

    # -- running -----------------------------------------------------------

    def run(self, tree: Node) -> GauntletResult:
        """Measure everything. Judge only where a threshold exists."""
        result = evaluate(tree, self.panel)
        ranked = rank_rows(result.values)
        ranked[: result.warmup] = np.nan

        ic_series = daily_rank_ic(ranked, self._forward[self.horizon])
        overall = _mean(ic_series)
        folds = purged_folds(
            self.panel.n_days,
            warmup=result.warmup,
            horizon=self.horizon,
            n_folds=self.n_folds,
        )

        outcomes = (
            self._magnitude(overall),
            *self._folds(ic_series, overall, folds),
            self._delay(tree, overall),
            *self._jitter(tree, overall),
            self._regimes(ic_series, overall),
            self._breadth(ic_series, overall),
        )
        return GauntletResult(tree=tree, outcomes=outcomes, n_folds=len(folds))

    # -- the tests ---------------------------------------------------------

    def _magnitude(self, overall: float) -> TestOutcome:
        """|IC| against the threshold the null measured."""
        statistic = abs(overall) if np.isfinite(overall) else 0.0
        return _judge("ic_magnitude", statistic, self.thresholds.min_abs_ic, above=True)

    def _folds(
        self, ic_series: np.ndarray, overall: float, folds: Sequence[Fold]
    ) -> tuple[TestOutcome, TestOutcome]:
        """Sign stability and magnitude stability across purged folds.

        Sign stability is the test that only exists because sign was decoupled
        from identity. While the hash carried the sign, a candidate and its
        negation were different hypotheses, so asking whether the sign held
        was asking whether the candidate was itself. With sign an attribute,
        the question becomes answerable and it is a sharp one: a candidate
        whose IC changes sign between folds fitted its direction to noise, and
        it is dead whatever its aggregate magnitude says.
        """
        fold_ics = [_mean(ic_series[f.start : f.stop]) for f in folds]
        usable = [v for v in fold_ics if np.isfinite(v)]
        if not usable or not np.isfinite(overall) or overall == 0.0:
            return (
                _judge("fold_sign", 0.0, self.thresholds.min_fold_sign_agreement, True),
                _judge("fold_ic", 0.0, self.thresholds.min_fold_ic_fraction, True),
            )

        direction = np.sign(overall)
        agreement = float(np.mean([np.sign(v) == direction for v in usable]))
        worst = min(abs(v) for v in usable) / abs(overall)

        detail = {f"fold_{i}": float(v) for i, v in enumerate(fold_ics)}
        return (
            _judge(
                "fold_sign",
                agreement,
                self.thresholds.min_fold_sign_agreement,
                above=True,
                detail=detail,
            ),
            _judge(
                "fold_ic",
                float(worst),
                self.thresholds.min_fold_ic_fraction,
                above=True,
                detail=detail,
            ),
        )

    def _delay(self, tree: Node, overall: float) -> TestOutcome:
        """Is the decay a decay, or a cliff at the first step?

        Normalised by the signal's own decay rather than bucketed by horizon.
        A monthly alpha and a three-day alpha lose very different amounts to a
        one-day delay, and both are behaving correctly; what neither should do
        is fall off a cliff at delay one and then decay normally, which is the
        signature of same-day information.

        The statistic is the first decrement over the second. Smooth decay
        gives something near one. A cliff gives something large.
        """
        ics = []
        for delay in DELAY_STEPS:
            delayed = self._delayed_ic(tree, delay)
            ics.append(delayed)

        detail = {f"delay_{d}": float(v) for d, v in zip(DELAY_STEPS, ics, strict=True)}
        if any(not np.isfinite(v) for v in ics) or ics[0] == 0.0:
            return _judge("delay_shape", np.inf, self.thresholds.max_delay_cliff,
                          above=False, detail=detail)

        direction = np.sign(ics[0])
        aligned = [float(v) * direction for v in ics]
        first = aligned[0] - aligned[1]
        second = aligned[1] - aligned[2]
        if second <= 0.0:
            # Still rising or flat at the second step. Not a cliff.
            statistic = 0.0 if first <= 0.0 else float(first / max(aligned[0], 1e-12))
        else:
            statistic = float(first / second)
        return _judge("delay_shape", statistic, self.thresholds.max_delay_cliff,
                      above=False, detail=detail)

    def _jitter(self, tree: Node, overall: float) -> tuple[TestOutcome, TestOutcome]:
        """A plateau across the neighbourhood, not a spike.

        Deliberately not "IC exceeds tau at every jittered setting". That test
        fails every real alpha whose IC sits near tau, because half its
        neighbourhood lands below by construction, and it fails them for a
        property of where the threshold happens to be rather than of the
        signal.

        What should hold is that the neighbourhood is flat in shape: the IC
        keeps a decent fraction of its centre value across plus or minus
        twenty per cent of every window, and never changes sign.
        """
        neighbours = list(_jittered(tree))
        detail: dict[str, float] = {"n_neighbours": float(len(neighbours))}
        if not neighbours or not np.isfinite(overall) or overall == 0.0:
            return (
                _judge("jitter_plateau", 1.0, self.thresholds.min_jitter_plateau, True,
                       detail),
                _judge("jitter_sign", 0.0, self.thresholds.max_jitter_sign_flips, False,
                       detail),
            )

        direction = np.sign(overall)
        ratios: list[float] = []
        flips = 0
        for probe in neighbours:
            value = self._plain_ic(probe)
            if not np.isfinite(value):
                continue
            ratios.append(float(value * direction / abs(overall)))
            if np.sign(value) != direction:
                flips += 1

        plateau = min(ratios) if ratios else 0.0
        detail["worst_ratio"] = float(plateau)
        detail["flips"] = float(flips)
        return (
            _judge("jitter_plateau", float(plateau),
                   self.thresholds.min_jitter_plateau, above=True, detail=detail),
            _judge("jitter_sign", float(flips),
                   self.thresholds.max_jitter_sign_flips, above=False, detail=detail),
        )

    def _regimes(self, ic_series: np.ndarray, overall: float) -> TestOutcome:
        """Does the direction hold in both volatility regimes?

        Split on realised cross-sectional dispersion rather than on calendar,
        because a date range is a poor proxy for the thing that matters.
        """
        calm, rough = regime_split(self._volatility.tolist())
        values = []
        for group in (calm, rough):
            if group:
                values.append(_mean(ic_series[np.array(group, dtype=np.int64)]))
        detail = {
            "calm": float(values[0]) if values else float("nan"),
            "rough": float(values[1]) if len(values) > 1 else float("nan"),
        }
        usable = [v for v in values if np.isfinite(v)]
        if len(usable) < 2 or not np.isfinite(overall) or overall == 0.0:
            return _judge("regime_sign", 0.0,
                          self.thresholds.min_regime_sign_agreement, True, detail)
        direction = np.sign(overall)
        agreement = float(np.mean([np.sign(v) == direction for v in usable]))
        return _judge("regime_sign", agreement,
                      self.thresholds.min_regime_sign_agreement, True, detail)

    def _breadth(self, ic_series: np.ndarray, overall: float) -> TestOutcome:
        """How much of the IC survives dropping its best tenth of days?

        A signal that earns everything on twenty days out of two thousand is
        not an alpha, it is an event study, and it will not survive the twenty
        days not repeating.
        """
        scored = ic_series[np.isfinite(ic_series)]
        if scored.size < 20 or not np.isfinite(overall) or overall == 0.0:
            return _judge("breadth", 0.0, self.thresholds.min_breadth, True)
        direction = np.sign(overall)
        aligned = scored * direction
        cut = int(np.ceil(0.1 * aligned.size))
        trimmed = np.sort(aligned)[:-cut]
        statistic = float(trimmed.mean() / abs(overall))
        return _judge("breadth", statistic, self.thresholds.min_breadth, above=True,
                      detail={"n_days": float(scored.size), "dropped": float(cut)})

    # -- helpers -----------------------------------------------------------

    def _plain_ic(self, tree: Node) -> float:
        try:
            result = evaluate(tree, self.panel)
        except (EvaluationError, ValueError, FloatingPointError):
            return float("nan")
        ranked = rank_rows(result.values)
        ranked[: result.warmup] = np.nan
        return _mean(daily_rank_ic(ranked, self._forward[self.horizon]))

    def _delayed_ic(self, tree: Node, delay: int) -> float:
        """IC of the signal applied ``delay`` days later than it was known."""
        try:
            result = evaluate(tree, self.panel)
        except (EvaluationError, ValueError, FloatingPointError):
            return float("nan")
        ranked = rank_rows(result.values)
        ranked[: result.warmup] = np.nan
        if delay:
            shifted = np.full_like(ranked, np.nan)
            shifted[delay:] = ranked[:-delay]
            ranked = shifted
        return _mean(daily_rank_ic(ranked, self._forward[self.horizon]))


# --------------------------------------------------------------------------
# free functions
# --------------------------------------------------------------------------


def _judge(
    name: str,
    statistic: float,
    threshold: float | int | None,
    above: bool,
    detail: Mapping[str, float] | None = None,
) -> TestOutcome:
    """Measure always. Judge only when a threshold has been measured."""
    passed: bool | None = None
    if threshold is not None:
        passed = statistic >= threshold if above else statistic <= threshold
    return TestOutcome(
        name=name,
        statistic=float(statistic),
        threshold=None if threshold is None else float(threshold),
        passed=passed,
        detail=dict(detail or {}),
    )


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _jittered(tree: Node):
    """Every single-parameter perturbation of the tree, as probes.

    One parameter at a time, because moving several at once measures a
    direction in parameter space rather than the sensitivity to each.
    """
    nodes = tree.nodes()
    for index, node in enumerate(nodes):
        if not node.params:
            continue
        for which, value in enumerate(node.params):
            for fraction in JITTER_FRACTIONS:
                moved = int(round(value * (1.0 + fraction)))
                if moved == value or moved < 1:
                    continue
                params = list(node.params)
                params[which] = moved
                try:
                    probe = Node(node.op, node.children, tuple(params), True)
                    yield tree.replace_at(index, probe) if index else probe
                except ValueError:
                    continue


def _daily_dispersion(panel: Panel) -> np.ndarray:
    """Cross-sectional standard deviation of returns, per day."""
    returns = np.asarray(panel["returns"], dtype=np.float64)
    present = ~np.isnan(returns)
    counts = np.count_nonzero(present, axis=1)
    centred = np.where(present, returns, 0.0)
    mean = centred.sum(axis=1) / np.maximum(counts, 1)
    centred = np.where(present, returns - mean[:, None], 0.0)
    variance = np.einsum("ij,ij->i", centred, centred) / np.maximum(counts - 1, 1)
    return np.where(counts >= MIN_CROSS_SECTION, np.sqrt(variance), np.nan)
