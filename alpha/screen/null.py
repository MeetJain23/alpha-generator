"""What the screen's statistics look like when there is nothing there.

Every threshold downstream is a statement about how unusual a number is, and
"unusual" has no meaning until the distribution under no effect is known.
Guessing a threshold is guessing the answer.

These numbers are properties of a dataset, not of markets
---------------------------------------------------------
A calibration is a measurement taken on one panel. Run it on a synthetic panel
and every constant it produces describes the generator: its Gaussian returns,
its constant volatility, its independent instruments. Real equity returns have
fat tails, volatility clustering, cross-sectional dispersion that varies
enormously between 1997 and 2017, and sector block correlation that makes the
effective breadth of a three thousand name universe far smaller than three
thousand, and none of that is in a generated panel unless it was put there.

A tau measured on synthetic data and applied to real data is a number with no
relationship to the thing it is thresholding. So a calibration carries the
``data_snapshot_id`` of the panel it was measured on, and
``NullCalibration.assert_applies_to`` refuses a mismatch. The guard exists for
the same reason ``source_hash`` does: the failure is silent, so it has to be
made impossible rather than documented.

The null
--------
Within each block of days, the instrument labels of the forward returns are
permuted, and the signal panel is left untouched. Instrument *i*'s signal is
scored against instrument *pi(i)*'s return, with one permutation per block.

This destroys the pairing completely, in both directions. There is no time
alignment left between a signal and the return it is scored against, and no
static cross-sectional relationship either, because a name's signal now meets
a different name's returns.

It preserves everything the shape of the null depends on. A day's return
cross-section is intact, because a permutation only relabels it: the same
returns are present, so the day's dispersion and any market-wide move survive.
Cross-instrument correlation survives for the same reason, which matters
because the sampling distribution of a daily cross-sectional IC depends on how
many *effectively independent* instruments there are.

Blocks rather than a fresh permutation each day, so the day-to-day dependence
of the IC series survives. That dependence is what sets the standard error of
the mean IC, and permuting independently every day would understate it. The
block length is recorded on the calibration, because it is a choice and the
numbers depend on it.

The candidate source has to be the real one
-------------------------------------------
``calibrate`` takes the function that produces candidates, and it must be the
same one the search will use.

The maximum of N statistics depends on how correlated those N are, and
grammar-generated candidates are heavily correlated: they share subtrees,
reuse the same eight fields and draw windows from the same short ladder. An
evolutionary search is more correlated still, because it breeds from
survivors. Calibrating on uniform random trees and then running a genetic
search gives a maximum drawn from a different distribution than the one that
was measured, and in the direction that matters: a more correlated search has
a *smaller* effective number of trials, so an uncorrelated calibration
overstates the bar and discards real candidates.

The source is recorded by name on the calibration so the mismatch is visible.

Two nulls that do not work
--------------------------
Both were built and measured before this one, and the measurements are the
reason for the design.

*Resampling the panel itself*, taking contiguous blocks of days, moves a
signal and its future return together. Inside a block, a signal computed from
days entirely within that block is the original signal and the return that
follows is the original return, so the relationship survives everywhere except
at the seams.

*Permuting the returns in time*, leaving the panel alone, is worse, and it
fails in a way that is not obvious until it is measured. ``close[t]`` is the
cumulative product of the returns up to *t*. Pairing it with returns from an
earlier day scores it against returns it mechanically contains. Measured on a
synthetic panel, ``close`` scored a mean null IC of +0.0101 and ``mcap``
+0.0083 under that null, against +0.0006 and -0.0002 under this one, while
signals not derived from cumulative prices sat at zero under both. The null
came out biased upward and tau three times too high, which would have
discarded real candidates while reporting appropriate strictness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Final, Mapping

import numpy as np

from alpha.data.cache import snapshot_id
from alpha.data.types import DTYPE, Panel
from alpha.expr.ast import Node, random_tree, strip_elidable_root
from alpha.expr.evaluator import EvaluationError, evaluate
from alpha.logging_config import get_logger
from alpha.registry.db import grammar_fingerprint
from alpha.screen.metrics import rank_forward, summarise

_log = get_logger(__name__)

DEFAULT_BLOCK: Final[int] = 40
"""Days per block for the label permutation.

Between twenty and sixty covers the horizon over which daily equity returns
carry serial structure. Forty is the middle of that range. It is a choice, it
changes the numbers, and it is recorded on every calibration for that reason.
"""

LEAK_SIGMAS: Final[float] = 4.0
"""Standard errors from zero before the mean null IC is called a leak.

Four rather than two, because this runs over tens of thousands of trials and a
two-sigma rule would cry leak on a clean system often enough to be ignored,
which is the failure mode of every alarm.
"""

BATCH_LADDER: Final[tuple[int, ...]] = (10, 32, 100, 316, 1000, 3162, 10_000, 31_623)
"""Search sizes at which the maximum is measured, roughly half a decade apart.

The bar a search has to clear depends on how many candidates it tried, so what
is wanted is not one number but the curve. Measuring it at a ladder of sizes
and interpolating in log space is honest about that; quoting a single
``E[max]`` is not, because it silently assumes a search size nobody stated.
"""

MIN_BATCHES: Final[int] = 8
"""Disjoint batches needed before a maximum's distribution means anything."""

CandidateSource = Callable[[np.random.Generator], Node]
"""Produces one candidate. Must be the source the real search will use."""


class CalibrationMismatch(RuntimeError):
    """A calibration is being applied to something it does not describe."""


# --------------------------------------------------------------------------
# the permutation
# --------------------------------------------------------------------------


def block_label_permutation(
    n_days: int, n_instruments: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Which column supplies each cell's return, per day.

    One permutation of the instrument labels per block of days, held fixed
    across the block. Returns an index array for ``np.take_along_axis``.

    Holding it fixed within a block preserves the day-to-day dependence of the
    IC series. Redrawing every day would make successive ICs independent and
    understate the standard error of their mean, which is the quantity the
    leak check is measured against.
    """
    if n_days <= 0 or n_instruments <= 0:
        raise ValueError("a panel needs at least one day and one instrument")
    if not 1 <= block_length <= n_days:
        raise ValueError(f"block length {block_length} must be in 1..{n_days}")

    out = np.empty((n_days, n_instruments), dtype=np.int64)
    start = 0
    while start < n_days:
        stop = min(start + block_length, n_days)
        out[start:stop] = rng.permutation(n_instruments)
        start = stop
    return out


# --------------------------------------------------------------------------
# what a calibration is
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaximumCurve:
    """The distribution of the maximum statistic, by search size.

    Not a point estimate. The bar depends on how many candidates were tried,
    and a single number hides the search size it assumed.
    """

    sizes: tuple[int, ...]
    mean: tuple[float, ...]
    p50: tuple[float, ...]
    p95: tuple[float, ...]
    largest: tuple[float, ...]
    n_batches: tuple[int, ...]

    def expected_at(self, search_size: int) -> float:
        """Interpolate the mean maximum in log search size.

        Extrapolation beyond the measured ladder is refused. The growth of a
        maximum is slow but it is not a straight line forever, and guessing
        past the data is exactly the substitution this whole module exists to
        remove.
        """
        if not self.sizes:
            return float("nan")
        if search_size < self.sizes[0] or search_size > self.sizes[-1]:
            raise CalibrationMismatch(
                f"search size {search_size} is outside the measured ladder "
                f"{self.sizes[0]}..{self.sizes[-1]}. Run the calibration with "
                f"enough trials to cover it rather than extrapolating."
            )
        return float(
            np.interp(
                math.log(search_size),
                [math.log(size) for size in self.sizes],
                self.mean,
            )
        )


@dataclass(frozen=True, slots=True)
class NullCalibration:
    """The null distribution of the screen's statistics, and what it describes.

    The binding fields are not metadata. They say which panel, grammar,
    horizon and candidate source these numbers describe, and applying them to
    anything else produces a threshold with no relationship to what it is
    thresholding.
    """

    data_snapshot_id: str
    grammar_fingerprint: str
    block_length: int
    horizon: int
    source_name: str
    panel_shape: tuple[int, int]

    n_trials: int
    abs_ic_quantiles: Mapping[str, float]
    tau: float
    tau_quantile: float
    sigma_sr: float
    """Standard deviation of the signed IC-IR under the null.

    Signed, not absolute. It is the dispersion of the statistic, and halving
    it by taking absolute values first would make the analytic comparator look
    far smaller than the measured maximum for no reason but the accounting.
    """

    mean_ic: float
    mean_ic_standard_error: float
    leaks: bool
    median_scored_days: float

    max_abs_ic: MaximumCurve
    max_ic_ir: MaximumCurve

    abs_ic: np.ndarray = field(repr=False)
    ic_ir: np.ndarray = field(repr=False)

    @property
    def leak_sigmas(self) -> float:
        if self.mean_ic_standard_error <= 0.0:
            return 0.0
        return abs(self.mean_ic) / self.mean_ic_standard_error

    def analytic_max_ic_ir(self, search_size: int) -> float:
        """``sigma_SR * sqrt(2 ln N)``, kept only as a sanity check.

        The False Strategy Theorem was a substitute for an experiment nobody
        could run. It assumes the trials are independent and their Sharpes
        Gaussian, and neither holds for a grammar search whose candidates
        share subtrees. It is reported beside the empirical curve so a wild
        divergence is visible, and it is not what anything decides on. See
        ``docs/REPRODUCIBILITY.md``.
        """
        if search_size < 2 or not np.isfinite(self.sigma_sr):
            return float("nan")
        return float(self.sigma_sr * math.sqrt(2.0 * math.log(search_size)))

    def binding(self) -> dict[str, object]:
        """The fields that say what these numbers describe, for the run config."""
        return {
            "null_data_snapshot_id": self.data_snapshot_id,
            "null_grammar_fingerprint": self.grammar_fingerprint,
            "null_block_length": self.block_length,
            "null_horizon": self.horizon,
            "null_source": self.source_name,
            "null_trials": self.n_trials,
            "null_tau": self.tau,
            "null_sigma_sr": self.sigma_sr,
        }

    def assert_applies_to(
        self, panel: Panel, *, horizon: int, panel_snapshot: str | None = None
    ) -> None:
        """Refuse to be used on anything this calibration does not describe.

        The snapshot check is the one that matters. A tau measured on a
        synthetic panel and applied to real data is a number with no
        relationship to what it is thresholding, and nothing in the output
        would look wrong: the screen would run, report a pass rate, and every
        number in it would be meaningless.

        The grammar fingerprint is checked for the same reason a run records
        it. If the operator set changed, the space of candidates changed, and
        the distribution of the maximum over that space changed with it.
        """
        actual = panel_snapshot if panel_snapshot is not None else snapshot_id(panel)
        if actual != self.data_snapshot_id:
            raise CalibrationMismatch(
                f"this calibration was measured on snapshot "
                f"{self.data_snapshot_id} and the panel is {actual}. Its tau "
                f"and sigma_SR describe a different dataset, so applying them "
                f"here would threshold against a distribution that was never "
                f"measured. Recalibrate on this panel."
            )
        current = grammar_fingerprint()
        if current != self.grammar_fingerprint:
            raise CalibrationMismatch(
                f"this calibration was measured under grammar "
                f"{self.grammar_fingerprint[:12]} and the grammar is now "
                f"{current[:12]}. The candidate space changed, so the "
                f"distribution of the maximum over it changed too."
            )
        if horizon != self.horizon:
            raise CalibrationMismatch(
                f"this calibration scores at horizon {self.horizon} and the "
                f"screen is scoring at {horizon}."
            )


# --------------------------------------------------------------------------
# measuring it
# --------------------------------------------------------------------------


def _maximum_curve(values: np.ndarray) -> MaximumCurve:
    """Distribution of the maximum over disjoint batches, at each ladder size."""
    sizes: list[int] = []
    mean: list[float] = []
    p50: list[float] = []
    p95: list[float] = []
    largest: list[float] = []
    counts: list[int] = []

    for size in BATCH_LADDER:
        batches = values.size // size
        if batches < MIN_BATCHES:
            break
        block = values[: batches * size].reshape(batches, size)
        maxima = block.max(axis=1)
        sizes.append(size)
        mean.append(float(maxima.mean()))
        p50.append(float(np.quantile(maxima, 0.50)))
        p95.append(float(np.quantile(maxima, 0.95)))
        largest.append(float(maxima.max()))
        counts.append(int(batches))

    return MaximumCurve(
        sizes=tuple(sizes),
        mean=tuple(mean),
        p50=tuple(p50),
        p95=tuple(p95),
        largest=tuple(largest),
        n_batches=tuple(counts),
    )


def calibrate(
    panel: Panel,
    rng: np.random.Generator,
    *,
    source: CandidateSource | None = None,
    source_name: str | None = None,
    n_trials: int = 10_000,
    max_depth: int = 4,
    block_length: int = DEFAULT_BLOCK,
    horizon: int = 1,
    admit_fraction: float = 0.001,
    min_dispersion: float = 0.01,
    min_scored_days: int = 250,
    panel_snapshot: str | None = None,
) -> NullCalibration:
    """Score candidates against label-permuted returns and keep every number.

    ``source`` must be the function the real search will use. The default is
    uniform random trees, which is correct only if the search is also uniform
    random trees; anything else and the correlation structure differs, and so
    does the maximum.

    Each trial draws its own candidate and its own permutation, so a draw is a
    pair rather than one candidate scored repeatedly. That samples the joint
    distribution the search actually faces at the cost of one IC per trial.

    A trial is admitted only if the screen would have admitted it. Degenerate
    signals score an undefined IC and would pull the distribution towards
    zero; trials with too few scored days do the opposite, and that filter
    does more work than it sounds. In a run of fifteen hundred, the single
    largest null |IC| was 0.111 against a 99.9th percentile of 0.010, and it
    was scored on three days. A null calibrated on trials the screen would
    have thrown out is not calibrating the screen.
    """
    draw = source if source is not None else (lambda r: random_tree(r, max_depth))
    name = source_name or (
        "random_tree" if source is None else getattr(source, "__name__", "custom")
    )
    identity = panel_snapshot if panel_snapshot is not None else snapshot_id(panel)
    ranked_forward = rank_forward(panel, horizon)

    abs_ic: list[float] = []
    signed_ic: list[float] = []
    ic_ir: list[float] = []
    scored_days: list[int] = []
    skipped = 0
    failed = 0

    for index in range(n_trials):
        tree = draw(rng)
        to_evaluate, _ = strip_elidable_root(tree)
        try:
            result = evaluate(to_evaluate, panel)
        except (EvaluationError, ValueError, FloatingPointError):
            failed += 1
            continue

        labels = block_label_permutation(
            panel.n_days, panel.n_instruments, block_length, rng
        )
        shuffled = np.ascontiguousarray(
            np.take_along_axis(ranked_forward, labels, axis=1), dtype=DTYPE
        )

        metrics, _ = summarise(result.values, result.warmup, panel, shuffled)
        if (
            metrics.dispersion < min_dispersion
            or not np.isfinite(metrics.ic)
            or metrics.n_scored_days < min_scored_days
        ):
            skipped += 1
            continue

        signed_ic.append(metrics.ic)
        abs_ic.append(abs(metrics.ic))
        scored_days.append(metrics.n_scored_days)
        # Signed. sigma_SR is the dispersion of the statistic itself, and
        # taking absolute values first would halve it and make the analytic
        # comparator look far too small. The maximum curve takes the absolute
        # value afterwards, where it belongs.
        ic_ir.append(metrics.ic_ir if np.isfinite(metrics.ic_ir) else np.nan)

        if (index + 1) % 5000 == 0:
            _log.info(
                "null calibration progress",
                extra={"trials": index + 1, "kept": len(abs_ic)},
            )

    if len(abs_ic) < 100:
        raise RuntimeError(
            f"only {len(abs_ic)} usable trials out of {n_trials}; too few to "
            f"calibrate anything. {skipped} were degenerate or too thin and "
            f"{failed} failed to evaluate."
        )

    absolute = np.array(abs_ic, dtype=np.float64)
    signed = np.array(signed_ic, dtype=np.float64)
    ratios = np.array(ic_ir, dtype=np.float64)
    finite_ratios = ratios[np.isfinite(ratios)]

    quantiles = {
        f"p{q}": float(np.quantile(absolute, q / 100.0))
        for q in (50, 75, 90, 95, 99, 99.9)
    }
    quantiles["max"] = float(absolute.max())

    mean_ic = float(signed.mean())
    standard_error = float(signed.std(ddof=1) / np.sqrt(signed.size))

    calibration = NullCalibration(
        data_snapshot_id=identity,
        grammar_fingerprint=grammar_fingerprint(),
        block_length=block_length,
        horizon=horizon,
        source_name=name,
        panel_shape=(panel.n_days, panel.n_instruments),
        n_trials=int(absolute.size),
        abs_ic_quantiles=quantiles,
        tau=float(np.quantile(absolute, 1.0 - admit_fraction)),
        tau_quantile=1.0 - admit_fraction,
        sigma_sr=(
            float(finite_ratios.std(ddof=1)) if finite_ratios.size > 1 else float("nan")
        ),
        mean_ic=mean_ic,
        mean_ic_standard_error=standard_error,
        leaks=(abs(mean_ic) > LEAK_SIGMAS * standard_error) if standard_error > 0 else False,
        median_scored_days=float(np.median(scored_days)),
        max_abs_ic=_maximum_curve(absolute),
        max_ic_ir=_maximum_curve(np.abs(finite_ratios)),
        abs_ic=absolute,
        ic_ir=ratios,
    )
    _log.info(
        "null calibrated",
        extra={
            "snapshot": identity,
            "source": name,
            "block_length": block_length,
            "trials": calibration.n_trials,
            "tau": round(calibration.tau, 5),
            "sigma_sr": round(calibration.sigma_sr, 4),
            "leak_sigmas": round(calibration.leak_sigmas, 2),
        },
    )
    return calibration
