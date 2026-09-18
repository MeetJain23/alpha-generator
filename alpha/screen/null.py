"""What the screen's statistics look like when there is nothing there.

Every threshold in the gauntlet is a statement about how unusual a number is,
and "unusual" has no meaning until the distribution under no effect is known.
Guessing a threshold is guessing the answer.

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
many *effectively independent* instruments there are, and equities are
correlated enough that three thousand names are worth far fewer. A null that
destroyed that would be far too tight, and the threshold read from it would
admit far more noise than intended.

Blocks rather than a fresh permutation each day, so the day-to-day dependence
of the IC series survives. That dependence is what sets the standard error of
the mean IC, and permuting independently every day would understate it.

Two nulls that do not work
--------------------------
Both were built and measured before this one, and the measurements are the
reason for the design.

*Resampling the panel itself*, taking contiguous blocks of days, moves a
signal and its future return together. Inside a block, a signal computed from
days entirely within that block is the original signal and the return that
follows is the original return, so the relationship survives everywhere except
at the seams. With a forty-day block and a twenty-day window that is most of
the sample, and a genuinely predictive random tree would still score,
contaminating exactly the tail the threshold is read from.

*Permuting the returns in time*, leaving the panel alone, is worse, and it
fails in a way that is not obvious until it is measured. ``close[t]`` is the
cumulative product of the returns up to *t*. Pairing it with returns from an
earlier day scores it against returns it mechanically contains, which is a
backward leak manufactured by the permutation itself. Measured on a synthetic
panel, ``close`` scored a mean null IC of +0.0101 and ``mcap`` +0.0083 under
that null, against +0.0006 and -0.0002 under this one, while signals not
derived from cumulative prices, ``volume`` and ``returns``, sat at zero under
both. The bias tracked exactly whether a signal was a function of past
returns.

That failure is worth recording because of how it would have presented. The
null would have been biased upward, tau would have been set too high, and the
search would have silently discarded real candidates while reporting that it
was being appropriately strict.

One consequence of this null is recorded rather than hidden. Instrument *i* is
listed on days when *pi(i)* is not, so those cells do not score, and the
effective cross-section shrinks. The calibration reports the scored-day count
so the shrinkage is visible rather than assumed away.

What it produces
----------------
Three things, and the gauntlet cannot be written without them.

``tau``, the |IC| threshold, read off the null at the quantile that admits the
intended fraction of noise. Set from the distribution rather than chosen.

``sigma_sr``, the dispersion of the IC-IR under the null. The analytic
Deflated Sharpe assumes independent trials. Grammar-generated candidates share
subtrees, reuse fields and overlap in windows, so they are heavily correlated
and the analytic value is simply wrong for this search. The empirical one is
what the deflation should use.

A leak check. Under this null the mean signed IC must sit at zero. If it does
not, something upstream is leaking, and finding that out here, on data where
the answer is known to be nothing, is far cheaper than finding it in a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from alpha.data.types import DTYPE, Panel
from alpha.expr.ast import random_tree, strip_elidable_root
from alpha.expr.evaluator import EvaluationError, evaluate
from alpha.logging_config import get_logger
from alpha.screen.metrics import rank_forward, summarise

_log = get_logger(__name__)

DEFAULT_BLOCK: Final[int] = 40
"""Days per block. Between twenty and sixty covers the horizon over which
daily equity returns carry serial structure."""

LEAK_SIGMAS: Final[float] = 4.0
"""Standard errors away from zero before the mean null IC is called a leak.

Four rather than two, because this runs over tens of thousands of trials and a
two-sigma rule would cry leak on a clean system regularly enough to be
ignored, which is the failure mode of every alarm.
"""


def block_label_permutation(
    n_days: int, n_instruments: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Which column supplies each cell's return, per day.

    One permutation of the instrument labels per block of days, held fixed
    across the block. Returns an index array to be used with
    ``np.take_along_axis`` on the forward returns.

    Holding the permutation fixed within a block is what preserves the
    day-to-day dependence of the IC series. Redrawing every day would make
    successive ICs independent and understate the standard error of their
    mean, which is the quantity the leak check is measured against.
    """
    if n_days <= 0 or n_instruments <= 0:
        raise ValueError("a panel needs at least one day and one instrument")
    if not 1 <= block_length <= n_days:
        raise ValueError(
            f"block length {block_length} must be between 1 and {n_days}"
        )

    out = np.empty((n_days, n_instruments), dtype=np.int64)
    start = 0
    while start < n_days:
        stop = min(start + block_length, n_days)
        out[start:stop] = rng.permutation(n_instruments)
        start = stop
    return out


@dataclass(frozen=True, slots=True)
class NullCalibration:
    """The null distribution of the screen's statistics, and what it implies."""

    n_trials: int
    abs_ic_quantiles: dict[str, float]
    tau: float
    tau_quantile: float
    sigma_sr: float
    mean_ic: float
    mean_ic_standard_error: float
    leaks: bool
    median_scored_days: float
    abs_ic: np.ndarray = field(repr=False)
    ic_ir: np.ndarray = field(repr=False)

    @property
    def leak_sigmas(self) -> float:
        """How far the mean sits from zero, in standard errors."""
        if self.mean_ic_standard_error <= 0.0:
            return 0.0
        return abs(self.mean_ic) / self.mean_ic_standard_error


def calibrate(
    panel: Panel,
    rng: np.random.Generator,
    *,
    n_trials: int = 10_000,
    max_depth: int = 4,
    block_length: int = DEFAULT_BLOCK,
    horizon: int = 1,
    admit_fraction: float = 0.001,
    min_dispersion: float = 0.01,
    min_scored_days: int = 250,
) -> NullCalibration:
    """Run random trees against block-shuffled returns and keep every number.

    Each trial draws its own tree and its own bootstrap ordering, so a draw is
    a pair rather than a tree scored many times. That samples the joint
    distribution the search actually faces and costs one IC per trial rather
    than one per trial per replicate.

    The whole ``|IC|`` distribution is returned, not only its tail. A
    threshold is read from the tail, but a distribution that is the wrong
    shape in the middle is the first sign that the null is not a null, and
    that is invisible if only the maximum is kept.

    A trial is admitted to the null only if the screen would have admitted
    it. Degenerate signals are skipped, because a flat signal scores an
    undefined IC and letting those in would pull the distribution towards zero
    and set a threshold that is too easy. Trials with too few scored days are
    skipped for the opposite reason, and that filter does more work than it
    sounds: in a run of fifteen hundred, the single largest null |IC| was
    0.111 against a 99.9th percentile of 0.010, and it was scored on three
    days. One trial was setting the threshold at ten times its right value.

    Both filters exist so the null is the distribution of the statistic *as
    the screen computes and admits it*. A null calibrated on trials the screen
    would have thrown out is not calibrating the screen.
    """
    ranked_forward = rank_forward(panel, horizon)

    abs_ic: list[float] = []
    signed_ic: list[float] = []
    ic_ir: list[float] = []
    scored_days: list[int] = []
    skipped = 0
    failed = 0

    for index in range(n_trials):
        tree = random_tree(rng, max_depth)
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
        if np.isfinite(metrics.ic_ir):
            ic_ir.append(metrics.ic_ir)

        if (index + 1) % 1000 == 0:
            _log.info(
                "null calibration progress",
                extra={"trials": index + 1, "kept": len(abs_ic)},
            )

    if len(abs_ic) < 100:
        raise RuntimeError(
            f"only {len(abs_ic)} usable trials out of {n_trials}; too few to "
            f"calibrate anything. {skipped} were degenerate and {failed} "
            f"failed to evaluate."
        )

    absolute = np.array(abs_ic, dtype=np.float64)
    signed = np.array(signed_ic, dtype=np.float64)
    ratios = np.array(ic_ir, dtype=np.float64)

    quantiles = {
        f"p{q}": float(np.quantile(absolute, q / 100.0))
        for q in (50, 75, 90, 95, 99, 99.9)
    }
    quantiles["max"] = float(absolute.max())

    tau_quantile = 1.0 - admit_fraction
    tau = float(np.quantile(absolute, tau_quantile))

    mean_ic = float(signed.mean())
    standard_error = float(signed.std(ddof=1) / np.sqrt(signed.size))

    calibration = NullCalibration(
        n_trials=int(absolute.size),
        abs_ic_quantiles=quantiles,
        tau=tau,
        tau_quantile=tau_quantile,
        sigma_sr=float(ratios.std(ddof=1)) if ratios.size > 1 else float("nan"),
        mean_ic=mean_ic,
        mean_ic_standard_error=standard_error,
        leaks=abs(mean_ic) > LEAK_SIGMAS * standard_error if standard_error > 0 else False,
        median_scored_days=float(np.median(scored_days)),
        abs_ic=absolute,
        ic_ir=ratios,
    )
    _log.info(
        "null calibrated",
        extra={
            "trials": calibration.n_trials,
            "tau": round(calibration.tau, 5),
            "sigma_sr": round(calibration.sigma_sr, 4),
            "leak_sigmas": round(calibration.leak_sigmas, 2),
        },
    )
    return calibration
