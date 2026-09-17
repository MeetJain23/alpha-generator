"""Signal metrics: forward returns, rank IC, turnover, coverage, deciles.

Everything here is rank-based, which is the reason the screen can afford to
treat a whole family of expressions as one hypothesis. Rank IC, decile
membership and turnover are all invariant under a strictly monotone per-day
transform, so ``x`` and ``zscore(x)`` produce identical numbers from every
function in this module.

The alignment rule
------------------
A signal observed on date t is scored against the return earned on t+1. Not
t. Scoring a signal against the same day's return is the most ordinary form
of look-ahead there is, it inflates every IC in the system, and it is
invisible in the output because the numbers stay in a plausible range.

``forward_returns`` is the only place that shift happens, so there is one
line in the system to get it right and one line to check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from alpha.data.types import DTYPE, Panel
from alpha.expr.evaluator import rank_rows

TRADING_DAYS_PER_YEAR: Final[int] = 252
MIN_CROSS_SECTION: Final[int] = 20
"""Instruments a day needs before its IC means anything.

A correlation over a handful of names is mostly an artefact of which names
happened to be listed, and averaging those days into an IC series gives the
thinnest days the same weight as the fullest ones.
"""


@dataclass(frozen=True, slots=True)
class SignalMetrics:
    """What the screen decides on."""

    ic: float
    """Mean daily rank correlation between the signal and the next day's
    return."""

    ic_ir: float
    """Mean daily IC over its standard deviation, annualised. The IC that
    matters is the one that shows up consistently, not the one that was large
    once."""

    turnover: float
    """Mean fraction of the book that changes hands per day."""

    coverage: float
    """Fraction of post-warmup cells the signal actually defines, measured
    against the instruments that were listed and tradable."""

    dispersion: float
    """Median per-day standard deviation of the ranked signal. Near zero
    means the signal barely distinguishes between instruments."""

    n_scored_days: int


def forward_returns(panel: Panel, horizon: int = 1) -> np.ndarray:
    """The return earned over the ``horizon`` days after each date.

    Row t holds what an instrument returns from t to t + horizon, so a signal
    observed on t is scored against row t. Rows within ``horizon`` of the end
    are NaN, because the future they would need has not happened.

    This is the single alignment in the system. Everything that scores a
    signal goes through it rather than shifting arrays itself.
    """
    if horizon < 1:
        raise ValueError("a forward return needs a horizon of at least one day")
    daily = np.asarray(panel["returns"], dtype=np.float64)
    out = np.full(daily.shape, np.nan, dtype=np.float64)
    if horizon >= daily.shape[0]:
        return out

    compounded = np.ones_like(daily)
    for step in range(1, horizon + 1):
        compounded[: daily.shape[0] - horizon] *= (
            1.0 + daily[step : daily.shape[0] - horizon + step]
        )
    out[: daily.shape[0] - horizon] = compounded[: daily.shape[0] - horizon] - 1.0
    return out


def daily_rank_ic(
    ranked_signal: np.ndarray, ranked_forward: np.ndarray
) -> np.ndarray:
    """Per-day Spearman correlation, as a series with NaN on unusable days.

    Both inputs are already ranks, so the Spearman correlation is the Pearson
    correlation of what was passed in. Ranking once and reusing it is what
    keeps the screen affordable, since the rank is the expensive part and
    every metric here needs the same one.
    """
    usable = ~np.isnan(ranked_signal) & ~np.isnan(ranked_forward)
    counts = np.count_nonzero(usable, axis=1)

    left = np.where(usable, ranked_signal, 0.0).astype(np.float64)
    right = np.where(usable, ranked_forward, 0.0).astype(np.float64)

    n = np.maximum(counts, 1).astype(np.float64)
    mean_left = left.sum(axis=1) / n
    mean_right = right.sum(axis=1) / n

    left -= np.where(usable, mean_left[:, None], 0.0)
    right -= np.where(usable, mean_right[:, None], 0.0)

    covariance = np.einsum("ij,ij->i", left, right)
    var_left = np.einsum("ij,ij->i", left, left)
    var_right = np.einsum("ij,ij->i", right, right)

    denominator = np.sqrt(var_left * var_right)
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = np.where(
            (counts >= MIN_CROSS_SECTION) & (denominator > 0.0),
            covariance / np.where(denominator > 0.0, denominator, 1.0),
            np.nan,
        )
    return ic


def turnover(ranked_signal: np.ndarray) -> float:
    """Mean fraction of the book traded per day.

    Weights are the centred ranks scaled so the gross book is one, which is
    the natural portfolio for a ranked signal: already zero net exposure, and
    already bounded in how much it can put into any one name. Turnover is
    half the sum of absolute weight changes, so a complete reversal is 1.0
    rather than 2.0.

    A name that leaves the universe has its weight go to zero, and that
    counts. It is a trade, and pretending otherwise would let a signal that
    churns through delistings look cheap to run.
    """
    weights = _gross_normalised(ranked_signal)
    if weights.shape[0] < 2:
        return float("nan")
    change = np.abs(np.diff(weights, axis=0)).sum(axis=1) * 0.5
    traded = change[np.isfinite(change)]
    return float(np.mean(traded)) if traded.size else float("nan")


def _gross_normalised(ranked: np.ndarray) -> np.ndarray:
    """Ranks scaled so each day's absolute weights sum to one."""
    weights = np.where(np.isnan(ranked), 0.0, ranked).astype(np.float64)
    gross = np.abs(weights).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(gross > 0.0, weights / np.where(gross > 0.0, gross, 1.0), 0.0)


def coverage(values: np.ndarray, warmup: int, panel: Panel) -> float:
    """Fraction of the tradable cells the signal actually defines.

    Measured against the instruments that were listed, not against the shape
    of the array. A panel is mostly empty at its corners, so dividing by rows
    times columns would score every signal as poor and rank them all the
    same.
    """
    if warmup >= values.shape[0]:
        return 0.0
    listed = np.asarray(panel["is_listed"], dtype=np.float32)[warmup:] > 0.0
    tradable = int(np.count_nonzero(listed))
    if tradable == 0:
        return 0.0
    defined = int(np.count_nonzero(listed & ~np.isnan(values[warmup:])))
    return defined / tradable


def dispersion(ranked_signal: np.ndarray) -> float:
    """Median per-day spread of the ranked signal.

    A signal that assigns almost every instrument the same rank has nothing
    to say about which to hold, whatever its IC happens to be on the few days
    it does distinguish them. The median rather than the mean, so a handful of
    wild days cannot rescue a flat signal.
    """
    finite = ~np.isnan(ranked_signal)
    counts = np.count_nonzero(finite, axis=1)
    centred = np.where(finite, ranked_signal, 0.0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = centred.sum(axis=1) / np.maximum(counts, 1)
        centred -= np.where(finite, mean[:, None], 0.0)
        variance = np.einsum("ij,ij->i", centred, centred) / np.maximum(counts - 1, 1)
    usable = variance[counts >= MIN_CROSS_SECTION]
    if usable.size == 0:
        return 0.0
    return float(np.sqrt(np.median(usable)))


def summarise(
    values: np.ndarray,
    warmup: int,
    panel: Panel,
    ranked_forward: np.ndarray,
    *,
    ranked_signal: np.ndarray | None = None,
) -> tuple[SignalMetrics, np.ndarray]:
    """Every metric, plus the ranked signal so the caller can reuse it.

    The rank is the expensive part, so it is computed once here and handed
    back. The screen passes it on to ``value_hash`` rather than ranking the
    signal a second time.
    """
    ranked = rank_rows(values) if ranked_signal is None else ranked_signal
    blanked = ranked.copy()
    blanked[:warmup] = np.nan

    ic_series = daily_rank_ic(blanked, ranked_forward)
    scored = ic_series[~np.isnan(ic_series)]

    if scored.size < 2:
        mean_ic, ic_ir = float("nan"), float("nan")
    else:
        mean_ic = float(np.mean(scored))
        spread = float(np.std(scored, ddof=1))
        ic_ir = (
            mean_ic / spread * np.sqrt(TRADING_DAYS_PER_YEAR) if spread > 0 else float("nan")
        )

    metrics = SignalMetrics(
        ic=mean_ic,
        ic_ir=ic_ir,
        turnover=turnover(blanked),
        coverage=coverage(values, warmup, panel),
        dispersion=dispersion(blanked),
        n_scored_days=int(scored.size),
    )
    return metrics, ranked


def decile_membership(ranked_signal_row: np.ndarray, buckets: int = 10) -> list[np.ndarray]:
    """Split one day's defined instruments into equal-count buckets."""
    live = np.flatnonzero(~np.isnan(ranked_signal_row))
    if live.size < buckets:
        return []
    order = live[np.argsort(ranked_signal_row[live], kind="stable")]
    return [np.asarray(part) for part in np.array_split(order, buckets)]


def rank_forward(panel: Panel, horizon: int = 1) -> np.ndarray:
    """Ranked forward returns, which every candidate is scored against.

    Computed once per panel and reused across the whole run. It does not
    depend on the candidate, and ranking a full panel is not something to do
    a million times.
    """
    return rank_rows(np.asarray(forward_returns(panel, horizon), dtype=DTYPE))
