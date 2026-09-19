"""Purged time splits.

A candidate that survives on the whole sample has said nothing yet. What is
worth knowing is whether it survives on pieces of the sample it was not
selected on, and that requires the pieces not to overlap.

Purging
-------
Adjacent folds overlap through two channels, and both have to be cut.

A windowed signal at the first day of a fold was computed from a window
reaching back into the previous fold. A 250-day operator reaches back 250
days. Without a gap, the first year of every fold is partly a function of the
fold before it, and the folds are not independent evidence.

A forward return at the last day of a fold reaches into the next fold, by the
scoring horizon.

So consecutive folds are separated by ``warmup + horizon`` days that belong to
neither. That is a real cost, and on a short panel with a 250-day operator the
gaps can eat a meaningful fraction of the sample; the alternative is folds
that agree with each other because they share their data, which looks like
robustness and is not.

Equal-length folds in time, not equal numbers of observations. A fold covering
a quiet year and a fold covering a crisis are the comparison that matters, and
equalising observation counts would blur exactly that difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

DEFAULT_FOLDS: Final[int] = 5


@dataclass(frozen=True, slots=True)
class Fold:
    """One scoring segment, with the rows that may be used."""

    index: int
    start: int
    stop: int

    @property
    def n_days(self) -> int:
        return self.stop - self.start

    def __repr__(self) -> str:
        return f"Fold({self.index}: {self.start}..{self.stop})"


def purged_folds(
    n_days: int,
    *,
    warmup: int,
    horizon: int = 1,
    n_folds: int = DEFAULT_FOLDS,
    min_fold_days: int = 120,
) -> tuple[Fold, ...]:
    """Contiguous scoring segments separated by an embargo.

    The first fold starts after ``warmup``, because nothing before it is data.
    Consecutive folds are separated by ``warmup + horizon`` days, which is the
    exact reach of the overlap: a windowed signal looks back ``warmup`` days
    and a forward return looks ahead ``horizon``.

    Returns fewer folds than asked for rather than folds too short to say
    anything, and none at all if the panel cannot support even one. A fold of
    thirty days produces an IC whose standard error swamps any difference
    between folds, and comparing such folds is measuring noise against noise.
    """
    if n_days <= 0 or warmup < 0 or horizon < 1 or n_folds < 1:
        raise ValueError("a purged split needs a panel, a warmup and a horizon")

    embargo = warmup + horizon
    usable = n_days - warmup - horizon
    if usable <= 0:
        return ()

    for folds in range(n_folds, 0, -1):
        span = (usable - embargo * (folds - 1)) // folds
        if span >= min_fold_days:
            break
    else:
        return ()

    out: list[Fold] = []
    cursor = warmup
    for index in range(folds):
        stop = cursor + span
        out.append(Fold(index=index, start=cursor, stop=stop))
        cursor = stop + embargo
    return tuple(out)


def regime_split(
    volatility: "list[float] | tuple[float, ...]", quantile: float = 0.5
) -> tuple[list[int], list[int]]:
    """Day indices split into calm and turbulent halves.

    Regime is defined by realised cross-sectional volatility rather than by
    calendar, because a date range is a proxy for the thing that actually
    matters and a poor one: 2011 and 2015 were both quiet years with a violent
    quarter inside them.
    """
    import numpy as np

    values = np.asarray(volatility, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [], []
    cut = float(np.quantile(finite, quantile))
    calm = [i for i, v in enumerate(values) if np.isfinite(v) and v <= cut]
    rough = [i for i, v in enumerate(values) if np.isfinite(v) and v > cut]
    return calm, rough
