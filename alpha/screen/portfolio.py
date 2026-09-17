"""Decile portfolios on a monthly rebalance.

Separate from ``metrics`` because these are portfolio returns rather than a
signal statistic: they depend on how the book is built, not only on how the
signal orders instruments. Two signals with identical rank IC can produce
different decile spreads if one concentrates its extremes differently.

Everything here obeys the same alignment rule as ``metrics``. Holdings are set
from the signal known on the rebalance date and earn from the following day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from alpha.data.types import Panel
from alpha.screen.metrics import TRADING_DAYS_PER_YEAR, decile_membership

DECILES: Final[int] = 10


@dataclass(frozen=True, slots=True)
class DecileResult:
    """Daily and monthly returns of a decile sort."""

    daily: pd.DataFrame
    """One column per decile, plus ``market``, indexed by date."""

    monthly_spread: pd.Series
    spread_annualised: float
    top_minus_market_annualised: float
    n_months: int
    n_rebalances: int


def decile_returns(
    signal: np.ndarray,
    warmup: int,
    panel: Panel,
    *,
    lag: bool = True,
    buckets: int = DECILES,
) -> DecileResult:
    """Monthly rebalanced equal-weight deciles on a cross-sectional signal.

    Holdings are set on the last session of each month from the signal known
    that day and held through the following month. Returns accrue from the day
    after formation when ``lag`` is set, which is the only honest choice: a
    portfolio cannot earn the return of the day whose close decided it.
    """
    returns = np.asarray(panel["returns"], dtype=np.float64)
    dates = panel.dates
    n_days = panel.n_days

    month = dates.to_period("M")
    is_rebalance = np.zeros(n_days, dtype=bool)
    is_rebalance[:-1] = month[:-1] != month[1:]
    is_rebalance[-1] = False
    is_rebalance[: max(warmup, 1)] = False

    columns = [f"D{i + 1}" for i in range(buckets)]
    daily = np.full((n_days, buckets + 1), np.nan)

    holdings: list[np.ndarray] = []
    n_rebalances = 0
    for t in range(n_days):
        if is_rebalance[t]:
            holdings = decile_membership(signal[t], buckets)
            n_rebalances += 1
        if not holdings:
            continue
        row = returns[t]
        for bucket, members in enumerate(holdings):
            if members.size:
                daily[t, bucket] = np.nanmean(row[members])
        live = np.flatnonzero(~np.isnan(row))
        if live.size:
            daily[t, buckets] = np.nanmean(row[live])

    frame = pd.DataFrame(daily, index=dates, columns=[*columns, "market"])
    if lag:
        # Formed on t, earned from t+1.
        frame = frame.shift(1)
    frame = frame.dropna(how="all")

    top = f"D{buckets}"
    spread = frame[top] - frame["D1"]
    excess = frame[top] - frame["market"]
    monthly = to_monthly(spread)
    return DecileResult(
        daily=frame,
        monthly_spread=monthly,
        spread_annualised=annualise(spread),
        top_minus_market_annualised=annualise(excess),
        n_months=int(monthly.size),
        n_rebalances=n_rebalances,
    )


def to_monthly(daily: pd.Series) -> pd.Series:
    """Compound a daily series within each calendar month."""
    clean = daily.dropna()
    if clean.empty:
        return clean
    return clean.groupby(clean.index.to_period("M")).apply(
        lambda block: float(np.prod(1.0 + block.to_numpy()) - 1.0)
    )


def annualise(daily: pd.Series) -> float:
    """Geometric annualised return of a daily series."""
    clean = daily.dropna().to_numpy()
    if clean.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + clean))
    if growth <= 0.0:
        return float("nan")
    return growth ** (TRADING_DAYS_PER_YEAR / clean.size) - 1.0
