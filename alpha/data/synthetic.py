"""Synthetic panel generator.

Produces a Panel with the full field schema, including staggered listings,
mid-sample delistings with terminal returns, halts and sector
reclassifications, so that Layers 1 and 2 are testable end to end without
vendor data.

The point is not realism. It is that the awkward cases are present in every
test run rather than only in production. A generator that emits a clean
rectangle of returns tests nothing that matters: the bugs live in the ragged
edges, in the instrument that lists on day 400 and dies on day 900, in the
name halted for a week in the middle of a 250-day window.

By default the returns carry no signal at all. ``momentum_strength`` plants
one, and it exists to test that the decile machinery can find an effect that
is known to be there. It does not make the synthetic panel a substitute for
real data in ``verify_momentum.py``: finding a planted effect tests the
pipeline, and only real data tests the claim.

Randomness is an explicitly passed ``numpy.random.Generator``. There is no
module-level default and no implicit seeding, so a test that passes cannot
quietly be a test that got lucky on whatever the global state happened to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import numpy as np
import pandas as pd

from alpha.data.types import DTYPE, FIELDS, Panel
from alpha.logging_config import get_logger

_log = get_logger(__name__)

TRADING_DAYS_PER_YEAR: Final[int] = 252


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """Shape and pathology of a generated panel.

    Defaults are small enough to keep a test suite fast and ragged enough to
    exercise the cases that break things.
    """

    n_days: int = 1200
    n_instruments: int = 300
    start: str = "2015-01-02"

    annual_vol: float = 0.35
    annual_drift: float = 0.06

    late_listing_fraction: float = 0.20
    """Instruments that start after the panel does."""

    delist_fraction: float = 0.15
    """Instruments that stop before the panel does."""

    delist_return_mean: float = -0.35
    """Terminal return to the holder. Negative on average, and occasionally
    strongly positive, because delisting covers both bankruptcy and
    acquisition."""

    delist_return_sd: float = 0.40

    halt_probability: float = 0.002
    """Per instrument per day. A halted day has no trade, so its prices and
    volume are NaN, which is what exercises the min_periods path."""

    reclassification_rate: float = 0.02
    """Annual probability that an instrument changes sector."""

    n_sectors: int = 11

    momentum_strength: float = 0.0
    """Planted 12-1 momentum, in daily standard deviations per sigma of
    trailing return. Zero by default: synthetic data should not accidentally
    validate a hypothesis."""

    def __post_init__(self) -> None:
        if self.n_days < 2 or self.n_instruments < 2:
            raise ValueError("a panel needs at least two days and two instruments")
        if not 0.0 <= self.late_listing_fraction + self.delist_fraction <= 1.0:
            raise ValueError("listing and delisting fractions must leave survivors")


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """A generated panel plus the reference facts used to build it.

    The reference data is returned rather than discarded so that tests can
    check the adapter against what the generator actually did, instead of
    against a second guess at it.
    """

    panel: Panel
    market_returns: np.ndarray
    """Returns before the terminal return was composed in: what a vendor
    writes in its bars file, with the delisting kept separately."""

    delistings: Mapping[str, tuple[pd.Timestamp, float]]
    first_dates: Mapping[str, pd.Timestamp]
    last_dates: Mapping[str, pd.Timestamp]
    sector_changes: int


def generate(rng: np.random.Generator, spec: SyntheticSpec | None = None) -> SyntheticDataset:
    """Build a synthetic dataset from an explicit generator."""
    spec = spec or SyntheticSpec()
    dates = pd.bdate_range(start=spec.start, periods=spec.n_days, name="date")
    instruments = np.array(
        [f"SYN{i:05d}" for i in range(spec.n_instruments)], dtype=object
    )
    shape = (spec.n_days, spec.n_instruments)

    listed_from, listed_to, delist_return = _listing_windows(rng, spec)
    listed = _listing_mask(shape, listed_from, listed_to)
    halted = _halts(rng, spec, listed)
    tradable = listed & ~halted

    returns = _returns(rng, spec, tradable)
    market_returns = returns.copy()
    _apply_delisting_returns(returns, listed_to, delist_return, spec)

    close = _prices_from_returns(rng, returns, tradable)
    planes = _derive_planes(rng, spec, close, returns, tradable, listed, halted)
    sector, changes = _sectors(rng, spec, listed)
    planes["sector"] = sector

    panel = Panel(dates=dates, instruments=instruments, fields=planes)
    panel.validate()

    _log.info(
        "synthetic panel generated",
        extra={
            "n_days": spec.n_days,
            "n_instruments": spec.n_instruments,
            "delisted": int(np.count_nonzero(listed_to < spec.n_days - 1)),
            "sector_changes": changes,
        },
    )

    return SyntheticDataset(
        panel=panel,
        market_returns=np.ascontiguousarray(market_returns, dtype=DTYPE),
        delistings={
            str(instruments[j]): (dates[listed_to[j]], float(delist_return[j]))
            for j in range(spec.n_instruments)
            if listed_to[j] < spec.n_days - 1
        },
        first_dates={str(instruments[j]): dates[listed_from[j]] for j in range(spec.n_instruments)},
        last_dates={str(instruments[j]): dates[listed_to[j]] for j in range(spec.n_instruments)},
        sector_changes=changes,
    )


# --------------------------------------------------------------------------
# listing lifecycle
# --------------------------------------------------------------------------


def _listing_windows(
    rng: np.random.Generator, spec: SyntheticSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First and last listed row per instrument, plus terminal returns.

    Windows are drawn so that every instrument has a usable stretch of life.
    An instrument listed for six days is not an interesting test case, it is
    just a column of NaN that hides whatever the test was trying to measure.
    """
    n = spec.n_instruments
    minimum_life = max(2, spec.n_days // 10)

    listed_from = np.zeros(n, dtype=np.int64)
    late = rng.random(n) < spec.late_listing_fraction
    latest_start = max(1, spec.n_days - minimum_life)
    listed_from[late] = rng.integers(1, latest_start, size=int(late.sum()))

    listed_to = np.full(n, spec.n_days - 1, dtype=np.int64)
    dies = rng.random(n) < spec.delist_fraction
    for j in np.flatnonzero(dies):
        earliest_end = listed_from[j] + minimum_life
        if earliest_end >= spec.n_days - 1:
            continue
        listed_to[j] = rng.integers(earliest_end, spec.n_days - 1)

    delist_return = np.where(
        listed_to < spec.n_days - 1,
        rng.normal(spec.delist_return_mean, spec.delist_return_sd, size=n),
        np.nan,
    )
    # A total loss is the floor. Nothing returns less than -100%.
    delist_return = np.maximum(delist_return, -1.0)
    return listed_from, listed_to, delist_return


def _listing_mask(
    shape: tuple[int, int], listed_from: np.ndarray, listed_to: np.ndarray
) -> np.ndarray:
    rows = np.arange(shape[0])[:, None]
    return (rows >= listed_from[None, :]) & (rows <= listed_to[None, :])


def _halts(
    rng: np.random.Generator, spec: SyntheticSpec, listed: np.ndarray
) -> np.ndarray:
    """Trading halts, only on days the instrument is listed."""
    return (rng.random(listed.shape) < spec.halt_probability) & listed


# --------------------------------------------------------------------------
# returns and prices
# --------------------------------------------------------------------------


def _returns(
    rng: np.random.Generator, spec: SyntheticSpec, tradable: np.ndarray
) -> np.ndarray:
    """Daily returns, NaN wherever the instrument did not trade."""
    daily_vol = spec.annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    daily_drift = spec.annual_drift / TRADING_DAYS_PER_YEAR
    values = rng.normal(daily_drift, daily_vol, size=tradable.shape)

    if spec.momentum_strength:
        values = _plant_momentum(values, spec.momentum_strength, daily_vol)

    values[~tradable] = np.nan
    return values.astype(np.float64)


def _plant_momentum(
    values: np.ndarray, strength: float, daily_vol: float
) -> np.ndarray:
    """Add a component of next-period return explained by trailing 12-1 return.

    ``strength`` is measured in daily standard deviations, not in return. A
    one-sigma trailing winner gets ``strength * daily_vol`` added to its daily
    return. Scaling by the series' own volatility is what keeps the parameter
    meaningful: as an absolute daily return, a strength of 0.06 would be six
    per cent a day compounding, which drives prices past the float32 range
    within a few years and produces a panel the validator rightly refuses.

    Strictly backward looking: the return on day t is nudged by a window that
    ends on day t-21. A planted effect that peeked forward would make every
    look-ahead test pass for the wrong reason, which is worse than having no
    planted effect at all.
    """
    n_days = values.shape[0]
    planted = values.copy()
    for t in range(250, n_days):
        trailing = values[t - 250 : t - 20].sum(axis=0)
        centred = trailing - np.nanmean(trailing)
        scale = np.nanstd(centred)
        if scale > 0:
            planted[t] += strength * daily_vol * centred / scale
    return planted


def _apply_delisting_returns(
    returns: np.ndarray,
    listed_to: np.ndarray,
    delist_return: np.ndarray,
    spec: SyntheticSpec,
) -> None:
    """Compose the terminal return into the final valid day, in place.

    This is the single most important thing the generator does. If the
    terminal return is dropped, a panel of survivors is produced, every value
    strategy tested on it looks profitable, and nothing downstream can detect
    the problem because the evidence was never written down.

    Composed with the day own market return rather than replacing it, because
    both happened: the stock traded that day and then the position settled.
    """
    for j in np.flatnonzero(listed_to < spec.n_days - 1):
        own = returns[listed_to[j], j]
        terminal = delist_return[j]
        returns[listed_to[j], j] = (
            terminal if np.isnan(own) else (1.0 + own) * (1.0 + terminal) - 1.0
        )


def _prices_from_returns(
    rng: np.random.Generator, returns: np.ndarray, tradable: np.ndarray
) -> np.ndarray:
    """Compound returns into a price path, starting from a random level.

    Starting levels vary across instruments so that a signal keyed on price
    level rather than on return is visibly wrong instead of accidentally
    working.
    """
    n_days, n_instruments = returns.shape
    start_price = rng.uniform(5.0, 500.0, size=n_instruments)
    close = np.full(returns.shape, np.nan)

    filled = np.where(np.isnan(returns), 0.0, returns)
    for j in range(n_instruments):
        live = np.flatnonzero(tradable[:, j])
        if live.size == 0:
            continue
        path = start_price[j] * np.cumprod(1.0 + filled[live[0] : live[-1] + 1, j])
        segment = np.full(live[-1] + 1 - live[0], np.nan)
        segment[live - live[0]] = path[live - live[0]]
        close[live[0] : live[-1] + 1, j] = segment

    # A delisting return of -100% would put the price at zero, and a zero price
    # is not a price. The name is gone; the return is what the holder got.
    close[close <= 0.0] = np.nan
    return close


# --------------------------------------------------------------------------
# remaining planes
# --------------------------------------------------------------------------


def _derive_planes(
    rng: np.random.Generator,
    spec: SyntheticSpec,
    close: np.ndarray,
    returns: np.ndarray,
    tradable: np.ndarray,
    listed: np.ndarray,
    halted: np.ndarray,
) -> dict[str, np.ndarray]:
    """Everything the schema requires that is not price, return or sector."""
    shape = close.shape
    nan_where_untraded = np.where(tradable, 1.0, np.nan)

    intraday = np.abs(rng.normal(0.0, 0.01, size=shape))
    high = close * (1.0 + intraday)
    low = close * (1.0 - intraday)
    open_ = low + (high - low) * rng.random(shape)
    vwap = low + (high - low) * rng.random(shape)

    volume = rng.lognormal(mean=12.0, sigma=1.0, size=shape) * nan_where_untraded
    shares = rng.lognormal(mean=17.0, sigma=0.8, size=spec.n_instruments)
    mcap = close * shares[None, :]

    planes: dict[str, np.ndarray] = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "vwap": vwap,
        "volume": volume,
        "returns": returns,
        "mcap": mcap,
        # No splits in the synthetic panel: the prices are already adjusted,
        # and an adjustment factor that varies would be describing an
        # adjustment that was never applied.
        "adj_factor": np.where(listed, 1.0, np.nan),
        "lot_size": np.where(listed, 1.0, np.nan),
        "can_short": np.where(listed, 1.0, np.nan),
        "borrow_cost": np.where(listed, 0.003, np.nan),
        # Constant in the US, present so the schema travels.
        "band_upper": np.full(shape, np.nan),
        "band_lower": np.full(shape, np.nan),
        "surveillance_flag": np.where(listed, 0.0, np.nan),
        "halted": np.where(listed, halted.astype(np.float64), np.nan),
        "is_listed": listed.astype(np.float64),
        "index_membership": np.where(
            listed, _index_membership(rng, mcap, listed), np.nan
        ),
    }

    for name in ("open", "high", "low", "close", "vwap"):
        planes[name] = np.where(listed, planes[name], np.nan)

    return {name: np.ascontiguousarray(planes[name], dtype=DTYPE) for name in FIELDS if name != "sector"}


def _index_membership(
    rng: np.random.Generator, mcap: np.ndarray, listed: np.ndarray
) -> np.ndarray:
    """Bit 0 for the large-cap index, assigned by rank on market cap.

    Membership follows size rather than being drawn at random, because a
    universe filter that is independent of size would let a test pass that
    would fail on any real index.
    """
    membership = np.zeros(mcap.shape)
    with np.errstate(invalid="ignore"):
        for t in range(mcap.shape[0]):
            row = mcap[t]
            live = np.flatnonzero(listed[t] & ~np.isnan(row))
            if live.size == 0:
                continue
            cutoff = max(1, int(0.5 * live.size))
            largest = live[np.argsort(row[live])[-cutoff:]]
            membership[t, largest] = 1.0
    return membership


def _sectors(
    rng: np.random.Generator, spec: SyntheticSpec, listed: np.ndarray
) -> tuple[np.ndarray, int]:
    """Sector labels with occasional reclassification.

    Reclassifications are generated on purpose so that the adapter's
    point-in-time check has something to detect. A synthetic panel with a
    constant sector per instrument would look exactly like a vendor file that
    backfilled current state, and the check would be untested against the
    case it exists for.
    """
    n_days, n_instruments = listed.shape
    daily_rate = spec.reclassification_rate / TRADING_DAYS_PER_YEAR
    labels = rng.integers(0, spec.n_sectors, size=n_instruments)

    sector = np.full(listed.shape, np.nan)
    changes = 0
    current = labels.copy()
    for t in range(n_days):
        moving = np.flatnonzero(rng.random(n_instruments) < daily_rate)
        for j in moving:
            current[j] = (current[j] + 1 + rng.integers(0, spec.n_sectors - 1)) % spec.n_sectors
            changes += 1
        sector[t] = current
    sector[~listed] = np.nan
    return np.ascontiguousarray(sector, dtype=DTYPE), changes


# --------------------------------------------------------------------------
# on-disk layout
# --------------------------------------------------------------------------


def write_layout(dataset: SyntheticDataset, root: str | Path) -> Path:
    """Write a dataset to the parquet layout ``USAdapter`` expects.

    Documented in ``docs/PARQUET_LAYOUT.md``. This exists so the adapter is
    tested against files rather than against a mock of files: a mock agrees
    with whatever the adapter believes, which is precisely the belief under
    test.

    Delisted instruments are written the way a careful vendor would write
    them, with rows ending on the final valid day and the terminal return kept
    separately in ``delistings.parquet``. The adapter has to compose the two
    itself, because that is the step where the bias gets lost.
    """
    base = Path(root)
    panel = dataset.panel
    (base / "bars").mkdir(parents=True, exist_ok=True)
    (base / "reference").mkdir(parents=True, exist_ok=True)
    (base / "costs").mkdir(parents=True, exist_ok=True)
    (base / "calendar").mkdir(parents=True, exist_ok=True)

    frame = _long_frame(panel, dataset.market_returns)
    for year, block in frame.groupby(frame["date"].dt.year, sort=True):
        target = base / "bars" / f"year={year}"
        target.mkdir(parents=True, exist_ok=True)
        block.drop(columns=["year"], errors="ignore").to_parquet(
            target / "part-0.parquet", index=False
        )

    pd.DataFrame(
        {
            "instrument_id": [str(x) for x in panel.instruments],
            "ticker": [str(x) for x in panel.instruments],
            "name": [f"Synthetic {x}" for x in panel.instruments],
            "first_date": [dataset.first_dates[str(x)] for x in panel.instruments],
            "last_date": [dataset.last_dates[str(x)] for x in panel.instruments],
            "primary_exchange": ["SYN"] * panel.n_instruments,
        }
    ).to_parquet(base / "reference" / "instruments.parquet", index=False)

    delisted = sorted(dataset.delistings)
    pd.DataFrame(
        {
            "instrument_id": delisted,
            "delist_date": [dataset.delistings[k][0] for k in delisted],
            "delist_return": np.array(
                [dataset.delistings[k][1] for k in delisted], dtype=np.float32
            ),
            "delist_code": np.zeros(len(delisted), dtype=np.int32),
        }
    ).to_parquet(base / "reference" / "delistings.parquet", index=False)

    pd.DataFrame(
        {
            "instrument_id": [None],
            "date": [None],
            "commission_bps": np.array([1.0], dtype=np.float32),
            "half_spread_bps": np.array([2.5], dtype=np.float32),
            "impact_coef": np.array([10.0], dtype=np.float32),
        }
    ).to_parquet(base / "costs" / "costs.parquet", index=False)

    pd.DataFrame(
        {"date": panel.dates, "is_half": np.zeros(panel.n_days, dtype=bool)}
    ).to_parquet(base / "calendar" / "sessions.parquet", index=False)

    _log.info("synthetic layout written", extra={"root": str(base)})
    return base


def _long_frame(panel: Panel, market_returns: np.ndarray | None = None) -> pd.DataFrame:
    """Pivot the panel back to one row per (date, instrument), dropping the
    rows where the instrument was not listed.

    A vendor file does not carry a row for a company that did not exist, and
    an adapter that only ever sees dense rectangles will not handle one that
    does not.
    """
    n_days, n_instruments = panel.shape
    dates = np.repeat(panel.dates.values, n_instruments)
    ids = np.tile(np.array([str(x) for x in panel.instruments], dtype=object), n_days)

    columns: dict[str, np.ndarray] = {"date": dates, "instrument_id": ids}
    for name, plane in panel.fields.items():
        columns[name] = plane.reshape(-1)
    if market_returns is not None:
        columns["returns"] = market_returns.reshape(-1)

    frame = pd.DataFrame(columns)
    frame = frame[frame["is_listed"] == 1.0].copy()

    for name in ("can_short", "surveillance_flag", "halted", "is_listed"):
        frame[name] = frame[name].fillna(0).astype(bool)
    for name in ("lot_size", "index_membership"):
        frame[name] = frame[name].fillna(0).astype(np.int32)
    frame["sector"] = frame["sector"].astype("Int32")
    frame = frame.drop(columns=["is_listed"])
    return frame.sort_values(["date", "instrument_id"], ignore_index=True)
