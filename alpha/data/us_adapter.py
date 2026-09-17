"""USAdapter: materializes a Panel from local parquet.

The expected on-disk layout is documented in ``docs/PARQUET_LAYOUT.md``.

What this module is actually for is the load-time assertions. Reading parquet
and pivoting it is mechanical. Deciding what to do about a company that went
bankrupt in 2008 is not, and getting it wrong is undetectable downstream:
every strategy simply looks better than it was, and nothing in the output says
why.

Assertions
----------
  * a delisted instrument is NaN on every date after its delisting, with the
    delisting return applied on its final valid day, never forward filled;
  * dates strictly increasing, instrument ids unique;
  * bar dates are a subset of the trading calendar;
  * every value is as-of the panel date, checked against ``knowledge_date``
    where the source carries one.

Diagnostics
-----------
Point-in-time sector classification cannot be asserted, only measured. The
adapter computes the reclassification rate and reports it. See
``LoadDiagnostics`` and the known-leaks section of the layout document.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from alpha.data import cache
from alpha.data.types import (
    CONSTANT_IN_US,
    DTYPE,
    FIELDS,
    CostModel,
    Panel,
    PanelError,
)
from alpha.logging_config import get_logger

_log = get_logger(__name__)

BOOLEAN_FIELDS: Final[tuple[str, ...]] = (
    "can_short",
    "surveillance_flag",
    "halted",
)

TRADING_DAYS_PER_YEAR: Final[int] = 252


class LayoutError(FileNotFoundError):
    """The source directory does not match the documented layout."""


@dataclass(frozen=True, slots=True)
class LoadDiagnostics:
    """What the adapter measured but could not assert.

    Returned alongside the panel and recorded with the run, so that a result
    carries the provenance of the data it was computed from rather than
    relying on anyone remembering which files were in the directory that day.
    """

    n_days: int
    n_instruments: int
    n_delisted: int
    delist_returns_applied: int
    sector_change_rate_per_year: float
    sector_point_in_time: bool
    """False when no instrument was ever reclassified across a panel long
    enough that some should have been. Evidence of current state backfilled
    over history rather than proof of it."""

    halted_fraction: float
    listed_fraction: float


# --------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------


class USAdapter:
    """Reads the documented parquet layout and produces validated panels."""

    def __init__(
        self,
        root: str | Path,
        *,
        cache_dir: str | Path | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise LayoutError(f"no data root at {self.root}")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cost_model = cost_model
        self._diagnostics: LoadDiagnostics | None = None

    # -- protocol ----------------------------------------------------------

    def panel(self, start: Any, end: Any) -> Panel:
        """The validated panel covering [start, end].

        A cached materialization is used when one exists for the same range,
        because the parquet decode and pivot cost seconds and produce exactly
        the same bytes every time.
        """
        first, last = pd.Timestamp(start), pd.Timestamp(end)
        if first > last:
            raise ValueError(f"start {first.date()} is after end {last.date()}")

        cached = self._cache_path(first, last)
        if cached is not None and cached.exists():
            _log.info("panel served from cache", extra={"path": str(cached)})
            return cache.read(cached)

        built, diagnostics = self._build(first, last)
        self._diagnostics = diagnostics
        if cached is not None:
            cache.write(built, cached)
        return built

    def costs(self) -> CostModel:
        """The cost model, from ``costs/costs.parquet`` unless overridden."""
        if self._cost_model is not None:
            return self._cost_model
        path = self.root / "costs" / "costs.parquet"
        if not path.exists():
            raise LayoutError(f"no cost file at {path}")
        frame = pd.read_parquet(path)
        default = frame[frame["instrument_id"].isna()]
        if default.empty:
            raise LayoutError(
                "costs.parquet has no default row (instrument_id null); a "
                "per-instrument table with no fallback silently prices "
                "unknown names at zero"
            )
        row = default.iloc[0]
        return CostModel(
            commission_bps=float(row["commission_bps"]),
            half_spread_bps=float(row["half_spread_bps"]),
            impact_coef=float(row.get("impact_coef", 10.0)),
        )

    def calendar(self) -> pd.DatetimeIndex:
        """Every trading session, whether or not a panel covers it."""
        path = self.root / "calendar" / "sessions.parquet"
        if not path.exists():
            raise LayoutError(f"no calendar at {path}")
        sessions = pd.read_parquet(path)["date"]
        return pd.DatetimeIndex(sessions.sort_values().unique(), name="date")

    @property
    def diagnostics(self) -> LoadDiagnostics | None:
        """Measurements from the most recent build, or None if none was built."""
        return self._diagnostics

    # -- construction ------------------------------------------------------

    def _cache_path(self, start: pd.Timestamp, end: pd.Timestamp) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"panel_{start.date()}_{end.date()}.npz"

    def _build(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[Panel, LoadDiagnostics]:
        bars = self._read_bars(start, end)
        sessions = self.calendar()

        dates = pd.DatetimeIndex(np.sort(bars["date"].unique()), name="date")
        stray = dates.difference(sessions)
        if len(stray):
            raise PanelError(
                f"{len(stray)} bar dates are not trading sessions, first "
                f"{stray[0].date()}. The calendar is authoritative, so this "
                f"means the upstream pipeline emitted a bar that never traded."
            )

        instruments = np.array(
            sorted(bars["instrument_id"].astype(str).unique()), dtype=object
        )
        planes = _pivot(bars, dates, instruments)
        n_applied = self._apply_delistings(planes, dates, instruments)

        panel = Panel(dates=dates, instruments=instruments, fields=planes)
        panel.validate()

        diagnostics = self._measure(panel, n_applied)
        _log.info(
            "panel built",
            extra={
                "start": str(start.date()),
                "end": str(end.date()),
                "n_days": diagnostics.n_days,
                "n_instruments": diagnostics.n_instruments,
                "n_delisted": diagnostics.n_delisted,
                "sector_point_in_time": diagnostics.sector_point_in_time,
            },
        )
        return panel, diagnostics

    def _read_bars(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        path = self.root / "bars"
        if not path.exists():
            raise LayoutError(f"no bars directory at {path}")
        frame = pd.read_parquet(path)
        frame = frame.drop(columns=["year"], errors="ignore")
        frame["date"] = pd.to_datetime(frame["date"])

        _assert_as_of(frame)
        frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        if frame.empty:
            raise PanelError(f"no bars between {start.date()} and {end.date()}")

        duplicated = frame.duplicated(subset=["date", "instrument_id"])
        if duplicated.any():
            offender = frame[duplicated].iloc[0]
            raise PanelError(
                f"duplicate (date, instrument) rows, first "
                f"{offender['instrument_id']} on {offender['date'].date()}"
            )
        return frame

    def _apply_delistings(
        self,
        planes: dict[str, np.ndarray],
        dates: pd.DatetimeIndex,
        instruments: np.ndarray,
    ) -> int:
        """Compose the terminal return into the final valid day.

        The delisting return is the return to the holder from the last traded
        price to whatever the position finally settled at. It is composed with
        that day's own market return rather than replacing it, because both
        happened.

        Everything after the delisting date is already NaN, since the vendor
        writes no rows there. This asserts that rather than assuming it, since
        a source that pads its rows forward would otherwise sail straight
        through and leave every delisted name trading at its last price
        forever.
        """
        path = self.root / "reference" / "delistings.parquet"
        if not path.exists():
            _log.warning(
                "no delistings file; the panel may be a survivor panel",
                extra={"expected": str(path)},
            )
            return 0

        table = pd.read_parquet(path)
        table["delist_date"] = pd.to_datetime(table["delist_date"])
        positions = {str(name): j for j, name in enumerate(instruments)}
        returns = np.array(planes["returns"])
        listed = planes["is_listed"]
        applied = 0

        for row in table.itertuples(index=False):
            column = positions.get(str(row.instrument_id))
            if column is None or pd.isna(row.delist_return):
                continue
            live = np.flatnonzero(listed[:, column] == 1.0)
            if live.size == 0:
                continue
            final = int(live[-1])
            if dates[final] > row.delist_date:
                raise PanelError(
                    f"{row.instrument_id} has bars on {dates[final].date()}, "
                    f"after its delisting on {row.delist_date.date()}. A "
                    f"delisted name must not be carried forward."
                )
            if dates[final] < row.delist_date and final < len(dates) - 1:
                # Delisted outside the requested window: nothing to apply.
                continue

            own = returns[final, column]
            composed = (
                float(row.delist_return)
                if np.isnan(own)
                else (1.0 + own) * (1.0 + float(row.delist_return)) - 1.0
            )
            returns[final, column] = composed
            applied += 1

        planes["returns"] = np.ascontiguousarray(returns, dtype=DTYPE)
        return applied

    def _measure(self, panel: Panel, delist_returns_applied: int) -> LoadDiagnostics:
        listed = panel["is_listed"].astype(bool)
        sector = panel["sector"]
        years = max(panel.n_days / TRADING_DAYS_PER_YEAR, 1e-9)

        changed = np.zeros(panel.n_instruments, dtype=bool)
        for j in range(panel.n_instruments):
            column = sector[listed[:, j], j]
            column = column[~np.isnan(column)]
            if column.size:
                changed[j] = bool((column[1:] != column[:-1]).any())

        rate = float(changed.sum()) / max(panel.n_instruments, 1) / years
        # Below roughly a tenth of a percent a year across a multi-year panel,
        # the likeliest explanation is that history was never recorded.
        long_enough = years >= 3.0
        point_in_time = bool(rate > 0.001) or not long_enough
        if long_enough and not point_in_time:
            _log.warning(
                "no sector reclassifications across a multi-year panel; "
                "treating sector as current state backfilled over history",
                extra={"years": round(years, 2), "rate_per_year": rate},
            )

        delisted = int(
            np.count_nonzero(listed.any(axis=0) & ~listed[-1]) if panel.n_days else 0
        )
        return LoadDiagnostics(
            n_days=panel.n_days,
            n_instruments=panel.n_instruments,
            n_delisted=delisted,
            delist_returns_applied=delist_returns_applied,
            sector_change_rate_per_year=rate,
            sector_point_in_time=point_in_time,
            halted_fraction=float(np.nanmean(panel["halted"])),
            listed_fraction=float(listed.mean()),
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _assert_as_of(frame: pd.DataFrame) -> None:
    """No value may be knowable after the date it is attached to.

    Only checkable when the source carries a knowledge date. When it does not,
    the property is a claim about the vendor rather than about these files,
    and the layout document records it as such rather than pretending an
    assertion covers it.
    """
    if "knowledge_date" not in frame.columns:
        return
    known = pd.to_datetime(frame["knowledge_date"])
    restated = known > frame["date"]
    if restated.any():
        offender = frame[restated].iloc[0]
        raise PanelError(
            f"{int(restated.sum())} rows carry values known only after their "
            f"own date, first {offender['instrument_id']} on "
            f"{offender['date'].date()}. These are restated values, and using "
            f"them is look-ahead."
        )


def _pivot(
    frame: pd.DataFrame, dates: pd.DatetimeIndex, instruments: np.ndarray
) -> dict[str, np.ndarray]:
    """Long rows to dense planes, NaN wherever the source had no row.

    Factorized once and assigned by index rather than pivoted per field. A
    pivot per field would re-derive the same row and column positions
    nineteen times over, and on a full US panel that is the difference between
    seconds and minutes.
    """
    rows = dates.get_indexer(frame["date"])
    lookup = {str(name): j for j, name in enumerate(instruments)}
    columns = np.array([lookup[str(x)] for x in frame["instrument_id"]], dtype=np.int64)
    shape = (len(dates), len(instruments))

    planes: dict[str, np.ndarray] = {}
    for name in FIELDS:
        plane = np.full(shape, np.nan, dtype=DTYPE)
        if name == "is_listed":
            # A row in the source is the definition of being listed: the
            # vendor writes no row for a company that did not exist.
            plane[:] = 0.0
            plane[rows, columns] = 1.0
        elif name in frame.columns:
            values = frame[name]
            if name in BOOLEAN_FIELDS:
                values = values.fillna(False).astype(np.float32)
            plane[rows, columns] = np.asarray(values, dtype=DTYPE)
        elif name in CONSTANT_IN_US:
            # Present so the schema travels, with no US content to fill it.
            plane[rows, columns] = 1.0 if name == "lot_size" else np.nan
        else:
            raise PanelError(f"source has no column for required field {name!r}")
        planes[name] = plane

    planes["is_listed"] = np.ascontiguousarray(planes["is_listed"], dtype=DTYPE)
    return planes
