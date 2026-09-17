"""Panel, CostModel and the Adapter protocol.

``Panel`` is the single in-memory representation every layer above consumes:
a ``DatetimeIndex`` of dates, an array of stable string instrument ids, and a
dict of (n_days x n_instruments) float32 planes.

The field schema is deliberately wider than any one market needs. Non-US
markets fill ``band_upper``, ``band_lower`` and ``surveillance_flag`` with
constants. The columns stay in the schema regardless, so an expression written
against one market is structurally valid against another and a strategy does
not have to be rewritten to be tested somewhere else.

Everything is float32, including the flags and the sector labels. Uniform
dtype keeps the evaluator free of per-field branching, and it lets a missing
value be NaN in every plane. An int32 sector plane would need a sentinel for
"unclassified", and a sentinel is a value that some operator will eventually
treat as data.

Panels are immutable. The arrays are marked read-only on construction, so a
consumer that tries to patch a panel in place fails loudly instead of
producing results that no longer correspond to the panel anyone else holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from alpha.logging_config import get_logger

_log = get_logger(__name__)

# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

PRICE_FIELDS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "vwap")
"""Planes that must be NaN wherever an instrument is not listed."""

VALUE_FIELDS: Final[tuple[str, ...]] = (
    *PRICE_FIELDS,
    "volume",
    "returns",
    "mcap",
    "adj_factor",
)

TRADING_FIELDS: Final[tuple[str, ...]] = (
    "lot_size",
    "can_short",
    "borrow_cost",
    "band_upper",
    "band_lower",
    "surveillance_flag",
    "halted",
    "is_listed",
    "index_membership",
)

GROUP_FIELDS: Final[tuple[str, ...]] = ("sector",)

FIELDS: Final[tuple[str, ...]] = (*VALUE_FIELDS, *TRADING_FIELDS, *GROUP_FIELDS)
"""Every plane a Panel carries. Required, in full, for every market."""

CONSTANT_IN_US: Final[frozenset[str]] = frozenset(
    {"band_upper", "band_lower", "surveillance_flag", "lot_size"}
)
"""Fields with no US content. Present so the schema travels; filled with
constants by ``USAdapter`` rather than omitted."""

DTYPE: Final[np.dtype] = np.dtype(np.float32)


class PanelError(ValueError):
    """A panel violates an invariant the rest of the system relies on."""


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Panel:
    """A rectangular market panel.

    ``dates`` indexes rows, ``instruments`` indexes columns, and every entry
    of ``fields`` is an (n_days x n_instruments) float32 array sharing that
    orientation. The orientation is fixed and never transposed: time is axis
    0 everywhere in the system, so a rolling operation is always a walk down
    a column and a cross-sectional operation is always a walk across a row.
    """

    dates: pd.DatetimeIndex
    instruments: np.ndarray
    fields: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        for array in self.fields.values():
            array.flags.writeable = False
        self.instruments.flags.writeable = False

    # -- shape -------------------------------------------------------------

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_instruments(self) -> int:
        return len(self.instruments)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_days, self.n_instruments)

    def __getitem__(self, field: str) -> np.ndarray:
        try:
            return self.fields[field]
        except KeyError:
            raise KeyError(
                f"no field {field!r}; panel carries {sorted(self.fields)}"
            ) from None

    def __contains__(self, field: str) -> bool:
        return field in self.fields

    # -- slicing -----------------------------------------------------------

    def head(self, k: int) -> Panel:
        """The first ``k`` dates, as a panel in its own right.

        This is what the look-ahead check is built on. Evaluating a tree on
        ``panel.head(k)`` must give exactly what evaluating on the full panel
        and slicing to ``[:k]`` gives, for every tree and every k. Any
        operator that reaches forward in time breaks that equality, which is
        why the operation lives here rather than being improvised per test.
        """
        if not 0 <= k <= self.n_days:
            raise IndexError(f"cannot take {k} of {self.n_days} days")
        return Panel(
            dates=self.dates[:k],
            instruments=self.instruments.copy(),
            fields={name: np.array(plane[:k]) for name, plane in self.fields.items()},
        )

    # -- integrity ---------------------------------------------------------

    def validate(self) -> None:
        """Assert every invariant the layers above assume.

        Raises ``PanelError``. These are assertions rather than warnings: a
        panel that violates one of them produces results that are wrong in a
        direction that flatters the strategy, and a warning in a log nobody
        reads is not a defence against that.
        """
        _validate_axes(self)
        _validate_planes(self)
        _validate_listing(self)


def _validate_axes(panel: Panel) -> None:
    if not isinstance(panel.dates, pd.DatetimeIndex):
        raise PanelError(f"dates must be a DatetimeIndex, got {type(panel.dates)}")
    if not panel.dates.is_monotonic_increasing or panel.dates.has_duplicates:
        raise PanelError("dates must be strictly increasing with no duplicates")
    if panel.instruments.ndim != 1:
        raise PanelError("instruments must be one-dimensional")
    if len(set(panel.instruments.tolist())) != panel.n_instruments:
        raise PanelError("instrument ids must be unique")


def _validate_planes(panel: Panel) -> None:
    missing = set(FIELDS) - set(panel.fields)
    if missing:
        raise PanelError(f"panel is missing fields: {sorted(missing)}")
    unexpected = set(panel.fields) - set(FIELDS)
    if unexpected:
        raise PanelError(
            f"panel carries unknown fields {sorted(unexpected)}; the schema is "
            f"fixed so that an expression valid on one panel is valid on another"
        )
    for name, plane in panel.fields.items():
        if plane.shape != panel.shape:
            raise PanelError(
                f"field {name!r} has shape {plane.shape}, expected {panel.shape}"
            )
        if plane.dtype != DTYPE:
            raise PanelError(f"field {name!r} has dtype {plane.dtype}, expected float32")
        if np.isinf(plane).any():
            raise PanelError(
                f"field {name!r} contains an infinity; a missing or undefined "
                f"value must be NaN, which propagates, rather than an infinity, "
                f"which survives every downstream operator"
            )


def _validate_listing(panel: Panel) -> None:
    """Listing windows are contiguous and prices respect them.

    The two failures this catches are the expensive ones. A delisted name that
    reappears means the id was recycled or the delisting was dropped. A price
    surviving past the delisting date means the name was forward filled at its
    last quote, which is how a bankrupt company keeps contributing returns to
    a backtest forever.
    """
    listed = panel["is_listed"]
    if np.isnan(listed).any():
        raise PanelError("is_listed must be 0 or 1 everywhere, never NaN")
    if not np.isin(listed, (0.0, 1.0)).all():
        raise PanelError("is_listed must be 0 or 1")

    boolean = listed.astype(bool)
    # A contiguous run of listed days has at most one 0 -> 1 transition.
    starts = np.count_nonzero(boolean[1:] & ~boolean[:-1], axis=0)
    first_day_listed = boolean[0].astype(int) if panel.n_days else np.zeros(0, int)
    if np.any(starts + first_day_listed > 1):
        offenders = panel.instruments[starts + first_day_listed > 1]
        raise PanelError(
            f"instruments relist after delisting: {offenders[:5].tolist()}. "
            f"Either an id was recycled or a delisting was dropped."
        )

    unlisted = ~boolean
    for name in PRICE_FIELDS:
        leaked = unlisted & ~np.isnan(panel[name])
        if leaked.any():
            columns = panel.instruments[leaked.any(axis=0)]
            raise PanelError(
                f"field {name!r} has values on {leaked.sum()} unlisted cells "
                f"({columns[:5].tolist()}). A delisted name must be NaN, not "
                f"carried forward at its last price."
            )


# --------------------------------------------------------------------------
# costs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostModel:
    """Round-trip trading cost in basis points.

    Three terms, because they behave differently with size. Commission is
    flat, the half spread is what crossing costs at any size, and impact grows
    with participation. Modelling only the first two makes every high-turnover
    signal look tradable at any capital, which is the single most common way
    a backtest overstates capacity.

    The square-root impact term is the conventional shape and a placeholder
    for a calibrated one. It is deliberately a stated assumption rather than a
    fitted number nobody can reproduce.
    """

    commission_bps: float = 1.0
    half_spread_bps: float = 2.5
    impact_coef: float = 10.0

    def trade_cost_bps(self, participation_rate: np.ndarray | float) -> np.ndarray:
        """Cost of trading at a given fraction of daily volume.

        ``participation_rate`` is traded shares over daily volume. NaN
        propagates, so an instrument with no volume has no cost estimate
        rather than a free trade.
        """
        rate = np.asarray(participation_rate, dtype=DTYPE)
        if np.any(rate < 0):
            raise ValueError("participation rate cannot be negative")
        impact = self.impact_coef * np.sqrt(rate)
        return np.asarray(
            self.commission_bps + self.half_spread_bps + impact, dtype=DTYPE
        )


# --------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------


@runtime_checkable
class Adapter(Protocol):
    """Source of panels, costs and a trading calendar for one market.

    Everything above Layer 1 is written against this and nothing else, so a
    second market is a second Adapter rather than a branch in the engine.
    """

    def panel(self, start: Any, end: Any) -> Panel:
        """The panel covering [start, end], already validated."""
        ...

    def costs(self) -> CostModel:
        """The cost model for this market."""
        ...

    def calendar(self) -> pd.DatetimeIndex:
        """Every trading session, whether or not the panel covers it."""
        ...
