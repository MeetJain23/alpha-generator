# Parquet layout expected by `USAdapter`

> **Status: proposed contract, pending the real files.** Written before the
> data arrives so `USAdapter` has something concrete to be built against. If
> the actual files differ, this document changes and the adapter follows. The
> invariants near the end are not negotiable, because the system's freedom
> from look-ahead rests on them.

## Directory layout

```
<root>/
  bars/
    year=1996/part-0.parquet
    year=1997/part-0.parquet
    ...
  reference/
    instruments.parquet
    delistings.parquet
    sector_history.parquet
  costs/
    costs.parquet
  calendar/
    sessions.parquet
```

Bars are Hive-partitioned by `year` so a date-range load touches only the
relevant files. Partition granularity is a performance choice rather than a
semantic one, and the adapter discovers partitions instead of assuming them.

## `bars/`, one row per (date, instrument)

Long format, not wide. The adapter pivots to the (n_days x n_instruments)
planes that `Panel` holds. Storing it long keeps the files stable as the
universe changes size.

| column              | arrow type | null? | notes                                                   |
| ------------------- | ---------- | ----- | ------------------------------------------------------- |
| `date`              | `date32`   | no    | Trading session date, exchange local.                    |
| `instrument_id`     | `string`   | no    | Stable permanent id, **not** a ticker. See below.        |
| `open`              | `float32`  | yes   | Split and dividend adjusted.                             |
| `high`              | `float32`  | yes   |                                                          |
| `low`               | `float32`  | yes   |                                                          |
| `close`             | `float32`  | yes   |                                                          |
| `volume`            | `float32`  | yes   | Shares, adjusted consistently with price.                |
| `vwap`              | `float32`  | yes   | Session VWAP.                                            |
| `returns`           | `float32`  | yes   | One-day simple return, as-of this date.                  |
| `mcap`              | `float32`  | yes   | Point-in-time market cap.                                |
| `adj_factor`        | `float32`  | yes   | Cumulative adjustment applied to raw prices.             |
| `sector`            | `int32`    | yes   | Sector code in force on this date. Null if unclassified. |
| `lot_size`          | `int32`    | yes   | US: 1.                                                   |
| `can_short`         | `bool`     | yes   | Borrow availability that day.                            |
| `borrow_cost`       | `float32`  | yes   | Annualised borrow fee, decimal.                          |
| `band_upper`        | `float32`  | yes   | US: null, no price bands. Kept for other markets.        |
| `band_lower`        | `float32`  | yes   | US: null.                                                |
| `surveillance_flag` | `bool`     | yes   | US: false. Kept for other markets.                       |
| `halted`            | `bool`     | yes   | Session level trading halt.                              |
| `is_listed`         | `bool`     | no    | False before listing and on or after delisting.          |
| `index_membership`  | `int32`    | yes   | Bitmask. Bit 0 S&P 500, bit 1 R1000, bit 2 R2000.        |

### On `instrument_id`

A permanent identifier such as PERMNO, or a vendor equivalent. Never a
ticker. Tickers are recycled, so a ticker that means one company in 2003 can
mean an unrelated one in 2015, and keying the panel on tickers splices two
companies into a single time series. The panel's instrument axis is the id
axis. Tickers, if carried at all, are display metadata.

## `reference/instruments.parquet`

| column             | arrow type | null? | notes                   |
| ------------------ | ---------- | ----- | ----------------------- |
| `instrument_id`    | `string`   | no    | Primary key, unique.     |
| `ticker`           | `string`   | yes   | Display only.            |
| `name`             | `string`   | yes   | Display only.            |
| `first_date`       | `date32`   | no    | First tradable session.  |
| `last_date`        | `date32`   | no    | Last tradable session.   |
| `primary_exchange` | `string`   | yes   |                          |

## `reference/delistings.parquet`

Survivorship bias enters here if it enters anywhere.

| column          | arrow type | null? | notes                                                |
| --------------- | ---------- | ----- | ---------------------------------------------------- |
| `instrument_id` | `string`   | no    |                                                      |
| `delist_date`   | `date32`   | no    | First session on which the name no longer trades.    |
| `delist_return` | `float32`  | yes   | Terminal return to the holder. Often large, negative. |
| `delist_code`   | `int32`    | yes   | Vendor reason code, such as merger or bankruptcy.    |

The adapter applies `delist_return` on the instrument's final valid day,
composed into that day's `returns`, and writes NaN for every later date. A
delisted name is not carried forward at its last price.

If a bankrupt company's terminal loss is dropped rather than applied, every
value strategy in the system looks profitable and none of them are. The
adapter asserts this behaviour rather than trusting the file to be right.

## `reference/sector_history.parquet`

| column          | arrow type | null? | notes                                             |
| --------------- | ---------- | ----- | ------------------------------------------------- |
| `instrument_id` | `string`   | no    |                                                   |
| `effective_date`| `date32`   | no    | Date the classification took effect.               |
| `sector`        | `int32`    | no    | Code in force from `effective_date` until the next. |

This file is what makes the `sector` column in `bars/` point-in-time. If the
vendor supplies only a current classification per instrument, this file does
not exist and the `sector` column is backfilled current state. See the next
section.

## `costs/costs.parquet`

| column            | arrow type | null? | notes                                          |
| ----------------- | ---------- | ----- | ---------------------------------------------- |
| `instrument_id`   | `string`   | yes   | Null means the default applied to everything else. |
| `date`            | `date32`   | yes   | Null means time invariant.                      |
| `commission_bps`  | `float32`  | no    |                                                 |
| `half_spread_bps` | `float32`  | no    |                                                 |
| `impact_coef`     | `float32`  | yes   | Coefficient on participation rate.              |

## `calendar/sessions.parquet`

| column    | arrow type | null? | notes                        |
| --------- | ---------- | ----- | ---------------------------- |
| `date`    | `date32`   | no    | One row per trading session.  |
| `is_half` | `bool`     | yes   | Early close.                  |

The calendar is authoritative. The set of dates in `bars/` must be a subset of
it. A bar on a non-session date means the upstream pipeline is wrong, and the
adapter treats it as an error rather than quietly accepting the row.

## Invariants the adapter asserts at load

These are assertions, not validations. A violation raises and the run stops.
A silently tolerated one produces a backtest that is wrong in a direction that
flatters the strategy.

1. **Dates strictly increasing**, no duplicates.
2. **Instrument ids unique**, no duplicate (date, instrument) pairs.
3. **Delisted names are NaN after delisting**, never forward filled, with the
   delisting return applied on the final valid day.
4. **No resurrection.** Once an instrument is NaN following its delisting it
   has no later valid observation.
5. **`mcap` is as-of the panel date.** No restated fundamentals: the value on
   date *t* is what was knowable on *t*, not what was later revised to be true
   of *t*.
6. **Bar dates are a subset of calendar sessions.**
7. **`is_listed` is consistent** with `first_date` and `last_date` and with
   where the price planes are NaN.

## Known bounded leaks

Things the system cannot assert its way out of, recorded here so they are
visible rather than forgotten. Each is logged with the run.

### Sector classification may be current state applied backwards

**Unverified.** No vendor files have been supplied yet, so whether the sector
data is genuinely point-in-time is an open question about the source rather
than something this repository can determine. It is listed here instead of
under the invariants for that reason.

**What the adapter checks.** On load it computes the reclassification rate:
the fraction of instruments whose sector code changes at least once during
their life in the panel, and the count of change events per year. A real
classification history over a multi-decade US panel shows a low but clearly
non-zero rate, since a small percentage of names are reclassified each year.
A rate of exactly zero across decades is the signature of a single current
code broadcast across every historical date. The adapter logs the rate and
sets `sector_point_in_time` in the run config accordingly. It does not raise,
because a legitimately short panel can also show no changes.

**Why the leak is bounded.** `sector` is consumed by exactly two operators,
`demean_by` and `rank_within`. A signal that uses neither cannot be affected
at all. For signals that do use them, the leak is not knowledge of future
returns but knowledge of future peer membership: the demeaning is performed
against a peer set that was partly settled after the fact. The size of the
effect scales with the reclassification rate, which is small per year, and it
does not compound, since a name is either correctly or incorrectly grouped on
a given day.

**What to do about it.** If `sector_point_in_time` is false, results from
group-relative signals are provisional, and the flag is recorded in
`runs.config_json` so that a run cannot later be mistaken for a clean one.
Obtaining a classification history with effective dates is the fix, and it is
the only real one.

### Fundamentals beyond market cap are deferred

`mcap` is the only fundamental in the schema. Book value, earnings, accruals
and the rest are deferred rather than dropped: they need a point-in-time
database with both a report date and a knowledge date, and adding them
against a source that lacks the latter would introduce restatement look-ahead
that no assertion in this layer could detect. The grammar reserves no
placeholder for them, so adding them later is a schema and grammar change
made deliberately rather than by accident.

## Testing without real data

`alpha.data.synthetic` generates a panel with this full schema, including
staggered listings, mid-sample delistings with terminal returns, halts and
sector reclassifications, from an explicitly passed `numpy.random.Generator`.
Layers 1 and 2 are testable end to end before a single vendor file exists.
