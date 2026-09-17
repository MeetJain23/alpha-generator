"""Verification milestone: 12-1 momentum measured against Ken French's UMD.

Builds 12-1 momentum as an expression tree, forms decile portfolios on a
monthly rebalance, and reports the D10 minus D1 annualised spread alongside
the D10 minus market return.

The tree
--------
    div(ts_delay(close, 20), ts_delay(close, 250))

The price twenty days ago over the price two hundred and fifty days ago. It
skips the most recent month, where short-term reversal dominates and would
cancel much of the effect, and it is a ratio rather than a difference, so it
has no price units and ranking it sorts by return rather than by price level.
Warmup is 250 rows.

Pass criterion
--------------
Monthly correlation with UMD above 0.9 over the identical sample.

Correlation, not a level match. French builds UMD from a 2x3 sort on size and
prior return over NYSE breakpoints, value weighted, on his own universe. A
decile spread on a different universe differs in level for reasons that have
nothing to do with whether the signal was computed correctly. What must agree
is the month-to-month shape, because that is what the underlying effect drives
rather than the portfolio construction.

Below 0.9, something in Layers 0 through 2 is wrong, and it is far cheaper to
find out here, against a published series, than in the generator where every
candidate is unfamiliar and nothing can be checked by eye.

The one-day lag
---------------
A portfolio formed from the signal on date t earns returns from t+1 onward.
Using the same day's return would be look-ahead of the most ordinary kind.

For this particular tree the lag happens to change almost nothing, which is
worth knowing rather than glossing over: 12-1 momentum is built from prices
that are already twenty days stale, so there is no same-day information in it
to leak. The lag stays because the next signal through this machinery will not
have that property, and a harness that is only correct for the signal it was
written against is not a harness. ``--no-lag`` measures the difference and is
not a mode anything should be verified in.

Usage
-----
    python scripts/verify_momentum.py --root path/to/parquet
    python scripts/verify_momentum.py --synthetic     # machinery only
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.data.synthetic import SyntheticSpec, generate
from alpha.data.types import Panel
from alpha.data.us_adapter import USAdapter
from alpha.expr.ast import Node, from_string
from alpha.expr.evaluator import evaluate
from alpha.logging_config import configure

MOMENTUM = "div(ts_delay(close, 20), ts_delay(close, 250))"
DECILES = 10
TRADING_DAYS_PER_YEAR = 252
PASS_CORRELATION = 0.9

FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)


@dataclass(frozen=True, slots=True)
class DecileResult:
    """Everything the milestone reports."""

    daily: pd.DataFrame
    """One column per decile, plus ``market``, indexed by date."""

    monthly_spread: pd.Series
    spread_annualised: float
    top_minus_market_annualised: float
    n_months: int
    n_rebalances: int


# --------------------------------------------------------------------------
# portfolio construction
# --------------------------------------------------------------------------


def decile_returns(
    signal: np.ndarray,
    warmup: int,
    panel: Panel,
    *,
    lag: bool = True,
) -> DecileResult:
    """Monthly rebalanced equal-weight deciles on a cross-sectional signal.

    Holdings are set on the last session of each month from the signal known
    that day, and held through the following month. Returns accrue from the
    day after formation when ``lag`` is set, which is the only honest choice:
    a portfolio cannot earn the return of the day whose close decided it.
    """
    returns = np.asarray(panel["returns"], dtype=np.float64)
    dates = panel.dates
    n_days = panel.n_days

    month = dates.to_period("M")
    is_rebalance = np.zeros(n_days, dtype=bool)
    is_rebalance[:-1] = month[:-1] != month[1:]
    is_rebalance[-1] = False
    is_rebalance[: max(warmup, 1)] = False

    columns = [f"D{i + 1}" for i in range(DECILES)]
    daily = np.full((n_days, DECILES + 1), np.nan)

    holdings: list[np.ndarray] = []
    n_rebalances = 0
    for t in range(n_days):
        if is_rebalance[t]:
            holdings = _form_deciles(signal[t])
            n_rebalances += 1
        if not holdings:
            continue
        row = returns[t]
        for bucket, members in enumerate(holdings):
            if members.size:
                daily[t, bucket] = np.nanmean(row[members])
        live = np.flatnonzero(~np.isnan(row))
        if live.size:
            daily[t, DECILES] = np.nanmean(row[live])

    frame = pd.DataFrame(daily, index=dates, columns=[*columns, "market"])
    if lag:
        # Formed on t, earned from t+1. Without this the portfolio collects
        # the return of the day whose close chose it.
        frame = frame.shift(1)
    frame = frame.dropna(how="all")

    spread = frame["D10"] - frame["D1"]
    top = frame["D10"] - frame["market"]
    monthly = _to_monthly(spread)
    return DecileResult(
        daily=frame,
        monthly_spread=monthly,
        spread_annualised=_annualise(spread),
        top_minus_market_annualised=_annualise(top),
        n_months=int(monthly.size),
        n_rebalances=n_rebalances,
    )


def _form_deciles(row: np.ndarray) -> list[np.ndarray]:
    """Split one day's instruments into ten equal-count buckets by signal."""
    live = np.flatnonzero(~np.isnan(row))
    if live.size < DECILES:
        return []
    order = live[np.argsort(row[live], kind="stable")]
    return [np.asarray(part) for part in np.array_split(order, DECILES)]


def _to_monthly(daily: pd.Series) -> pd.Series:
    """Compound a daily series within each calendar month."""
    clean = daily.dropna()
    if clean.empty:
        return clean
    return clean.groupby(clean.index.to_period("M")).apply(
        lambda block: float(np.prod(1.0 + block.to_numpy()) - 1.0)
    )


def _annualise(daily: pd.Series) -> float:
    """Geometric annualised return of a daily series."""
    clean = daily.dropna().to_numpy()
    if clean.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + clean))
    if growth <= 0.0:
        return float("nan")
    return growth ** (TRADING_DAYS_PER_YEAR / clean.size) - 1.0


# --------------------------------------------------------------------------
# reference data
# --------------------------------------------------------------------------


def load_umd(cache_dir: Path, *, allow_download: bool) -> tuple[pd.Series, str]:
    """Monthly UMD from the Kenneth R. French data library, with its hash.

    Fetched once and cached. A run must not depend on network availability,
    and two runs must compare against identical bytes, so the cache is keyed
    by content hash and the hash is returned for the run record.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "F-F_Momentum_Factor_CSV.zip"

    if not cached.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"no cached factor file at {cached}. Re-run with --download to "
                f"fetch it from the French data library."
            )
        with urllib.request.urlopen(FRENCH_URL, timeout=60) as response:
            cached.write_bytes(response.read())

    payload = cached.read_bytes()
    digest = sha256(payload).hexdigest()[:32]

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = archive.namelist()[0]
        text = archive.read(name).decode("latin-1")
    return _parse_french(text), digest


def _parse_french(text: str) -> pd.Series:
    """Monthly UMD in decimal, indexed by period.

    The file carries a preamble and a trailing annual section, both of which
    have to be cut before parsing rather than coerced through, or the annual
    rows would silently join the monthly series.
    """
    values: dict[pd.Period, float] = {}
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or len(parts[0]) != 6 or not parts[0].isdigit():
            continue
        try:
            month = pd.Period(f"{parts[0][:4]}-{parts[0][4:]}", freq="M")
            values[month] = float(parts[1]) / 100.0
        except (ValueError, TypeError):
            continue
    if not values:
        raise ValueError("no monthly rows found in the French factor file")
    return pd.Series(values).sort_index()


def compare_to_umd(monthly: pd.Series, umd: pd.Series) -> tuple[float, int]:
    """Correlation over the overlapping months, and how many there were."""
    joined = pd.concat(
        [monthly.rename("ours"), umd.rename("umd")], axis=1, join="inner"
    ).dropna()
    if len(joined) < 24:
        raise ValueError(
            f"only {len(joined)} overlapping months; too few to say anything"
        )
    return float(joined["ours"].corr(joined["umd"])), int(len(joined))


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_panel(args: argparse.Namespace) -> Panel:
    if args.synthetic:
        spec = SyntheticSpec(
            n_days=args.days,
            n_instruments=args.instruments,
            momentum_strength=args.planted,
        )
        return generate(np.random.default_rng(args.seed), spec).panel
    if not args.root:
        raise SystemExit("pass --root with the parquet layout, or --synthetic")
    return USAdapter(args.root).panel(args.start, args.end)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2100-01-01")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--days", type=int, default=2500)
    parser.add_argument("--instruments", type=int, default=500)
    parser.add_argument("--planted", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--cache", default="cache/reference")
    parser.add_argument("--download", action="store_true",
                        help="allow fetching the French factor file if not cached")
    parser.add_argument("--no-lag", action="store_true",
                        help="earn the formation day's return, which is look-ahead")
    args = parser.parse_args()

    configure()
    panel = build_panel(args)
    tree: Node = from_string(MOMENTUM)
    result = evaluate(tree, panel)

    print(f"tree            {tree}")
    print(f"panel           {panel.n_days} x {panel.n_instruments}")
    print(f"warmup          {result.warmup} rows")
    print(f"degradation     {result.max_degradation:.4f} "
          f"(worst node: {result.worst_node()[1]})")

    deciles = decile_returns(result.values, result.warmup, panel, lag=not args.no_lag)
    print(f"rebalances      {deciles.n_rebalances}")
    print(f"months          {deciles.n_months}")
    print()
    print(f"D10 - D1        {deciles.spread_annualised:+.2%} annualised")
    print(f"D10 - market    {deciles.top_minus_market_annualised:+.2%} annualised")

    if args.synthetic:
        print()
        print("synthetic data: the machinery ran, and that is all this shows.")
        print("Finding a planted effect tests the pipeline. Only real data")
        print("tests the claim, so no pass or fail is reported here.")
        return 0

    umd, digest = load_umd(Path(args.cache), allow_download=args.download)
    correlation, overlap = compare_to_umd(deciles.monthly_spread, umd)
    print()
    print(f"UMD file        sha256:{digest}")
    print(f"overlap         {overlap} months")
    print(f"correlation     {correlation:.4f}  (pass above {PASS_CORRELATION})")
    print()
    if correlation > PASS_CORRELATION:
        print("PASS: the pipeline reproduces a published effect.")
        return 0
    print("FAIL: something in Layers 0 through 2 is wrong.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
