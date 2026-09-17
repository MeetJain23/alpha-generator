"""OPEN MILESTONE: 12-1 momentum against Ken French's UMD, on real data.

Status: unsatisfied. No vendor files have been supplied, so this has never
been run against anything it can validate. It is the milestone, not a passing
test, and nothing in the repository should be read as having cleared it.

What this exists to test
------------------------
Not the decile sort. That is checked on synthetic data by
``check_decile_machinery.py``, which plants a known effect and finds it.

This exists to test the data pipeline, and it is the only thing that can.
Three properties have no synthetic equivalent, because the generator
satisfies each of them by construction rather than by being right:

  delisting composition   The generator applies terminal returns because it
                          was written to. A vendor file either carries them
                          or silently does not, and a survivor panel makes
                          every strategy look profitable with no trace in the
                          output. If the composition is wrong, momentum's
                          correlation with UMD degrades, because momentum's
                          losers are exactly the names that delist.

  adjustment factors      Synthetic prices are already adjusted because they
                          were never unadjusted. A real split applied on the
                          wrong date puts a 50 per cent return in a price
                          series, and a 12-1 ratio reads that as the strongest
                          momentum in the universe.

  point-in-time alignment A restated market cap or a backfilled sector cannot
                          exist in generated data. They can exist in a vendor
                          file, and they move the universe and the neutrality
                          without moving anything a coverage check would see.

A correlation above 0.9 with a published series says those three are right,
which is a claim no amount of internal testing can make.

Pass criterion
--------------
Monthly correlation with UMD above 0.9 over the identical sample.

Correlation, not a level match. French builds UMD from a 2x3 sort on size and
prior return over NYSE breakpoints, value weighted, on his own universe. A
decile spread on a different universe differs in level for reasons unrelated
to whether the signal was computed correctly. The month-to-month shape is what
the underlying effect drives, so that is what must agree.

Below 0.9, something in Layers 0 through 2 is wrong. Finding that out here,
against a published series, is far cheaper than finding it in the generator,
where every candidate is unfamiliar and nothing can be checked by eye.

Usage
-----
    python scripts/verify_momentum.py --root path/to/parquet --download
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from hashlib import sha256
from pathlib import Path

import pandas as pd

from alpha.data.us_adapter import USAdapter
from alpha.expr.ast import from_string
from alpha.expr.evaluator import evaluate
from alpha.logging_config import configure
from alpha.screen.portfolio import decile_returns

MOMENTUM = "div(ts_delay(close, 20), ts_delay(close, 250))"
PASS_CORRELATION = 0.9
MIN_OVERLAP_MONTHS = 24

FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)


def load_umd(cache_dir: Path, *, allow_download: bool) -> tuple[pd.Series, str]:
    """Monthly UMD from the Kenneth R. French data library, with its hash.

    Fetched once and cached. A run must not depend on network availability and
    two runs must compare against identical bytes, so the cache is keyed by
    content hash and the hash is returned for the run record. Downloading is
    opt-in rather than something the script does on its own.
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
        text = archive.read(archive.namelist()[0]).decode("latin-1")
    return parse_french(text), digest


def parse_french(text: str) -> pd.Series:
    """Monthly UMD in decimal, indexed by period.

    The file carries a preamble and a trailing annual section, both of which
    are cut rather than coerced through. An annual row parses perfectly well
    as a number, and joining it to a monthly series would corrupt the
    comparison silently.
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
    if len(joined) < MIN_OVERLAP_MONTHS:
        raise ValueError(
            f"only {len(joined)} overlapping months; too few to say anything"
        )
    return float(joined["ours"].corr(joined["umd"])), int(len(joined))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="parquet root, per docs/PARQUET_LAYOUT.md")
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2100-01-01")
    parser.add_argument("--cache", default="cache/reference")
    parser.add_argument("--download", action="store_true",
                        help="allow fetching the French factor file if not cached")
    parser.add_argument("--no-lag", action="store_true",
                        help="earn the formation day's return, which is look-ahead")
    args = parser.parse_args()

    configure()
    adapter = USAdapter(args.root)
    panel = adapter.panel(args.start, args.end)
    diagnostics = adapter.diagnostics

    tree = from_string(MOMENTUM)
    result = evaluate(tree, panel)

    print(f"tree              {tree}")
    print(f"panel             {panel.n_days} x {panel.n_instruments}")
    print(f"warmup            {result.warmup} rows")
    if diagnostics is not None:
        print(f"delisted names    {diagnostics.n_delisted}")
        print(f"terminal returns  {diagnostics.delist_returns_applied} applied")
        print(f"sector pit        {diagnostics.sector_point_in_time}")

    deciles = decile_returns(result.values, result.warmup, panel, lag=not args.no_lag)
    print(f"rebalances        {deciles.n_rebalances}")
    print()
    print(f"D10 - D1          {deciles.spread_annualised:+.2%} annualised")
    print(f"D10 - market      {deciles.top_minus_market_annualised:+.2%} annualised")

    umd, digest = load_umd(Path(args.cache), allow_download=args.download)
    correlation, overlap = compare_to_umd(deciles.monthly_spread, umd)
    print()
    print(f"UMD file          sha256:{digest}")
    print(f"overlap           {overlap} months")
    print(f"correlation       {correlation:.4f}  (pass above {PASS_CORRELATION})")
    print()

    if correlation > PASS_CORRELATION:
        print("PASS: delisting composition, adjustment and point-in-time alignment")
        print("all reproduce a published effect.")
        return 0
    print("FAIL: something in Layers 0 through 2 is wrong.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
