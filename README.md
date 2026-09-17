# alpha-generator

A US equity alpha research system, built bottom up in numpy, pandas, scipy and
pyarrow. No ML libraries and no backtesting frameworks, so every number is
traceable to code in this repository.

The system exists to do one thing honestly: evaluate a large number of
candidate signals and report which of them, if any, survive after accounting
for the fact that a large number were tried.

## Layers

| layer | package          | what it owns                                 | status                  |
| ----- | ---------------- | -------------------------------------------- | ----------------------- |
| 0     | `alpha.registry` | append-only trial ledger (SQLite)            | done                    |
| 1     | `alpha.data`     | panel adapters, cost model, trading calendar | done                    |
| 2     | `alpha.expr`     | grammar, AST, evaluator, numba kernels       | grammar and AST done    |

Each layer depends only on the ones below it. There is no global state
anywhere, and every source of randomness is an explicitly passed
`numpy.random.Generator`.

### Layer 0, registry

`trials`, `runs`, `pool`. A `trials` row is never updated. A candidate
re-examined at a later stage gets a new row.

The reason is `trial_count(run_id)`, which is the N that feeds the Deflated
Sharpe Ratio. If a row could be mutated or deleted, N would undercount the
real multiple-testing burden and the deflation would come out optimistic,
which is the exact failure the deflation exists to prevent. Every candidate
counts, including the ones killed on the first check.

`runs.config_json` records every constant that changes what a signal
evaluates to, `MIN_PERIODS_FRACTION` among them, alongside `git_sha` for
retrieval and `source_hash` for verification. Runs with different values are
not comparable, and recording them is what makes that detectable afterwards
rather than a matter of memory.

`docs/REPRODUCIBILITY.md` is the contract that makes a logged result
regenerable: what has to be recorded, and what breaks when each part is
missing.

### Layer 1, data

One `Panel`: a `DatetimeIndex`, an array of stable string instrument ids, and
a dict of (n_days x n_instruments) float32 planes.

The field schema is deliberately wider than the US needs. `band_upper`,
`band_lower`, `surveillance_flag` and `lot_size` are constants here. They stay
in the schema so an expression written against one market is structurally
valid against another.

Fundamentals beyond `mcap` are deferred, not dropped. Book value, earnings and
accruals need a point-in-time source carrying both a report date and a
knowledge date. Adding them against a source without the latter would
introduce restatement look-ahead that no assertion in this layer could catch,
so they wait for the right source rather than arriving on a weak one.

`docs/PARQUET_LAYOUT.md` documents the on-disk contract `USAdapter` expects,
the invariants it asserts at load, and the leaks it can only measure and
report. `alpha.data.synthetic` generates a conforming panel so everything
above is testable before real data exists.

### Layer 2, expression engine

- `grammar.py`, the typed operator vocabulary. Data, not behaviour.
- `ast.py`, immutable `Node` trees. Trees are canonical, strings are for
  display and hashing only.
- `evaluator.py`, bottom-up numpy evaluation with subtree caching.
- `kernels.py`, numba, for the ops where numpy is pathological.

## The grammar

30 operators. Everything the search can express lives in
`alpha/expr/grammar.py`, and nothing can be registered at runtime.

| family  | operators                                                                              |
| ------- | -------------------------------------------------------------------------------------- |
| fields  | `open` `high` `low` `close` `volume` `vwap` `returns` `mcap`                             |
| group   | `sector`, a `GROUP` typed label plane rather than a matrix                                |
| unary   | `rank` `zscore` `log` `abs` `sign`                                                       |
| ts      | `ts_delay` `delta` `ts_mean` `ts_std` `ts_rank` `ts_min` `ts_max` `ts_argmax` `decay_linear` |
| binary  | `add` `sub` `mul` `div` `correlation`                                                    |
| grouped | `demean_by` `rank_within`                                                                |

Windows come from a fixed ladder, 1, 2, 3, 5, 10, 20, 60, 120, 250, with
per-operator floors that exclude the degenerate short end. `ts_mean(x, 1)` is
the identity, `ts_std(x, 2)` is a scaled first difference, and `ts_rank` below
5 carries little beyond the sign of a recent change. A coarse, log-spaced
ladder is deliberate: adjacent windows produce near-identical signals, so a
dense ladder multiplies the multiple-testing burden without adding hypotheses.

### Axes

Every operator couples along exactly one axis.

- `Axis.NONE`, elementwise.
- `Axis.CROSS_SECTION`, per day across instruments: `rank`, `zscore`,
  `demean_by`, `rank_within`.
- `Axis.TIME_SERIES`, per instrument across time: everything windowed.

Only time-series ops look backwards, so only they cost warmup. The registry
asserts at import that windowed, parameterised and time-series always agree.

### Warmup

`warmup(node)` counts the leading rows that are not valid data. Row `warmup`
is the first computed entirely from in-window observations, and rows before it
are reported as invalid rather than returned as values.

Two rules cover everything.

| rule             | meaning                                           | operators                                 |
| ---------------- | ------------------------------------------------- | ----------------------------------------- |
| `WINDOW_MINUS_1` | a *d* observation window first closes at *d-1*    | all `ts_*` aggregates, `decay_linear`, `correlation` |
| `WINDOW`         | a value reading *d* days back first exists at *d* | `delta`, `ts_delay`                       |

Warmup accumulates up the tree, and `grammar.warmup()` is the only place in
the system that does the arithmetic. The evaluator calls it and stores the
result rather than re-deriving offsets per operator, because per-operator
re-derivation is precisely where an off-by-one hides.

Warmup is structural: a count of row indices from window lengths alone,
identical for every instrument. It is not a statement about data availability.
An instrument that IPOs mid-panel is NaN before its listing, which is the NaN
policy and is per-instrument. Warmup says "this row could not have been
computed yet". NaN says "there was nothing to compute it from". Conflating the
two is how look-ahead gets in.

### Missing data inside a window

`MIN_PERIODS_FRACTION` is 0.8, so a rolling result needs
`ceil(0.8 * d)` present observations. `WindowPolicy` declares how each
windowed operator handles the rest, because the answer differs by operator.

| policy        | operators            | behaviour                                                       |
| ------------- | -------------------- | --------------------------------------------------------------- |
| `POINT`       | `delta`, `ts_delay`  | exempt; two endpoints required, whatever lies between            |
| `MIN_PERIODS` | the `ts_*` aggregates | compute over present values, subject to the floor                |
| `RENORMALIZE` | `decay_linear`       | rescale weights over present observations                        |
| `PAIRWISE`    | `correlation`        | a day counts only if both inputs are present                     |

`decay_linear` has to renormalise. Treating a missing observation as zero
weight without rescaling would shrink the result in proportion to how much
data an instrument is missing, turning missingness into a stealth liquidity
factor: frequently halted names would carry systematically smaller signal
magnitudes for no reason anyone chose.

Excluding missing values from a window is the same operation the
cross-sectional ops already perform on the day's population, and it adds no
look-ahead, since the window still ends at *t*. The floor is what keeps it
honest by refusing a result built from a handful of scattered observations.

Because this constant changes every signal, the screen's coverage metric
measures realized window fill rather than only whether the output is non-NaN.
A signal that produces values everywhere while running on windows that are
barely above the floor is a different object from one running on full windows,
and the coverage metric has to make that visible rather than scoring both as
fully covered.

### Normalisation

Cross-sectional ranks are centred:

```
rank = (position + 0.5) / n - 0.5,   range (-0.5, 0.5)
```

`position` is the 0-based ascending position among the day's non-NaN entries,
ties take the average position, and `n` is the day's non-NaN count. The output
sums to zero, so a ranked signal is already a zero-net-exposure score and
needs no separate demeaning before weighting.

Centring makes `rank` and `zscore` interchangeable along one dimension only,
robustness. Both produce a centred cross-sectional score, but `rank` bounds
the influence of a tail value while `zscore` lets that value set the day's
scale. Swapping one for the other is a robustness choice rather than a
different hypothesis, and the generator should price the pair as one idea.

`ts_rank` is deliberately not centred. It is a within-instrument feature
feeding further operators, not a cross-sectional score, and centring it would
imply a neutrality across instruments that it does not have.

### NaN policy

NaN propagates. No operator fills, interpolates or drops. Cross-sectional ops
exclude NaN from the day's population rather than treating it as a value.

A null group label gives NaN, with no residual bucket. Pooling unclassified
names into a synthetic group would demean a company against an arbitrary set
that shares nothing except that the vendor failed to label them, producing
neutrality against a group that does not exist.

Two cases where a numerically defined input still yields NaN, both declared on
the operator as a `Domain`:

- `log(x)` for `x <= 0` gives NaN rather than negative infinity. A
  non-positive price is bad data.
- `div(x, y)` for `|y| <= DIV_EPS` gives NaN rather than an infinity. An
  infinity survives every downstream operator and surfaces only as a nonsense
  IC, whereas a NaN is caught by the coverage checks.

## Verification

`scripts/verify_momentum.py` builds 12-1 momentum as
`div(ts_delay(close, 20), ts_delay(close, 250))`, forms monthly rebalanced
deciles, and checks monthly correlation above 0.9 against Ken French's UMD
over the identical sample. Correlation rather than a level match, because
French's construction differs in universe, weighting and breakpoints, and only
the month-to-month shape is driven by the underlying effect.

`scripts/check_lookahead.py` asserts prefix equality across 500 random trees:
evaluating on `panel[:k]` must equal evaluating on the full panel and slicing
to `[:k]`. `Panel.head(k)` exists so that both sides of that comparison are
built the same way rather than improvised per test.

## Trees

Expression trees are canonical, immutable and valid by construction. A tree
that does not type-check against the grammar cannot be built, so the ledger
never records a candidate that was never a hypothesis.

Commutative operators have their children reordered at construction, so
`add(close, open)` and `add(open, close)` are one object that compares equal,
prints the same and hashes the same. Hashing is bottom up: children are hashed
first, those hashes are sorted, and the parent is hashed from them. A parent's
hash does not exist while its children are still being built, so the order is
forced rather than chosen. Hashes are 128 bits, because the ledger keys trials
on them and a collision would merge two hypotheses into one row.

Generation is type-directed, so there is no generate-and-reject loop, and the
group argument of `demean_by` or `rank_within` can only ever resolve to
`sector`. Every function takes an explicit `numpy.random.Generator`.

## Development

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

## Layout

```
alpha/
  registry/db.py        layer 0: append-only ledger
  data/
    types.py            Panel, CostModel, Adapter protocol
    us_adapter.py       parquet to Panel
    synthetic.py        conforming panels for tests
    cache.py            npz materialisation, mmap reload
  expr/
    grammar.py          typed operator vocabulary
    ast.py              immutable trees
    evaluator.py        bottom-up evaluation
    kernels.py          numba incremental kernels
  logging_config.py     structured JSON logging
scripts/
  verify_momentum.py    12-1 momentum against French UMD
  check_lookahead.py    prefix equality across 500 random trees
docs/
  PARQUET_LAYOUT.md     what USAdapter expects on disk
```

`scripts/` is the only place `print` is allowed. Library code logs.
