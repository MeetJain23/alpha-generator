# Reproducibility

The contract that makes a logged result regenerable.

A result from this system is a claim: that a particular expression, evaluated
on particular data under a particular configuration, produced a particular
number. The claim is worth nothing if the number cannot be produced again. This
document states what must be recorded for that to hold, and what breaks when
each part is missing.

Every section below follows the same shape as `PARQUET_LAYOUT.md`: the
invariant, then what breaks if it is violated.

## The four things a run needs

To regenerate a result you need the data, the configuration, the randomness
and the code. Nothing else. If all four are pinned, the run is a pure function
and re-running it returns the same numbers.

| what | where it lives | column |
| ---- | -------------- | ------ |
| data | registry | `runs.data_snapshot_id` |
| configuration | registry | `runs.config_json` |
| randomness | registry, inside the config | `runs.config_json` |
| code | outside the registry, pointed at from it | `runs.git_sha`, `runs.source_hash` |

Three of the four live in the ledger as values. The code is the exception: the
registry stores a pointer to it and a fingerprint of it, but not the code
itself. That asymmetry is the source of every reproducibility failure this
document is trying to prevent, because a pointer can dangle and a value cannot.

### Data: `data_snapshot_id`

A content hash of the materialized panel, covering the dates, the instrument
ids and every plane. Computed by `alpha.data.cache.snapshot_id`.

**Invariant:** two runs recording the same `data_snapshot_id` saw the same
numbers.

**What breaks otherwise:** a path and a modification time would not survive a
file being regenerated with different content at the same location, which is
exactly the case worth catching. A vendor reissues a history with restated
market caps, the pipeline rebuilds the panel in place, and every earlier result
silently refers to data that no longer exists. With a content hash the
mismatch is visible; with a path it is invisible.

### Configuration: `config_json`

Every constant that changes what a signal evaluates to, merged automatically by
`engine_config()` rather than supplied by the caller. Currently
`min_periods_fraction`, `div_eps`, `windows`, `max_arity` and
`grammar_fingerprint`.

**Invariant:** runs with different values in this column are not comparable.

`MIN_PERIODS_FRACTION` is the clearest case. It changes the value of every
windowed expression in the system, so a signal evaluated at 0.8 and the same
signal evaluated at 0.6 are different signals wearing the same string. Their
ICs cannot be pooled, ranked against each other, or compared across runs.

**What breaks otherwise:** the comparison happens anyway. Nobody sets out to
compare incomparable runs; they do it because the two numbers are sitting in
the same table and nothing distinguishes them. Recording the constant is what
turns a silent error into a visible one.

A caller supplying a key that collides with a recorded constant raises rather
than overwriting it. The column says what the engine used, not what the caller
claims it used.

`grammar_fingerprint` covers the operator set, signatures, admissible windows,
warmup rules, window policies and domains. Sampling weights are deliberately
excluded: they change which candidates get drawn, not what any given candidate
means. Two runs with different fingerprints are not comparable whatever the
rest of the config says, because the same expression string can denote
different computations under different grammars.

### Randomness: the seed, passed explicitly

All randomness flows from a `numpy.random.Generator` passed as an argument.
There is no module-level generator, no implicit seeding, and no function in
this system calls `np.random.seed` or draws from the global state.

**Invariant:** given the same seed, generation, mutation and crossover produce
the same trees in the same order.

**What breaks otherwise:** a global generator makes a run depend on how many
draws happened before it, which depends on import order, on whether a test ran
first, and on whether anything else in the process drew a number. A search that
found something would not be re-runnable, and worse, it would appear
re-runnable most of the time.

Record the seed in `config_json` alongside the rest of the configuration.

### Code: `git_sha` and `source_hash`

Two columns, because retrieval and verification are different problems.

`git_sha` says **where to find the code**: a commit you can check out. It
carries a `-dirty` suffix when the working tree had uncommitted changes, which
matters more than the SHA itself. A clean SHA identifies the code exactly. A
dirty one says only that the code resembled that commit, and a result that
cannot be regenerated should say so in the ledger rather than looking clean.

`source_hash` says **whether the code you found is the code that ran**: sha256
over every `alpha/**/*.py` in sorted order, each path mixed in alongside its
bytes. It is self-contained. It does not refer to a repository, a branch, a
remote or a history, so nothing done to any of those can invalidate it.

**Invariant:** a checkout whose `source_hash` matches the recorded one is
byte-identical to the code that produced the result, whatever the SHA says.

**What breaks otherwise:** see the next section.

## Git history is immutable, for the same reason `trials` is

### `trials` is append-only

A `trials` row is never updated. A candidate re-examined at a later stage gets
a new row. This is enforced by SQLite triggers rather than convention, because
a comment asking future code not to mutate a row is worth nothing and a trigger
that aborts the transaction is worth something.

**Invariant:** `trial_count(run_id)` only ever grows, and it counts every
distinct hypothesis that was tested against the data.

**What breaks otherwise:** that count is the N in the Deflated Sharpe Ratio. N
is how the deflation knows how many hypotheses were tested, and therefore how
high a Sharpe has to be before it is more than the best of many coin flips. A
mutable or deletable row lets N undercount the real multiple-testing burden,
and an undercounted N makes the deflation optimistic. The deflation would then
be doing the opposite of its job: certifying results it exists to reject.

### What counts as a hypothesis

Three things do not earn a row, and each omission is deliberate.

A candidate rejected before evaluation never looked at the data. A structural
duplicate is the same tree that was already tried, and it cannot have been
lucky a second time without being evaluated a second time.

A candidate whose ranked values match one already tested is the same
hypothesis wearing a different spelling. Rank IC, decile membership and
turnover are all invariant under a strictly monotone per-day transform, so
`x`, `rank(x)`, `zscore(x)` and `log(x)` produce identical numbers by
construction.

**N counts distinct hypotheses tested, not expressions enumerated.** The
collapse is exact, not a conservative approximation. *k* spellings of one
signal are *k* names for a single draw from the null, and the maximum over
them is that draw: not `sigma * sqrt(2 ln k)`, which is what the maximum of
*k* independent draws would be. Recording *k* rows would tell the deflation
the search took *k* chances when it took one.

**The risk therefore runs the other way.** Undercounting is not the hazard
here; over-collapsing is. A hash collision between two genuinely different
hypotheses silently drops a draw, N understates the search, and the deflation
certifies a result it exists to reject. Nothing in the output would look
wrong.

Float rounding before hashing is where that would come from, so
`trials.value_hash` is taken over exact integer ordinal positions rather than
over float ranks. Two distinct orderings differ by a whole unit of position,
not by an epsilon, so a rounding difference cannot manufacture a collision.
`tests/test_evaluator.py` asserts it directly: across four hundred random
trees, any two signals sharing a hash have a rank correlation above 0.99.

### Sign is an attribute, not an identity

A signal and its negation share a hash. The ranks are oriented before hashing,
by an arbitrary deterministic rule: scanning the panel in row-major order, the
first cell that is neither missing nor exactly at the median rank is made
positive.

The screen's verdict is `|IC| > tau`. It has no way to prefer `x` over `-x`
and no business doing so, and the two are one draw from the null wearing two
signs. Counting both would double N for nothing. This halves the reported N,
correctly.

The orientation rule is a function of the signal alone. Nothing about the
forward returns enters it, which is the property that matters: a
canonicalisation that consulted the labels would be fitting the sign to the
data it is about to be scored against, and that leak would be invisible
precisely because the sign is what the screen cannot check.

The direction is not discarded. It moves to `trials.ic_sign`, and decoupling
it from identity is what makes a gauntlet test possible that could not exist
otherwise: whether a candidate's IC keeps its sign across purged folds and
across regimes. A candidate that flips sign between folds fitted its sign to
noise and is dead however large its aggregate `|IC|`. That question is only
askable once identity has stopped depending on the answer.

`sign(x)` deliberately does not collapse onto `x`. It is monotone but not
strictly, and coarsening a continuous signal to three levels changes the
ranks, the deciles and the IC. It is a different hypothesis and gets its own
row.

Everything that reached evaluation with a hypothesis not already tested earns
a row, whether it passed or died. A candidate killed on IC, on turnover, on
coverage or by an evaluation error all looked at the data, and all of them
had their chance.

The collapses are not lost. The screen logs each one and reports the counts,
so the audit trail shows what the search did even where the ledger
deliberately does not grow. What the ledger holds is the set of hypotheses,
and N is the size of that set.

### Git history is append-only too

Do not rewrite published history. No `filter-branch`, no squash-and-force-push,
no amending a commit that has been pushed. Commit forward.

**Invariant:** every `git_sha` in the ledger resolves to a commit that is still
reachable.

**What breaks otherwise:** a rewrite orphans every logged run's `git_sha` at
once. The SHAs in the ledger stop resolving, so a result can no longer be
checked out and verified. The ledger still holds the numbers; what it loses is
the ability to show where they came from.

The cost is concrete, and it arrives later, which is what makes it easy to
discount at the time. Suppose a bug is found in a kernel six months from now.
With intact history you find the commit that fixed it, and you invalidate
exactly the runs whose `git_sha` predates that commit. Every run after it
stands. Without intact history you cannot order the runs against the fix, so
you have to discard everything logged before the bug was found, including the
good results, including the ones that took months of compute. The choice is
between invalidating a subset and invalidating everything.

`source_hash` is the hedge against this having already happened. If a SHA
dangles, a matching source hash still proves the code was byte-identical, and
it survives anything done to the repository or its graph.

## Calibration constants describe a dataset

`tau`, `sigma_SR` and the maximum curve are measurements taken on one panel.
They are not properties of markets, and they are not properties of this code.

**Invariant:** a calibration may only be applied to the panel it was measured
on. `NullCalibration` carries the `data_snapshot_id`, the grammar
fingerprint, the scoring horizon, the permutation block length and the name of
the candidate source, and `assert_applies_to` refuses a mismatch on any of
the first three. The screen calls it in its constructor, and refuses a config
whose `min_abs_ic` is not the measured `tau`.

**What breaks otherwise:** nothing visible. A `tau` measured on a synthetic
panel and applied to real data still produces a pass rate. The screen runs,
candidates survive, the ledger fills, and every number in it is thresholded
against a distribution that was never measured.

This is the same class of guard as `source_hash`, for the same reason: the
failure is silent and flattering, so it is made impossible rather than
written down.

The gap is not small. A generated panel has Gaussian returns, constant
volatility and independent instruments. Real equity returns have fat tails,
volatility clustering, cross-sectional dispersion that differs by a factor of
several between calm and crisis decades, and sector block correlation that
makes a three thousand name universe behave like a far smaller one. Every one
of those changes the width of the null, and therefore `tau`.

Any constant currently in this repository that came from a calibration came
from a synthetic panel and is a placeholder.

### The candidate source is part of the calibration

The maximum of N statistics depends on how correlated the N are. Grammar
candidates already share subtrees, reuse eight fields and draw windows from
one short ladder; an evolutionary search that breeds from survivors is more
correlated still. Calibrating on uniform random trees and then running a
genetic search measures the maximum of the wrong distribution, and in the
direction that overstates the bar.

`calibrate` therefore takes the candidate source as an argument and records
its name. It defaults to uniform random trees, which is correct only for a
search that is also uniform random trees.

## E[max SR] is measured, not assumed

**The empirical maximum curve is authoritative. `sigma * sqrt(2 ln N)` is a
sanity check and decides nothing.**

The False Strategy Theorem exists because the experiment was not runnable. It
assumes the trials are independent, that their Sharpe ratios are Gaussian, and
that an effective N can be estimated. None of the three holds for a grammar
search, and all three were substitutes for a measurement.

The measurement is now cheap. `calibrate` partitions its trials into disjoint
batches of size *k* at a ladder of *k*, takes the maximum within each batch,
and reports the distribution of those maxima. That is `E[max]` at search size
*k*, read off the data, with no independence assumption, no effective-N
estimate and no distributional assumption about the statistic.

It is reported as a curve rather than a number, because the bar depends on how
many candidates were tried and a single figure hides the search size it
assumed. Reading the curve outside the measured ladder raises rather than
extrapolates.

Measured on a synthetic panel, the empirical `E[max SR]` came out 13 to 15 per
cent *below* `sigma_SR * sqrt(2 ln N)` at every size on the ladder. That is
the expected direction and the expected reason: correlated candidates have a
smaller effective number of independent trials, so their maximum is smaller.
Deflating by the analytic value would have set the bar around 15 per cent too
high on that panel. The figure is a property of that panel and that source,
and it has to be remeasured on real data, but the sign of the error is
structural.

## The search budget is set by the false-acceptance rate

Not by compute. This is the arithmetic that decides what the generator has to
be, and it is short enough that nobody does it.

    P(survive | null) = P(pass screen | null)
                      x P(pass gauntlet | passed screen, null)

Expected null survivors at search size N is `N x P(survive | null)`. If the
pool is meant to hold a handful of real alphas, the N at which that count
crosses one is the budget, and no amount of evaluator throughput moves it.

**Measured on the synthetic panel: one expected null survivor at N around
1,100, with a 95 per cent interval of 300 to 5,000.** The supportable search
is a few hundred to a few thousand candidates. Not a million.

### The conditional is the number that matters, and it rises

`P(pass gauntlet)` has to be measured on candidates that already passed the
screen. Those are a different population, and the difference runs the wrong
way:

| magnitude cut | candidates | pass all robustness tests |
| ------------- | ---------- | ------------------------- |
| median        | 1050       | 2.7% |
| 90th pct      | 210        | 13.3% |
| 99th pct      | 21         | 61.9% |

Every robustness statistic is a ratio with the candidate's own IC in the
denominator. A null candidate that cleared a high bar by luck has a large
denominator, so its folds look consistent, its neighbourhood looks flat and
its decay looks smooth. **The gauntlet is weakest exactly where it is needed.**

The rate at tau itself cannot be measured by brute force: one null candidate
in a thousand clears tau, so pinning that rate to within a factor of two needs
roughly twenty events above tau and therefore around fourteen thousand full
gauntlet runs on this panel. The calibration therefore takes the highest cut
with a usable sample as a **lower bound** on the conditional, which makes every
budget it reports an **upper bound** on N. Reading the empty sample at tau as
zero would report no risk at all.

### Every rate carries an interval

A rate of 0.5 per cent measured on two hundred candidates has a 95 per cent
Wilson interval of roughly 0.06 to 2.7 per cent: a factor of forty in the rate
and a factor of forty in the budget. Wilson rather than the normal
approximation, because the counts are small and the rates near zero, which is
where the normal interval extends below zero.

## The generator samples uniformly, and does not evolve

A genetic program is the obvious thing to reach for and it is the wrong tool
here, for three reasons that compound.

**It has no cheap null.** Selection means later generations are drawn from a
region already chosen for high IC *on this data*. The null distribution of a
GP's maximum is therefore only measurable by running the identical
evolutionary loop, same population, same generations, same selection pressure,
against permuted labels. Anything else measures a different search. That null
costs as much as the search itself, every time a parameter changes.

**Its trial count is not its trial count.** A hundred generations of a
thousand individuals is not `1e5` independent trials. It is a search climbing
the noise gradient, and the maximum it reaches under the null sits far above
what uniform sampling reaches at the same nominal count.

**There is no space left to need it.** A GP exists to search spaces too large
to sample. The budget above is a few hundred to a few thousand candidates.
Uniform random sampling covers that comfortably, and its null is one line to
measure.

So `--source` stays an explicit argument with `random_tree` as the only
implementation, and it refuses an unknown name rather than silently defaulting.
Revisiting this needs two things: a false-acceptance rate tight enough to
justify a much larger N, and the budget to run the evolutionary loop's own
null through the same loop.

## Unsatisfied milestones

Two, and both need the real panel. Neither can be closed with generated data,
and nothing downstream should be read as validated until they are.

### `verify_momentum.py`

12-1 momentum against Ken French's UMD, monthly correlation above 0.9. It
exists to validate the data pipeline: delisting composition, adjustment
factors and point-in-time alignment are all satisfied by construction in a
generated panel, so synthetic data cannot exercise any of them.

### Real-data power

Run 12-1 momentum, short-term reversal and low-volatility through the full
gauntlet on the real panel.

These three are the only ground-truth positives that exist. They have been
replicated across decades, across markets and by people with no stake in this
code. If the gauntlet rejects one of them, the gauntlet is wrong, and that is
a conclusion no synthetic plant can deliver: a plant is a thing this
repository invented, and it can only confirm that the tests detect what the
generator was told to put there.

The synthetic calibration already points at where this is likely to fail. A
plant present only in high-volatility regimes reaches an IC of 0.033, well
above tau, and is still rejected 55 per cent of the time by the
regime-agreement test, which is set to require both regimes. Regime-dependent
effects are ordinary. Low-volatility, in particular, is an effect whose
behaviour differs sharply between calm and turbulent periods, so it is a live
candidate to be rejected for being what it is.

## The asset line

The repository splits in two, and the split is not about sensitivity in the
usual sense. Everything here is either infrastructure, which has no value to
anyone who does not also have the data and the compute, or it is the result,
which is the entire point of the exercise.

### Public: the machinery

The grammar, the evaluator, the kernels, the screen, the gauntlet, the
adapters, the scripts and the tests. All of it.

There is no edge in any of it. An operator set, a rank IC and a purged
cross-validation are published in a dozen textbooks, and anyone capable of
using this code already knows how to write it. What takes the time is getting
the details right, and the details are the assertions, which are worth more
to their author as review-bait than as a secret. A rolling variance that
cancels in float32, a window that emits before its warmup, a decay that fails
to renormalise: those are the kind of thing another reader catches, and
publishing is how the reader arrives.

### Never committed: the result

The registry database. The accepted pool. Any surviving expression, in any
form, including in a commit message, a docstring, a test fixture, a README
example or a notebook output cell.

**Invariant:** no file matching the registry, pool or panel patterns is ever
tracked, and no surviving expression appears in a tracked file.

**What breaks otherwise:** a public repository containing surviving
expressions is a public repository containing the strategy. Not a hint at it
or a description of it. The expression string is the strategy, it is short
enough to fit in a tweet, and `from_string` turns it back into a running
signal in one call. There is no partial disclosure here: one line of a pool
table is the whole thing.

This is the one asymmetry worth being careful about. The machinery took most
of the effort and is worth publishing. The output took most of the compute and
is worth nothing once shared, because an alpha that other people are trading
is an alpha that has already been arbitraged. Publishing the first costs
nothing and publishing the second costs everything, and the two live in the
same working directory.

### How it is enforced

Not by care. `.gitignore` anchors `/data/`, `/cache/`, `/runs/` and
`/scratch/` to the repository root, and denies `*.db`, `*.sqlite`, `*.parquet`,
`*.npz`, `*.csv` and `*.ipynb` at any depth. `tests/test_repository.py`
asserts that nothing matching those patterns is tracked, and `pre-commit`
refuses the commit rather than reporting it afterwards.

Notebooks are denied outright rather than filtered, because a notebook carries
its output cells inside the file. A `head()` of a vendor panel or a printed
pool table becomes a committed artefact, and it survives review because nobody
reads a JSON blob of base64.

### The consequence for reproducibility

A result logged in the registry cannot be reproduced from this repository
alone, and that is deliberate rather than an oversight. The four things a run
needs are listed at the top of this document, and two of them, the data and
the ledger, live outside version control on purpose.

What the repository guarantees is that given the data and the ledger, the code
that produced a number can be identified exactly, through `git_sha` and
`source_hash`. That is the half that can be made public without giving
anything away.

## Practical checklist

Before trusting a logged result:

1. `runs.git_sha` resolves, and has no `-dirty` suffix.
2. A checkout at that SHA produces a `source_hash` equal to the recorded one.
3. `runs.data_snapshot_id` matches the panel you are about to load.
4. `runs.config_json` matches the configuration you are about to use, including
   `grammar_fingerprint` and the seed.
5. `trial_count(run_id)` is the N you deflate by. Not the number of survivors,
   not the number that reached the gauntlet, and not the number of candidates
   the generator emitted. Every distinct hypothesis that was tested.

If any of the five fails, the result is not reproducible and should be treated
as an anecdote rather than a measurement.
