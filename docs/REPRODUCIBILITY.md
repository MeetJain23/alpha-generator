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
construction. Counting four is not conservative, it is wrong: three of them
had no independent chance of succeeding, and N enters the deflation through
`sqrt(2 ln N)`, so inflating it discards real results to guard against a risk
that was never taken. `trials.value_hash` holds the hash of the ranked signal
and is what makes the collapse detectable.

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
