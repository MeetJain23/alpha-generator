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
candidate evaluated, including the ones killed instantly, the duplicates and
the evaluation errors.

**What breaks otherwise:** that count is the N in the Deflated Sharpe Ratio. N
is how the deflation knows how many hypotheses were tested, and therefore how
high a Sharpe has to be before it is more than the best of many coin flips. A
mutable or deletable row lets N undercount the real multiple-testing burden,
and an undercounted N makes the deflation optimistic. The deflation would then
be doing the opposite of its job: certifying results it exists to reject.

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

## Practical checklist

Before trusting a logged result:

1. `runs.git_sha` resolves, and has no `-dirty` suffix.
2. A checkout at that SHA produces a `source_hash` equal to the recorded one.
3. `runs.data_snapshot_id` matches the panel you are about to load.
4. `runs.config_json` matches the configuration you are about to use, including
   `grammar_fingerprint` and the seed.
5. `trial_count(run_id)` is the N you deflate by. Not the number of survivors,
   not the number that reached the gauntlet. Every candidate evaluated.

If any of the five fails, the result is not reproducible and should be treated
as an anecdote rather than a measurement.
