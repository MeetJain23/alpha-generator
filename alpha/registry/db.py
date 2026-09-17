"""Append-only SQLite ledger of every candidate the system ever evaluated.

Tables
------
runs    one row per generator run: git SHA, config, data snapshot id.
trials  one row per candidate expression evaluated, INSERT only. A trial row
        is never UPDATEd. A candidate re-examined at a later stage gets a new
        row, and the ledger is the audit trail of what was tried.
pool    expressions that survived every stage, with incremental IC at accept.

Why append-only
---------------
``trial_count(run_id)`` is the N that feeds the Deflated Sharpe Ratio. If a
row could be mutated or removed, N would undercount the real multiple-testing
burden and the deflation would come out optimistic, which is the exact failure
the deflation exists to prevent. Every candidate counts, including the ones
killed on the first check.

That is enforced by triggers rather than by convention. A comment asking
future code not to UPDATE is worth nothing at three in the morning; a trigger
that aborts the transaction is worth something. ``trials`` and ``pool`` both
carry them. ``runs`` does not, since a run may legitimately acquire an end
state later.

Configuration
-------------
``start_run`` records the git SHA together with every constant that changes
what a signal evaluates to, and refuses to write a run without them. It also
records a fingerprint of the operator registry, so that a grammar edit between
two runs is visible in the ledger rather than inferred from the calendar.
Runs whose fingerprints differ are not comparable, whatever their configs say.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping

from alpha.expr import grammar
from alpha.logging_config import get_logger

_log = get_logger(__name__)


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------


class Stage(Enum):
    """The furthest point in the pipeline a candidate reached.

    Ordered, so ``stage_reached`` answers "how far did it get" rather than
    "what was running when it died".
    """

    GENERATED = "generated"
    """Constructed and hashed. A candidate killed here never ran."""

    SCREENED = "screened"
    """Evaluated and passed through the cheap screen."""

    GAUNTLET = "gauntlet"
    """Survived the screen and entered the expensive tests."""

    POOL = "pool"
    """Accepted into the pool."""


class Verdict(Enum):
    """What happened to the candidate at ``stage_reached``."""

    KILLED = "killed"
    PASSED = "passed"
    ACCEPTED = "accepted"


class KillReason(Enum):
    """Canonical kill reasons.

    An enum rather than free text because these get counted. Free-text reasons
    drift into a dozen spellings of the same thing and stop being countable
    exactly when someone wants to know what the search is actually rejecting.
    """

    DUPLICATE = "duplicate"
    """Structural hash already seen in this run."""

    DEGENERATE = "degenerate"
    """Constant, all-NaN, or otherwise carrying no cross-sectional signal."""

    LOW_COVERAGE = "low_coverage"
    """Too few instruments or too little realized window fill."""

    LOW_IC = "low_ic"
    HIGH_TURNOVER = "high_turnover"
    REDUNDANT = "redundant"
    """Too correlated with something already in the pool."""

    ERROR = "error"
    """Evaluation raised. Recorded rather than dropped, because a candidate
    that crashes the evaluator was still a candidate that was tried."""


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunRecord:
    """A generator run, as stored."""

    id: str
    started_at: str
    git_sha: str | None
    config: Mapping[str, Any]
    data_snapshot_id: str | None


@dataclass(frozen=True, slots=True)
class Trial:
    """A candidate evaluation, as written to the ledger.

    Everything except identity is optional, because a candidate killed at
    ``GENERATED`` has no IC to report and inventing a zero would be a lie that
    later aggregates would happily average in.
    """

    run_id: str
    expr_hash: str
    expr_str: str
    stage_reached: Stage
    verdict: Verdict
    value_hash: str | None = None
    ic: float | None = None
    ic_ir: float | None = None
    turnover: float | None = None
    kill_reason: KillReason | str | None = None


@dataclass(frozen=True, slots=True)
class PoolEntry:
    """An accepted expression."""

    trial_id: int
    accepted_at: str
    incremental_ic: float | None


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        id               TEXT PRIMARY KEY,
        started_at       TEXT NOT NULL,
        git_sha          TEXT,
        config_json      TEXT NOT NULL,
        data_snapshot_id TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trials (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        TEXT NOT NULL REFERENCES runs(id),
        expr_hash     TEXT NOT NULL,
        expr_str      TEXT NOT NULL,
        value_hash    TEXT,
        created_at    TEXT NOT NULL,
        stage_reached TEXT NOT NULL,
        ic            REAL,
        ic_ir         REAL,
        turnover      REAL,
        verdict       TEXT NOT NULL,
        kill_reason   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pool (
        trial_id       INTEGER PRIMARY KEY REFERENCES trials(id),
        accepted_at    TEXT NOT NULL,
        incremental_ic REAL
    )
    """,
    # Counting trials per run is the hot read, and it is the one that must not
    # get slow as a run grows, because it happens every time a candidate is
    # deflated.
    "CREATE INDEX IF NOT EXISTS trials_run_idx ON trials(run_id)",
    "CREATE INDEX IF NOT EXISTS trials_hash_idx ON trials(run_id, expr_hash)",
    # Append-only, enforced by the database rather than by good intentions.
    """
    CREATE TRIGGER IF NOT EXISTS trials_no_update
    BEFORE UPDATE ON trials
    BEGIN
        SELECT RAISE(ABORT, 'trials is append-only: log a new trial instead');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trials_no_delete
    BEFORE DELETE ON trials
    BEGIN
        SELECT RAISE(ABORT, 'trials is append-only: N would undercount');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS pool_no_update
    BEFORE UPDATE ON pool
    BEGIN
        SELECT RAISE(ABORT, 'pool is append-only: acceptance is a historical fact');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS pool_no_delete
    BEFORE DELETE ON pool
    BEGIN
        SELECT RAISE(ABORT, 'pool is append-only: acceptance is a historical fact');
    END
    """,
)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def git_sha(repo: Path | None = None) -> str | None:
    """The current commit, suffixed ``-dirty`` if the tree has changes.

    ``None`` when there is no repository, rather than a placeholder string: a
    run that cannot be tied to a commit should be visibly untied, not
    plausibly labelled.

    The dirty suffix matters more than the SHA. A clean SHA identifies the
    code exactly; a dirty one says only that it resembled that commit, and a
    result that cannot be regenerated should say so in the ledger.
    """
    root = repo or Path.cwd()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return f"{sha}-dirty" if dirty else sha


def grammar_fingerprint() -> str:
    """A hash of the operator registry as it currently stands.

    Covers every property that changes what an expression evaluates to: the
    operator set, signatures, admissible windows, warmup rules and window
    policies. Sampling weights are excluded, since they change which
    candidates get drawn but not what any given candidate means.

    Two runs with different fingerprints are not comparable, whatever their
    configs say, because the same expression string can denote different
    computations under different grammars.
    """
    parts: list[str] = []
    for name in sorted(grammar.OPS):
        op = grammar.OPS[name]
        windows = ";".join(
            f"{p.name}={','.join(str(v) for v in p.values)}" for p in op.params
        )
        parts.append(
            f"{op.signature()}|{op.axis.value}|{op.warmup_rule.value}"
            f"|{op.window_policy.value}|{op.domain.value}"
            f"|commutative={int(op.commutative)}|{windows}"
        )
    return sha256("\n".join(parts).encode("utf-8")).hexdigest()


def engine_config() -> dict[str, Any]:
    """Constants that change what every signal in a run evaluates to.

    Merged into ``runs.config_json`` automatically, so that recording them is
    not something a caller can forget. ``MIN_PERIODS_FRACTION`` in particular
    changes every windowed result, so runs with different values cannot be
    compared, and the ledger is where that has to be detectable afterwards.
    """
    return {
        "min_periods_fraction": grammar.MIN_PERIODS_FRACTION,
        "div_eps": grammar.DIV_EPS,
        "windows": list(grammar.WINDOWS),
        "max_arity": grammar.MAX_ARITY,
        "grammar_fingerprint": grammar_fingerprint(),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_run_id() -> str:
    """Sortable by time, unique without coordination.

    Microsecond precision, because two runs started in the same second would
    otherwise sort by their random suffix, and an id that is sortable only
    most of the time is worse than one that is not sortable at all: the
    ordering would look reliable right up to the point where it mattered.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_RESERVED_CONFIG_KEYS: frozenset[str] = frozenset(engine_config())


class Registry:
    """Handle on one ledger database.

    Holds its own connection. There is no module-level connection and no
    implicit default path, so two runs writing to two databases in one process
    cannot silently become one run.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> Registry:
        """Open or create a ledger at ``path``.

        ``:memory:`` is accepted, which is what the tests use.
        """
        target = str(path)
        conn = sqlite3.connect(target)
        conn.row_factory = sqlite3.Row
        # Foreign keys are off by default in SQLite, which would let a trial
        # reference a run that does not exist.
        conn.execute("PRAGMA foreign_keys = ON")
        if target != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        for statement in SCHEMA:
            conn.execute(statement)
        conn.commit()
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    # -- runs --------------------------------------------------------------

    def start_run(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        data_snapshot_id: str | None = None,
        repo: Path | None = None,
    ) -> RunRecord:
        """Open a run, recording the git SHA and the engine configuration.

        The caller's config is merged with ``engine_config()``. A caller key
        that collides with a reserved one raises, rather than overwriting the
        recorded truth with a claim: the point of the column is to say what
        the engine actually used.
        """
        user_config = dict(config or {})
        collisions = _RESERVED_CONFIG_KEYS & user_config.keys()
        if collisions:
            raise ValueError(
                f"config keys {sorted(collisions)} are recorded by the engine "
                f"and cannot be supplied by the caller"
            )

        merged = {**user_config, **engine_config()}
        record = RunRecord(
            id=_new_run_id(),
            started_at=_now(),
            git_sha=git_sha(repo),
            config=merged,
            data_snapshot_id=data_snapshot_id,
        )
        self._conn.execute(
            "INSERT INTO runs (id, started_at, git_sha, config_json, data_snapshot_id)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                record.id,
                record.started_at,
                record.git_sha,
                json.dumps(merged, sort_keys=True),
                record.data_snapshot_id,
            ),
        )
        self._conn.commit()
        _log.info(
            "run started",
            extra={
                "run_id": record.id,
                "git_sha": record.git_sha,
                "data_snapshot_id": record.data_snapshot_id,
                "grammar_fingerprint": merged["grammar_fingerprint"],
            },
        )
        return record

    def run(self, run_id: str) -> RunRecord:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such run: {run_id!r}")
        return RunRecord(
            id=row["id"],
            started_at=row["started_at"],
            git_sha=row["git_sha"],
            config=json.loads(row["config_json"]),
            data_snapshot_id=row["data_snapshot_id"],
        )

    # -- trials ------------------------------------------------------------

    def log_trial(self, trial: Trial) -> int:
        """Append one trial and return its id.

        Insert only. There is no update path, and the database would refuse
        one anyway.
        """
        if trial.verdict is Verdict.KILLED and trial.kill_reason is None:
            raise ValueError(
                "a killed trial needs a kill_reason: an uncounted rejection is "
                "a rejection nobody can learn from"
            )
        if trial.verdict is not Verdict.KILLED and trial.kill_reason is not None:
            raise ValueError(
                f"kill_reason is meaningless on a {trial.verdict.value} trial"
            )

        reason = trial.kill_reason
        reason_text = reason.value if isinstance(reason, KillReason) else reason

        cursor = self._conn.execute(
            "INSERT INTO trials (run_id, expr_hash, expr_str, value_hash,"
            " created_at, stage_reached, ic, ic_ir, turnover, verdict, kill_reason)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trial.run_id,
                trial.expr_hash,
                trial.expr_str,
                trial.value_hash,
                _now(),
                trial.stage_reached.value,
                trial.ic,
                trial.ic_ir,
                trial.turnover,
                trial.verdict.value,
                reason_text,
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def trial_count(self, run_id: str) -> int:
        """Every candidate evaluated in this run.

        This is the N for the Deflated Sharpe Ratio. It counts instantly
        killed candidates, duplicates and evaluation errors, because all of
        them were looks at the data. Any filtering here would understate the
        multiple-testing burden and flatter every surviving result.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM trials WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row["n"])

    def has_expr(self, run_id: str, expr_hash: str) -> bool:
        """Whether this structural hash was already tried in this run."""
        row = self._conn.execute(
            "SELECT 1 FROM trials WHERE run_id = ? AND expr_hash = ? LIMIT 1",
            (run_id, expr_hash),
        ).fetchone()
        return row is not None

    def kill_counts(self, run_id: str) -> dict[str, int]:
        """Rejections by reason, for seeing what the search is throwing away."""
        rows = self._conn.execute(
            "SELECT kill_reason, COUNT(*) AS n FROM trials"
            " WHERE run_id = ? AND verdict = ? GROUP BY kill_reason",
            (run_id, Verdict.KILLED.value),
        ).fetchall()
        return {row["kill_reason"]: int(row["n"]) for row in rows}

    # -- pool --------------------------------------------------------------

    def accept(self, trial_id: int, incremental_ic: float | None = None) -> PoolEntry:
        """Admit a trial to the pool.

        The trial must already carry the ACCEPTED verdict. Acceptance is
        recorded on the trial when it happens and mirrored here; this method
        does not confer it, because a trials row cannot be edited afterwards
        and a pool entry disagreeing with its trial would be unresolvable.
        """
        row = self._conn.execute(
            "SELECT verdict FROM trials WHERE id = ?", (trial_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such trial: {trial_id}")
        if row["verdict"] != Verdict.ACCEPTED.value:
            raise ValueError(
                f"trial {trial_id} has verdict {row['verdict']!r}; only an "
                f"accepted trial can enter the pool"
            )

        entry = PoolEntry(
            trial_id=trial_id,
            accepted_at=_now(),
            incremental_ic=incremental_ic,
        )
        self._conn.execute(
            "INSERT INTO pool (trial_id, accepted_at, incremental_ic) VALUES (?, ?, ?)",
            (entry.trial_id, entry.accepted_at, entry.incremental_ic),
        )
        self._conn.commit()
        _log.info(
            "accepted into pool",
            extra={"trial_id": trial_id, "incremental_ic": incremental_ic},
        )
        return entry

    def pool_entries(self, run_id: str) -> tuple[PoolEntry, ...]:
        """Accepted expressions for one run, oldest first."""
        rows = self._conn.execute(
            "SELECT p.trial_id, p.accepted_at, p.incremental_ic FROM pool AS p"
            " JOIN trials AS t ON t.id = p.trial_id"
            " WHERE t.run_id = ? ORDER BY p.accepted_at, p.trial_id",
            (run_id,),
        ).fetchall()
        return tuple(
            PoolEntry(
                trial_id=int(row["trial_id"]),
                accepted_at=row["accepted_at"],
                incremental_ic=row["incremental_ic"],
            )
            for row in rows
        )


@contextmanager
def open_registry(path: str | Path) -> Iterator[Registry]:
    """Open a ledger and close it afterwards."""
    registry = Registry.open(path)
    try:
        yield registry
    finally:
        registry.close()
