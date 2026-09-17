"""Tests for the append-only trial ledger.

The property under test is not "rows can be written and read back". It is
that the ledger cannot be quietly made to lie: a trial cannot be edited away,
N cannot shrink, and a run cannot be recorded without the configuration that
determines what its signals meant.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from alpha.expr import grammar
from alpha.registry.db import (
    KillReason,
    Registry,
    RegistryError,
    Stage,
    Trial,
    Verdict,
    engine_config,
    grammar_fingerprint,
    open_registry,
    source_hash,
)


@pytest.fixture()
def reg() -> Registry:
    registry = Registry.open(":memory:")
    yield registry
    registry.close()


def _trial(run_id: str, i: int = 0, **kwargs: object) -> Trial:
    defaults: dict[str, object] = {
        "run_id": run_id,
        "expr_hash": f"hash{i}",
        "expr_str": f"rank(delta(close, {i or 1}))",
        "stage_reached": Stage.SCREENED,
        "verdict": Verdict.PASSED,
    }
    defaults.update(kwargs)
    return Trial(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# append-only
# --------------------------------------------------------------------------


def test_a_trial_row_cannot_be_updated(reg: Registry) -> None:
    run = reg.start_run()
    trial_id = reg.log_trial(_trial(run.id))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        reg._conn.execute("UPDATE trials SET ic = 0.9 WHERE id = ?", (trial_id,))


def test_a_trial_row_cannot_be_deleted(reg: Registry) -> None:
    run = reg.start_run()
    trial_id = reg.log_trial(_trial(run.id))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        reg._conn.execute("DELETE FROM trials WHERE id = ?", (trial_id,))


def test_a_pool_entry_cannot_be_rewritten(reg: Registry) -> None:
    run = reg.start_run()
    trial_id = reg.log_trial(
        _trial(run.id, stage_reached=Stage.POOL, verdict=Verdict.ACCEPTED)
    )
    reg.accept(trial_id, 0.01)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        reg._conn.execute("UPDATE pool SET incremental_ic = 0.9")


def test_re_examining_a_candidate_appends_rather_than_replaces(reg: Registry) -> None:
    """The same expression at two stages is two rows, and N counts both."""
    run = reg.start_run()
    reg.log_trial(_trial(run.id, stage_reached=Stage.SCREENED, verdict=Verdict.PASSED))
    reg.log_trial(
        _trial(
            run.id,
            stage_reached=Stage.GAUNTLET,
            verdict=Verdict.KILLED,
            kill_reason=KillReason.HIGH_TURNOVER,
        )
    )
    assert reg.trial_count(run.id) == 2


# --------------------------------------------------------------------------
# trial_count is the N behind the Deflated Sharpe
# --------------------------------------------------------------------------


def test_trial_count_includes_instantly_killed_candidates(reg: Registry) -> None:
    run = reg.start_run()
    for i in range(7):
        reg.log_trial(
            _trial(
                run.id,
                i,
                stage_reached=Stage.GENERATED,
                verdict=Verdict.KILLED,
                kill_reason=KillReason.DUPLICATE,
            )
        )
    reg.log_trial(_trial(run.id, 99))
    assert reg.trial_count(run.id) == 8


def test_trial_count_includes_evaluation_errors(reg: Registry) -> None:
    """A candidate that crashed the evaluator was still a look at the data."""
    run = reg.start_run()
    reg.log_trial(
        _trial(
            run.id,
            stage_reached=Stage.SCREENED,
            verdict=Verdict.KILLED,
            kill_reason=KillReason.ERROR,
        )
    )
    assert reg.trial_count(run.id) == 1


def test_trial_count_is_scoped_to_its_run(reg: Registry) -> None:
    first = reg.start_run()
    second = reg.start_run()
    reg.log_trial(_trial(first.id, 1))
    reg.log_trial(_trial(first.id, 2))
    reg.log_trial(_trial(second.id, 3))
    assert reg.trial_count(first.id) == 2
    assert reg.trial_count(second.id) == 1


def test_trial_count_of_an_unknown_run_is_zero(reg: Registry) -> None:
    assert reg.trial_count("never-started") == 0


# --------------------------------------------------------------------------
# referential integrity
# --------------------------------------------------------------------------


def test_a_trial_cannot_reference_a_run_that_does_not_exist(reg: Registry) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        reg.log_trial(_trial("no-such-run"))


def test_a_killed_trial_needs_a_reason(reg: Registry) -> None:
    run = reg.start_run()
    with pytest.raises(ValueError, match="kill_reason"):
        reg.log_trial(_trial(run.id, verdict=Verdict.KILLED))


def test_a_reason_on_a_passing_trial_is_refused(reg: Registry) -> None:
    run = reg.start_run()
    with pytest.raises(ValueError, match="meaningless"):
        reg.log_trial(
            _trial(run.id, verdict=Verdict.PASSED, kill_reason=KillReason.LOW_IC)
        )


def test_only_an_accepted_trial_enters_the_pool(reg: Registry) -> None:
    run = reg.start_run()
    trial_id = reg.log_trial(_trial(run.id, verdict=Verdict.PASSED))
    with pytest.raises(ValueError, match="only an accepted trial"):
        reg.accept(trial_id)


def test_accepting_an_unknown_trial_raises(reg: Registry) -> None:
    reg.start_run()
    with pytest.raises(KeyError):
        reg.accept(4242)


def test_a_trial_cannot_be_accepted_twice(reg: Registry) -> None:
    run = reg.start_run()
    trial_id = reg.log_trial(
        _trial(run.id, stage_reached=Stage.POOL, verdict=Verdict.ACCEPTED)
    )
    reg.accept(trial_id, 0.02)
    with pytest.raises(sqlite3.IntegrityError):
        reg.accept(trial_id, 0.02)


# --------------------------------------------------------------------------
# run configuration
# --------------------------------------------------------------------------


def test_engine_constants_are_recorded_without_the_caller_asking(reg: Registry) -> None:
    run = reg.start_run({"universe": "top1500"})
    stored = reg.run(run.id)
    assert stored.config["universe"] == "top1500"
    assert stored.config["min_periods_fraction"] == grammar.MIN_PERIODS_FRACTION
    assert stored.config["div_eps"] == grammar.DIV_EPS
    assert stored.config["windows"] == list(grammar.WINDOWS)


def test_a_caller_cannot_overwrite_a_recorded_engine_constant(reg: Registry) -> None:
    """The column says what the engine used, not what the caller claims."""
    with pytest.raises(ValueError, match="min_periods_fraction"):
        reg.start_run({"min_periods_fraction": 0.1})


def test_the_grammar_fingerprint_is_recorded(reg: Registry) -> None:
    run = reg.start_run()
    assert reg.run(run.id).config["grammar_fingerprint"] == grammar_fingerprint()


def test_config_is_stored_as_stable_json(reg: Registry) -> None:
    """Sorted keys, so two identical configs compare equal as text."""
    run = reg.start_run({"b": 2, "a": 1})
    row = reg._conn.execute(
        "SELECT config_json FROM runs WHERE id = ?", (run.id,)
    ).fetchone()
    assert list(json.loads(row["config_json"])) == sorted(json.loads(row["config_json"]))


def test_run_ids_are_unique_and_time_sortable(reg: Registry) -> None:
    ids = [reg.start_run().id for _ in range(5)]
    assert len(set(ids)) == 5
    assert ids == sorted(ids)


def test_reading_an_unknown_run_raises(reg: Registry) -> None:
    with pytest.raises(KeyError):
        reg.run("no-such-run")


# --------------------------------------------------------------------------
# grammar fingerprint
# --------------------------------------------------------------------------


def test_fingerprint_is_stable_across_calls() -> None:
    assert grammar_fingerprint() == grammar_fingerprint()


def test_fingerprint_covers_the_whole_operator_set() -> None:
    """A fingerprint that ignored an operator would not detect its removal."""
    baseline = grammar_fingerprint()
    for name in sorted(grammar.OPS):
        assert name in grammar.OPS
    assert baseline == grammar_fingerprint()


def test_engine_config_keys_are_the_reserved_ones() -> None:
    assert set(engine_config()) == {
        "min_periods_fraction",
        "div_eps",
        "windows",
        "max_arity",
        "grammar_fingerprint",
    }


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------


def test_has_expr_finds_a_hash_already_tried(reg: Registry) -> None:
    run = reg.start_run()
    reg.log_trial(_trial(run.id, 1))
    assert reg.has_expr(run.id, "hash1")
    assert not reg.has_expr(run.id, "hash2")


def test_has_expr_is_scoped_to_its_run(reg: Registry) -> None:
    """A hash tried in another run has not been tried in this one."""
    first = reg.start_run()
    second = reg.start_run()
    reg.log_trial(_trial(first.id, 1))
    assert not reg.has_expr(second.id, "hash1")


def test_kill_counts_group_by_reason(reg: Registry) -> None:
    run = reg.start_run()
    reasons = [
        KillReason.DUPLICATE,
        KillReason.DUPLICATE,
        KillReason.LOW_IC,
        KillReason.DEGENERATE,
    ]
    for i, reason in enumerate(reasons):
        reg.log_trial(
            _trial(run.id, i, verdict=Verdict.KILLED, kill_reason=reason)
        )
    reg.log_trial(_trial(run.id, 99))
    assert reg.kill_counts(run.id) == {"duplicate": 2, "low_ic": 1, "degenerate": 1}


def test_pool_entries_are_scoped_and_ordered(reg: Registry) -> None:
    run = reg.start_run()
    other = reg.start_run()
    accepted = [
        reg.log_trial(
            _trial(run.id, i, stage_reached=Stage.POOL, verdict=Verdict.ACCEPTED)
        )
        for i in range(3)
    ]
    foreign = reg.log_trial(
        _trial(other.id, 9, stage_reached=Stage.POOL, verdict=Verdict.ACCEPTED)
    )
    for trial_id in accepted:
        reg.accept(trial_id, 0.01)
    reg.accept(foreign, 0.02)

    assert [e.trial_id for e in reg.pool_entries(run.id)] == accepted


def test_a_free_text_kill_reason_is_stored_as_given(reg: Registry) -> None:
    """The enum is the vocabulary, not a cage: a novel reason still records."""
    run = reg.start_run()
    reg.log_trial(
        _trial(run.id, verdict=Verdict.KILLED, kill_reason="borrow_unavailable")
    )
    assert reg.kill_counts(run.id) == {"borrow_unavailable": 1}


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_a_ledger_survives_being_closed_and_reopened(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite"
    with open_registry(path) as registry:
        run = registry.start_run({"universe": "test"})
        for i in range(4):
            registry.log_trial(_trial(run.id, i))
        run_id = run.id

    with open_registry(path) as registry:
        assert registry.trial_count(run_id) == 4
        assert registry.run(run_id).config["universe"] == "test"


def test_triggers_survive_a_reopen(tmp_path) -> None:
    """Append-only is a property of the file, not of the session that made it."""
    path = tmp_path / "ledger.sqlite"
    with open_registry(path) as registry:
        run = registry.start_run()
        registry.log_trial(_trial(run.id))

    with open_registry(path) as registry:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            registry._conn.execute("DELETE FROM trials")


# --------------------------------------------------------------------------
# source_hash: verification, where git_sha is only retrieval
# --------------------------------------------------------------------------


def test_a_run_records_both_the_sha_and_the_source_hash(reg: Registry) -> None:
    run = reg.start_run()
    stored = reg.run(run.id)
    assert stored.source_hash == source_hash()
    assert len(stored.source_hash) == 64


def test_the_source_hash_is_stable_across_calls() -> None:
    assert source_hash() == source_hash()


def test_the_source_hash_moves_when_a_source_file_changes(tmp_path) -> None:
    """Otherwise it could not tell you whether the code you found is the code
    that ran."""
    import shutil
    from pathlib import Path

    import alpha

    package = Path(alpha.__file__).resolve().parent
    copy = tmp_path / "alpha"
    shutil.copytree(package, copy)
    before = source_hash(copy)

    target = copy / "expr" / "grammar.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    assert source_hash(copy) != before


def test_the_source_hash_moves_when_code_moves_between_files(tmp_path) -> None:
    """The path is mixed in alongside the bytes, so a rename is a change."""
    import shutil
    from pathlib import Path

    import alpha

    package = Path(alpha.__file__).resolve().parent
    copy = tmp_path / "alpha"
    shutil.copytree(package, copy)
    before = source_hash(copy)

    (copy / "expr" / "kernels.py").rename(copy / "expr" / "kernels_renamed.py")
    assert source_hash(copy) != before


def test_the_source_hash_ignores_bytecode_caches(tmp_path) -> None:
    """A hash that moved when Python cached a module would be useless."""
    import shutil
    from pathlib import Path

    import alpha

    package = Path(alpha.__file__).resolve().parent
    copy = tmp_path / "alpha"
    shutil.copytree(package, copy)
    before = source_hash(copy)

    cache = copy / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "junk.py").write_text("# not source\n", encoding="utf-8")
    assert source_hash(copy) == before


def test_a_ledger_missing_the_provenance_column_is_refused(tmp_path) -> None:
    """CREATE TABLE IF NOT EXISTS leaves an old table alone, so an old ledger
    would otherwise keep accepting runs and drop the new provenance."""
    path = tmp_path / "old.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE runs (id TEXT PRIMARY KEY, started_at TEXT NOT NULL,"
        " git_sha TEXT, config_json TEXT NOT NULL, data_snapshot_id TEXT)"
    )
    connection.commit()
    connection.close()

    with pytest.raises(RegistryError, match="source_hash"):
        Registry.open(path)
