"""Layer 3: the cheap screen.

One candidate in, one verdict out, and a row in the ledger for every distinct
hypothesis that was actually tested.

Order of operations
-------------------
Checks run cheapest first, and the ordering is the whole design. A million
candidates go in; the ones that can be rejected without touching the panel
must be rejected before anything touches the panel.

1. Structural duplicate. The expression hash was already tried this run. No
   evaluation, and no ledger row, because it is the same tree.
2. Elide a root rank. Free, and it removes the most expensive no-op available.
3. Evaluate.
4. Coverage and degradation. Both come from the evaluation that already
   happened and neither needs the rank.
5. Rank the signal. This is the expensive step, and everything after it reuses
   the one array.
6. Degeneracy. A signal that assigns almost everyone the same rank has nothing
   to say, whatever its IC.
7. Semantic duplicate. The ranked signal hashes to something already seen, so
   this is a re-spelling of a hypothesis already tested.
8. IC, IC-IR and turnover.

What counts as a trial
----------------------
A candidate rejected before evaluation never looked at the data and cannot
have been lucky, so it gets no row. A candidate whose ranked values match one
already tested is the same hypothesis and gets no row either: ``x`` and
``zscore(x)`` produce the same IC by construction, and recording both would
inflate the N behind the Deflated Sharpe with a trial that had no independent
chance of succeeding.

Everything that reached evaluation with a hypothesis not already tested gets a
row, whether it passed or died, because each of those was a genuine look at
the data.

The collapses are not lost. They are logged, counted on the report, and the
structural hash of the survivor is what the ledger holds, so the audit trail
shows what the search did even where the ledger deliberately does not grow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from alpha.data.types import Panel
from alpha.expr.ast import Node, strip_elidable_root
from alpha.expr.evaluator import EvaluationError, evaluate, value_hash
from alpha.logging_config import get_logger
from alpha.registry.db import KillReason, Registry, Stage, Trial, Verdict
from alpha.screen.metrics import SignalMetrics, rank_forward, summarise

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ScreenConfig:
    """Thresholds. Every one of these belongs in the run config.

    They decide which candidates survive, so two runs with different values
    are not comparable, exactly as with the engine constants.
    """

    min_coverage: float = 0.30
    """Below this the signal is defined for too little of the tradable
    universe to be a strategy, however good the IC looks on what is left."""

    max_degradation: float = 0.50
    """Fraction of emitted cells that came from a partial window. A signal
    running mostly on degraded windows is measuring data availability."""

    min_dispersion: float = 0.01
    """Median per-day spread of the ranked signal."""

    min_abs_ic: float = 0.005
    min_ic_ir: float = 0.20
    max_turnover: float = 0.40
    min_scored_days: int = 250

    horizon: int = 1
    """Days ahead the signal is scored against."""

    def as_config(self) -> dict[str, float | int]:
        """For ``runs.config_json``."""
        return {
            "screen_min_coverage": self.min_coverage,
            "screen_max_degradation": self.max_degradation,
            "screen_min_dispersion": self.min_dispersion,
            "screen_min_abs_ic": self.min_abs_ic,
            "screen_min_ic_ir": self.min_ic_ir,
            "screen_max_turnover": self.max_turnover,
            "screen_min_scored_days": self.min_scored_days,
            "screen_horizon": self.horizon,
        }


@dataclass(frozen=True, slots=True)
class Outcome:
    """What happened to one candidate."""

    tree: Node
    expr_hash: str
    verdict: Verdict
    stage: Stage
    kill_reason: KillReason | None = None
    value_hash: str | None = None
    metrics: SignalMetrics | None = None
    logged: bool = False
    """Whether this produced a ledger row. Collapses do not."""

    elided: int = 0


@dataclass(slots=True)
class ScreenReport:
    """Counts across a run of the screen.

    Mutable by design: it accumulates as candidates are screened, and it is a
    report rather than a record. The ledger is the record.
    """

    seen: int = 0
    logged: int = 0
    passed: int = 0
    structural_duplicates: int = 0
    semantic_collapses: int = 0
    kills: dict[str, int] = field(default_factory=dict)
    elided_nodes: int = 0

    def note_kill(self, reason: KillReason) -> None:
        self.kills[reason.value] = self.kills.get(reason.value, 0) + 1


class Screen:
    """Screens candidates against one panel, writing to one ledger.

    Holds the ranked forward returns, which do not depend on the candidate and
    cost a full-panel rank to produce, plus the hashes seen so far in this
    run. Both are instance state built from what was passed in, so two screens
    over two panels cannot contaminate each other.
    """

    def __init__(
        self,
        panel: Panel,
        registry: Registry,
        run_id: str,
        config: ScreenConfig | None = None,
    ) -> None:
        self.panel = panel
        self.registry = registry
        self.run_id = run_id
        self.config = config or ScreenConfig()
        self.report = ScreenReport()
        self._ranked_forward = rank_forward(panel, self.config.horizon)
        self._expr_seen: set[str] = set()
        self._value_seen: dict[str, str] = {}

    # -- screening ---------------------------------------------------------

    def screen(self, tree: Node) -> Outcome:
        """Run one candidate through, logging it if it earned a row."""
        self.report.seen += 1
        expr_hash = tree.structural_hash()

        if expr_hash in self._expr_seen:
            self.report.structural_duplicates += 1
            return Outcome(tree, expr_hash, Verdict.KILLED, Stage.GENERATED,
                           KillReason.DUPLICATE)
        self._expr_seen.add(expr_hash)

        to_evaluate, elided = strip_elidable_root(tree)
        self.report.elided_nodes += elided

        try:
            result = evaluate(to_evaluate, self.panel)
        except (EvaluationError, ValueError, FloatingPointError) as exc:
            _log.warning("evaluation failed", extra={"expr": str(tree), "error": str(exc)})
            return self._kill(tree, expr_hash, KillReason.ERROR, None, None, elided)

        coverage_ok = self._check_coverage(result)
        if coverage_ok is not None:
            return self._kill(tree, expr_hash, coverage_ok, None, None, elided)

        metrics, ranked = summarise(
            result.values, result.warmup, self.panel, self._ranked_forward
        )

        if metrics.dispersion < self.config.min_dispersion:
            return self._kill(tree, expr_hash, KillReason.DEGENERATE, None, metrics, elided)

        digest = value_hash(ranked)
        seen_as = self._value_seen.get(digest)
        if seen_as is not None:
            self.report.semantic_collapses += 1
            _log.info(
                "collapsed onto an existing hypothesis",
                extra={"expr": str(tree), "same_as": seen_as, "value_hash": digest},
            )
            return Outcome(tree, expr_hash, Verdict.KILLED, Stage.SCREENED,
                           KillReason.DUPLICATE, digest, metrics, logged=False,
                           elided=elided)
        self._value_seen[digest] = expr_hash

        reason = self._judge(metrics)
        if reason is not None:
            return self._kill(tree, expr_hash, reason, digest, metrics, elided)

        self.report.passed += 1
        return self._record(
            tree, expr_hash, Verdict.PASSED, Stage.SCREENED, None, digest, metrics, elided
        )

    def screen_all(self, trees: list[Node]) -> list[Outcome]:
        return [self.screen(tree) for tree in trees]

    # -- checks ------------------------------------------------------------

    def _check_coverage(self, result) -> KillReason | None:
        """Coverage and degradation, both free from the evaluation."""
        from alpha.screen.metrics import coverage as coverage_of

        covered = coverage_of(result.values, result.warmup, self.panel)
        if covered < self.config.min_coverage:
            return KillReason.LOW_COVERAGE
        if result.max_degradation > self.config.max_degradation:
            return KillReason.LOW_COVERAGE
        return None

    def _judge(self, metrics: SignalMetrics) -> KillReason | None:
        """The scoring thresholds, cheapest comparison first."""
        if metrics.n_scored_days < self.config.min_scored_days:
            return KillReason.LOW_COVERAGE
        if not np.isfinite(metrics.ic) or abs(metrics.ic) < self.config.min_abs_ic:
            return KillReason.LOW_IC
        if not np.isfinite(metrics.ic_ir) or abs(metrics.ic_ir) < self.config.min_ic_ir:
            return KillReason.LOW_IC
        if not np.isfinite(metrics.turnover) or metrics.turnover > self.config.max_turnover:
            return KillReason.HIGH_TURNOVER
        return None

    # -- ledger ------------------------------------------------------------

    def _kill(
        self,
        tree: Node,
        expr_hash: str,
        reason: KillReason,
        digest: str | None,
        metrics: SignalMetrics | None,
        elided: int,
    ) -> Outcome:
        self.report.note_kill(reason)
        return self._record(
            tree, expr_hash, Verdict.KILLED, Stage.SCREENED, reason, digest, metrics, elided
        )

    def _record(
        self,
        tree: Node,
        expr_hash: str,
        verdict: Verdict,
        stage: Stage,
        reason: KillReason | None,
        digest: str | None,
        metrics: SignalMetrics | None,
        elided: int,
    ) -> Outcome:
        """Append the trial. Every candidate reaching here looked at the data."""
        self.registry.log_trial(
            Trial(
                run_id=self.run_id,
                expr_hash=expr_hash,
                expr_str=str(tree),
                value_hash=digest,
                stage_reached=stage,
                verdict=verdict,
                ic=_finite(metrics.ic) if metrics else None,
                ic_ir=_finite(metrics.ic_ir) if metrics else None,
                ic_sign=_sign_of(metrics.ic) if metrics else None,
                turnover=_finite(metrics.turnover) if metrics else None,
                kill_reason=reason,
            )
        )
        self.report.logged += 1
        return Outcome(tree, expr_hash, verdict, stage, reason, digest, metrics,
                       logged=True, elided=elided)


def _finite(value: float) -> float | None:
    """NaN is not a measurement, and a NULL column says so."""
    return float(value) if value is not None and np.isfinite(value) else None


def _sign_of(value: float) -> int | None:
    """Which direction this spelling ran in.

    Recorded as an attribute because the hash is orientation-free: x and -x
    are one hypothesis to a screen judging |IC|, so the sign cannot be part of
    identity. It still has to be written down, because the gauntlet asks
    whether it holds across folds.
    """
    if value is None or not np.isfinite(value) or value == 0.0:
        return None
    return 1 if value > 0.0 else -1


DEFAULT_CONFIG: Final[ScreenConfig] = ScreenConfig()
