"""The frozen gauntlet thresholds, and every decision that produced them.

Fitting thresholds on synthetic data before the search is calibration.
Adjusting them after seeing real results is fitting. Afterwards the two look
identical, and the only thing that distinguishes them is a record written
before the fact. A gauntlet loosened whenever nothing survives is not a
gauntlet, it is a formality.

So the thresholds live here as an append-only history. Changing one means
appending a ``ThresholdDecision`` with a date, a reason, the snapshot it was
measured on and the rates it achieved. The fingerprint of the active set goes
into ``runs.config_json`` beside ``source_hash``, so every logged result names
the gauntlet that judged it and a later change cannot be applied backwards.

``tests/test_gauntlet.py`` asserts that the history is contiguous, that every
entry carries a reason, and that the fingerprint matches the thresholds, so an
edit that skips the record fails the suite rather than passing quietly.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Final

from alpha.gauntlet.gauntlet import THRESHOLD_FOR_TEST, GauntletThresholds


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    """One version of the thresholds, and why it exists."""

    version: int
    dated: str
    reason: str
    data_snapshot_id: str
    panel: str
    measured: str
    """The rates this set achieved when it was measured, as reported."""

    limitations: str
    """What is known to be wrong with it. Written down rather than discovered
    later by someone wondering why a class of candidate never survives."""

    thresholds: GauntletThresholds

    def fingerprint(self) -> str:
        return fingerprint(self.thresholds)


def fingerprint(thresholds: GauntletThresholds) -> str:
    """Hash of the threshold values, for ``runs.config_json``.

    Over the values rather than the version number, so an edit that forgets to
    bump the version still changes the fingerprint and still shows up as a
    different gauntlet in the ledger.
    """
    parts = [
        f"{name}={getattr(thresholds, name)!r}"
        for name in sorted(THRESHOLD_FOR_TEST.values())
    ]
    return sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


HISTORY: Final[tuple[ThresholdDecision, ...]] = (
    ThresholdDecision(
        version=1,
        dated="2026-09-19",
        reason=(
            "First calibration. Magnitude threshold read from the null at the "
            "99.9th percentile; the seven robustness thresholds fitted on "
            "planted signals at the weakest IC the power budget is stated for, "
            "retaining 98 per cent of them. Fitted and measured on disjoint "
            "sets of panels."
        ),
        data_snapshot_id="synthetic",
        panel="900 x 120 generated, vol regime 0.4",
        measured=(
            "P(pass screen | null) 0.143% [0.049%, 0.419%]; "
            "P(pass robustness | |IC| >= p99) 61.9% [40.9%, 79.2%]; "
            "P(survive | null) >= 8.8e-4; one expected null survivor at "
            "N = 1131 [301, 5034]. Joint false rejection 15% at IC 0.029 and "
            "0% at IC 0.048, both above tau."
        ),
        limitations=(
            "Synthetic, so every number is a property of the generator rather "
            "than of equities, and all of it is a placeholder until the real "
            "panel lands. Two known problems. The conditional false-acceptance "
            "rate rises sharply with the magnitude cut, from 2.7 per cent at "
            "the median to 61.9 per cent at the 99th percentile, because every "
            "robustness statistic is a ratio with the candidate's own IC in "
            "the denominator: a null candidate that cleared a high bar by luck "
            "has a large denominator and therefore looks stable. The gauntlet "
            "is consequently weakest exactly where it is needed. And "
            "regime_sign at 1.00 requires agreement in both volatility "
            "regimes, which rejects a plant present only in high-volatility "
            "periods 55 per cent of the time at an IC of 0.033, well above "
            "tau. Regime-dependent effects are ordinary, so this threshold is "
            "excluding a real class of alpha by construction and should be "
            "revisited against real data rather than against a generator."
        ),
        thresholds=GauntletThresholds(
            min_abs_ic=0.0235,
            min_fold_sign_agreement=0.75,
            min_fold_ic_fraction=0.1065,
            max_delay_cliff=10.7123,
            min_jitter_plateau=0.6561,
            max_jitter_sign_flips=0,
            min_regime_sign_agreement=1.0,
            min_breadth=-1.1528,
        ),
    ),
)

CURRENT: Final[ThresholdDecision] = HISTORY[-1]


def current_thresholds() -> GauntletThresholds:
    """The thresholds in force. Complete, so a verdict is possible."""
    return CURRENT.thresholds
