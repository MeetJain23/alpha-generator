"""Layer 3: the cheap screen.

``metrics`` computes what a signal is worth: forward-aligned rank IC, IC-IR,
turnover, coverage and dispersion. ``screen`` applies thresholds in order of
cost and writes one ledger row per distinct hypothesis that was actually
tested.
"""
