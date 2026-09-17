"""Verification milestone: 12-1 momentum measured against Ken French's UMD.

Builds 12-1 momentum as an expression tree, forms decile portfolios on a
monthly rebalance, and prints the D10 minus D1 annualised spread alongside the
D10 minus market return.

The tree
--------
    div(ts_delay(close, 20), ts_delay(close, 250))

The price twenty days ago over the price two hundred and fifty days ago. It
skips the most recent month, where short-term reversal dominates and would
cancel much of the momentum effect, and it is a ratio rather than a
difference, so it carries no price units. Warmup is 250 rows.

Pass criterion
--------------
Monthly correlation with Ken French's UMD series above 0.9, computed over the
identical sample period.

Correlation, not a level match. A level match would be the wrong test: French
builds UMD from a 2x3 sort on size and prior return over NYSE breakpoints,
value-weighted, with its own universe and its own rebalancing convention. A
decile spread on a different universe will differ in level for reasons that
have nothing to do with whether the signal was computed correctly. What must
agree is the month-to-month shape, because that is what is driven by the
underlying effect rather than by portfolio construction.

A correlation above 0.9 means the pipeline reproduces a known effect. Below
that, something in Layers 0 through 2 is wrong, and it is far cheaper to find
out here, against a published series, than to discover it in the generator
where every candidate is unfamiliar and nothing can be checked by eye.

Reference data
--------------
The monthly momentum factor from the Kenneth R. French data library:

    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/
        F-F_Momentum_Factor_CSV.zip

Fetched once and cached on disk. The run must not depend on network
availability, and repeated runs must compare against identical bytes, so the
cache is keyed by content hash and the hash is recorded with the run. Values
in the file are monthly percentage returns with a YYYYMM index, and the file
carries a trailing annual section that has to be cut before parsing.

STATUS: not implemented yet. Requires the evaluator, which is the next layer.
"""
