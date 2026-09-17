"""Bottom-up, pure-numpy evaluation of expression trees.

Returns ``EvalResult(values: float32 array, warmup: int, window_fill: ...)``.

The warmup count accumulates up the tree. Rows before it are not data: they
are reported, never silently returned. This is the single largest look-ahead
risk in the system, and it is what ``scripts/check_lookahead.py`` exists to
falsify.

Invariants:
  * evaluating on ``panel[:k]`` equals evaluating on the full panel and
    slicing to ``[:k]``, for every tree and every k;
  * NaN propagates, never filled;
  * missing observations inside a window are handled per
    ``grammar.WindowPolicy``, subject to ``grammar.min_periods``;
  * division guards against a vanishing denominator (``grammar.DIV_EPS``).

On ``window_fill``
------------------
The result carries realized window fill alongside the values, because the
screen's coverage metric is defined on fill rather than on non-NaN output.
A signal computed from windows barely above ``ceil(0.8 * d)`` present
observations is a different object from one computed on full windows, and a
non-NaN count scores both as fully covered. The evaluator is the only place
that knows how many observations actually entered each result, so it is the
only place that can report it, and it has to do so at the root rather than
per subtree to stay within the memory budget.

STATUS: not implemented yet (Layer 2, after ast.py review).
"""

from __future__ import annotations
