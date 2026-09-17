"""numba kernels for the ops where numpy is pathological.

Only these: ts_rank, ts_argmax, ts_min, ts_max, rolling correlation and
decay_linear. Each uses an incremental algorithm (monotonic deque for the
running extrema, running sums for correlation) so cost is O(n) in the panel
rather than O(n * window). Everything else stays pure numpy.

STATUS: not implemented yet (Layer 2, after grammar review).
"""

from __future__ import annotations
