"""Alpha research system.

Layered, bottom-up. Each layer depends only on the ones below it:

    Layer 0  alpha.registry  append-only trial ledger (SQLite)
    Layer 1  alpha.data      panel adapters, cost model, trading calendar
    Layer 2  alpha.expr      grammar, AST, evaluator

Nothing here holds global state. Every source of randomness is an explicitly
passed ``numpy.random.Generator``.
"""

__version__ = "0.1.0"
