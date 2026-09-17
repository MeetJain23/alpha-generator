"""Layer 2: the expression engine.

``grammar``    what an operator is: arity, types, parameter ranges, warmup rule
``ast``        immutable Node trees - the canonical representation
``evaluator``  bottom-up numpy evaluation with subtree caching
``kernels``    the handful of ops where numpy is pathological (numba)
"""
