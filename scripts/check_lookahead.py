"""Falsify the no-look-ahead claim across 500 random trees.

For each tree, assert that evaluating on ``panel[:k]`` yields values identical
to evaluating on the full panel and slicing to ``[:k]``, for several k, and
that everything before ``warmup`` is reported as invalid rather than returned
as data.

STATUS: not implemented yet (verification milestone, this session).
"""
