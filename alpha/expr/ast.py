"""Immutable expression trees.

Trees are canonical. Strings exist for display and for the ledger, and are
never the thing that gets reasoned about: a string has to be parsed to be
understood, and anything that parses strings to make decisions will eventually
make a decision about whitespace.

A ``Node`` is a frozen, hashable dataclass, so subtree identity, and therefore
the evaluator's cache key, is structural. Construction validates against the
grammar, so an ill-formed tree cannot exist. That matters more than it sounds:
the ledger is append-only, so a candidate that was never a valid hypothesis
would still be counted forever in the N behind the Deflated Sharpe.

Canonical form
--------------
Commutative operators have their children reordered at construction, so
``add(close, open)`` and ``add(open, close)`` are the same object, compare
equal, print the same and hash the same. Without that, the search would spend
two trials on one idea and permanently inflate N.

The ordering is by child structural hash, computed bottom up: hash the
children first, sort those hashes, then hash the parent from them. Sorting by
anything else, or sorting before the children have hashes, does not produce a
canonical form. A parent's hash is not available while its children are still
being built, so the order of operations here is the whole of the correctness
argument.

Hashes are 128 bits. The ledger keys trials on them, so a collision would
merge two distinct hypotheses into one row, and 64 bits is not enough margin
for a search that runs to millions of candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Final, Iterator

import numpy as np

from alpha.expr import grammar
from alpha.expr.grammar import DType, OpSpec

HASH_BITS: Final[int] = 128
_HASH_CHARS: Final[int] = HASH_BITS // 4


@dataclass(frozen=True, slots=True)
class Node:
    """One node of an expression tree.

    ``op`` is an operator name rather than an ``OpSpec``. The registry is
    immutable and global to the process, so the name is the whole of the
    identity, and it keeps a node cheap to hash and trivial to serialize.
    """

    op: str
    children: tuple[Node, ...] = ()
    params: tuple[int, ...] = ()

    probe: bool = field(default=False, compare=False, repr=False)
    """A sensitivity instrument rather than a candidate.

    A probe may carry a window off the admissible ladder, which the gauntlet's
    jitter test needs because the ladder is far too coarse to express plus or
    minus twenty per cent: the neighbours of 20 are 10 and 60.

    Excluded from equality and from the hash, because it says how the node is
    being used rather than what it computes. A probe is never a hypothesis.
    Nothing logs one as a trial and nothing counts one in N.
    """

    _hash: str = field(init=False, compare=False, repr=False)
    _warmup: int = field(init=False, compare=False, repr=False)
    _depth: int = field(init=False, compare=False, repr=False)
    _size: int = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        spec = grammar.get(self.op)
        grammar.validate_child_types(spec, tuple(c.dtype for c in self.children))
        grammar.validate_params(spec, self.params, allow_off_ladder=self.probe)

        children = self.children
        if spec.commutative:
            # Canonical order, by hashes that already exist because the
            # children were constructed before this node was.
            children = tuple(sorted(children, key=lambda node: node._hash))
            object.__setattr__(self, "children", children)

        object.__setattr__(self, "_hash", _compute_hash(self.op, self.params, children))
        object.__setattr__(
            self,
            "_warmup",
            grammar.warmup(
                spec,
                self.params,
                tuple(c._warmup for c in children),
                allow_off_ladder=self.probe,
            ),
        )
        object.__setattr__(
            self, "_depth", 1 + max((c._depth for c in children), default=0)
        )
        object.__setattr__(self, "_size", 1 + sum(c._size for c in children))

    # -- identity ----------------------------------------------------------

    def structural_hash(self) -> str:
        """Canonical hash of this subtree.

        Two trees share a hash exactly when they denote the same computation
        up to the ordering of commutative arguments. This is the evaluator's
        cache key and the ledger's ``expr_hash``.
        """
        return self._hash

    # -- derived properties ------------------------------------------------

    @property
    def spec(self) -> OpSpec:
        return grammar.get(self.op)

    @property
    def dtype(self) -> DType:
        return grammar.get(self.op).out_type

    @property
    def warmup(self) -> int:
        """Leading rows that are not valid data for this subtree.

        Accumulated at construction through ``grammar.warmup``, so every node
        in a tree carries its own count and the evaluator never has to
        recompute one.
        """
        return self._warmup

    @property
    def depth(self) -> int:
        """1 for a terminal."""
        return self._depth

    @property
    def size(self) -> int:
        """Total node count, terminals included."""
        return self._size

    @property
    def is_terminal(self) -> bool:
        return not self.children

    # -- traversal ---------------------------------------------------------

    def walk(self) -> Iterator[Node]:
        """Every node in the tree, parents before children."""
        yield self
        for child in self.children:
            yield from child.walk()

    def nodes(self) -> tuple[Node, ...]:
        """Every node, in a stable order, so an index can address one."""
        return tuple(self.walk())

    def subtrees_of_type(self, dtype: DType) -> tuple[int, ...]:
        """Positions whose result is ``dtype``.

        Crossover and mutation both need somewhere type-compatible to cut, and
        asking by type here is what keeps them from producing a tree that has
        to be thrown away.
        """
        return tuple(i for i, node in enumerate(self.nodes()) if node.dtype is dtype)

    def replace_at(self, index: int, replacement: Node) -> Node:
        """A copy of this tree with the node at ``index`` replaced.

        Returns a new tree. Nothing is mutated, so a tree handed to the
        evaluator, cached, and later used as a crossover parent cannot change
        underneath either of them.
        """
        total = self._size
        if not 0 <= index < total:
            raise IndexError(f"node {index} of a tree with {total} nodes")
        if index == 0:
            if replacement.dtype is not self.dtype:
                raise ValueError(
                    f"cannot replace a {self.dtype.value} with a "
                    f"{replacement.dtype.value}"
                )
            return replacement
        return _replace_at(self, index, replacement)

    def __str__(self) -> str:
        return to_string(self)


def _replace_at(node: Node, index: int, replacement: Node) -> Node:
    """Rebuild along the path to ``index``, sharing everything off it."""
    offset = 1
    rebuilt: list[Node] = []
    for child in node.children:
        if offset <= index < offset + child._size:
            rebuilt.append(_replace_at(child, index - offset, replacement))
        else:
            rebuilt.append(child)
        offset += child._size
    if index == 0:
        return replacement
    return Node(node.op, tuple(rebuilt), node.params, node.probe)


def _compute_hash(op: str, params: tuple[int, ...], children: tuple[Node, ...]) -> str:
    """Hash a node from hashes its children already carry.

    The children are hashed first because they were constructed first. Sorting
    happens on those hashes, in the caller, before this is reached: by the
    time a parent is hashed, the order of its arguments is already settled.

    Sorting hex digests of equal length is sorting by hash bytes, since the
    lexicographic order of fixed-width lowercase hex matches the byte order it
    encodes.
    """
    payload = "|".join(
        (
            op,
            ",".join(str(p) for p in params),
            *(child._hash for child in children),
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:_HASH_CHARS]


def strip_elidable_root(tree: Node) -> tuple[Node, int]:
    """Drop root operators a rank-based screen cannot distinguish.

    Returns the tree to evaluate and how many nodes came off.

    ``rank`` at the root is a no-op for everything the screen measures. Rank
    IC, decile membership and turnover are all invariant under a strictly
    monotone per-day transform, and rank preserves the NaN pattern exactly, so
    ``rank(x)`` and ``x`` produce identical numbers everywhere it matters. On
    a full panel that no-op costs the better part of a second, which is the
    most expensive nothing in the system.

    Only the root chain is stripped. A rank in the middle of a tree is not
    elidable: ``ts_mean(rank(x), 20)`` averages ranks, which is a different
    quantity from the rank of an average, and nothing about the screen makes
    those interchangeable.

    The original tree is still what gets logged. This changes what is
    computed, never what was hypothesised.
    """
    node = tree
    stripped = 0
    while not node.is_terminal and node.spec.elidable_at_root:
        node = node.children[0]
        stripped += 1
    return node, stripped


def field_node(name: str) -> Node:
    """A terminal, by field name. Shorthand for hand-written trees."""
    return Node(name)


# --------------------------------------------------------------------------
# strings
# --------------------------------------------------------------------------
#
# The string form exists so a human can read an expression and the ledger can
# store one. It is not the canonical representation, and nothing in the engine
# makes a decision by inspecting it.
#
# It round-trips: from_string(to_string(tree)) is tree, for every tree. That
# is worth a test rather than a comment, because the ledger keeps strings and
# a pool entry nobody can turn back into a tree is a result nobody can reuse.


def to_string(node: Node) -> str:
    """Render a tree as ``op(child, child, param)``.

    Terminals render bare, without empty parentheses, because ``close`` is how
    anyone would write it and ``close()`` is how nobody would.
    """
    if node.is_terminal and not node.params:
        return node.op
    arguments = [to_string(child) for child in node.children]
    arguments.extend(str(p) for p in node.params)
    return f"{node.op}({', '.join(arguments)})"


class ParseError(ValueError):
    """A string is not a well-formed expression."""


def from_string(text: str) -> Node:
    """Parse a rendered expression back into a tree.

    Whitespace is insignificant, and a terminal written with empty parentheses
    (``close()``) is accepted and normalized to the bare form. Parsing is
    lenient about how an expression was written and strict about what it
    means: the result is validated by ``Node`` construction like any other
    tree, so a string naming an unknown operator or an inadmissible window
    fails here rather than somewhere downstream holding a tree that should not
    exist.
    """
    tokens = _tokenize(text)
    node, rest = _parse(tokens, text)
    if rest:
        raise ParseError(f"trailing input after a complete expression: {text!r}")
    return node


_PUNCTUATION: Final[frozenset[str]] = frozenset("(),")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for character in text:
        if character in _PUNCTUATION:
            if current.strip():
                tokens.append(current.strip())
            current = ""
            tokens.append(character)
        elif character.isspace():
            if current.strip():
                tokens.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        tokens.append(current.strip())
    return tokens


def _parse(tokens: list[str], source: str) -> tuple[Node, list[str]]:
    if not tokens:
        raise ParseError(f"expected an expression in {source!r}")

    name, *rest = tokens
    if name in _PUNCTUATION:
        raise ParseError(f"expected an operator name, found {name!r} in {source!r}")

    spec = _lookup(name, source)
    if not rest or rest[0] != "(":
        if spec.arity or spec.params:
            raise ParseError(
                f"{name} takes arguments but was written bare in {source!r}"
            )
        return Node(name), rest

    rest = rest[1:]
    children: list[Node] = []
    params: list[int] = []
    while True:
        if not rest:
            raise ParseError(f"unclosed parenthesis in {source!r}")
        if rest[0] == ")":
            rest = rest[1:]
            break
        if rest[0].lstrip("-").isdigit():
            params.append(int(rest[0]))
            rest = rest[1:]
        else:
            child, rest = _parse(rest, source)
            children.append(child)
        if rest and rest[0] == ",":
            rest = rest[1:]

    return Node(name, tuple(children), tuple(params)), rest


def _lookup(name: str, source: str) -> OpSpec:
    try:
        return grammar.get(name)
    except KeyError as exc:
        raise ParseError(f"{exc.args[0]} (in {source!r})") from None


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
#
# Type-directed throughout: to fill a hole of type T, draw from the operators
# that produce T. Trees come out well typed by construction, so there is no
# generate-and-reject loop, and the GROUP argument of a group op can only ever
# be resolved by the sector terminal because nothing else produces one.
#
# Every function here takes an explicit numpy Generator. There is no module
# level default and no implicit seeding, so a search that found something
# cannot later turn out to have depended on global state nobody recorded.


def random_tree(
    rng: np.random.Generator,
    max_depth: int = 4,
    *,
    dtype: DType = DType.MATRIX,
) -> Node:
    """A random well-typed tree of at most ``max_depth`` levels.

    The root is always an operator, never a bare field. A tree consisting of
    ``close`` is not a hypothesis, and admitting one would spend a trial, and
    a slot in N, on the observation that prices exist.
    """
    if max_depth < 2:
        raise ValueError("a tree needs at least two levels to say anything")
    return _grow(rng, max_depth, dtype, force_internal=True)


def _grow(
    rng: np.random.Generator,
    budget: int,
    dtype: DType,
    *,
    force_internal: bool = False,
) -> Node:
    """Grow a subtree of the required type within a depth budget."""
    if budget <= 1 and not force_internal:
        return _draw_terminal(rng, dtype)

    pool = grammar.internal_producing(dtype)
    if not force_internal:
        pool = pool + grammar.terminals_producing(dtype)
    if not pool:
        return _draw_terminal(rng, dtype)

    spec = _draw(rng, pool)
    if spec.is_terminal:
        return Node(spec.name)

    children = tuple(_grow(rng, budget - 1, child_type) for child_type in spec.in_types)
    return Node(spec.name, children, _draw_params(rng, spec))


def _draw(rng: np.random.Generator, pool: tuple[OpSpec, ...]) -> OpSpec:
    """Weighted choice among operators.

    Weights shift how often an operator is drawn and say nothing about what it
    means, which is why they are excluded from the grammar fingerprint: two
    runs with different weights explore differently but agree on what any
    given expression denotes.
    """
    weights = np.array([spec.weight for spec in pool], dtype=np.float64)
    return pool[int(rng.choice(len(pool), p=weights / weights.sum()))]


def _draw_terminal(rng: np.random.Generator, dtype: DType) -> Node:
    pool = grammar.terminals_producing(dtype)
    if not pool:
        raise ValueError(f"no terminal produces {dtype.value}")
    return Node(_draw(rng, pool).name)


def _draw_params(rng: np.random.Generator, spec: OpSpec) -> tuple[int, ...]:
    return tuple(
        int(param.values[rng.integers(len(param.values))]) for param in spec.params
    )


# --------------------------------------------------------------------------
# variation
# --------------------------------------------------------------------------


def mutate(node: Node, rng: np.random.Generator, *, max_depth: int = 4) -> Node:
    """One random change, returning a new tree.

    Three kinds, chosen among whichever apply at the chosen site:

    ``param``     move a window one rung along its ladder. The smallest step,
                  and the one most likely to find a better version of an idea
                  that already works.
    ``operator``  swap an operator for another with the same signature, which
                  keeps the whole subtree and changes only what is done to it.
    ``subtree``   replace the site with a fresh random subtree of the same
                  type. The largest step, and the only one that can introduce
                  an operator the tree does not already contain.

    Keeping all three matters because a mutation that can only make small
    moves cannot escape a local optimum, and one that can only make large
    moves never refines anything.

    The result is no deeper than ``max(node.depth, max_depth)``: the
    replacement subtree is grown within whatever depth remains below its site,
    down to a terminal if nothing remains, but a tree that already exceeded
    the bound is not pruned by mutating it somewhere else.
    """
    positions = node.nodes()
    index = int(rng.integers(len(positions)))
    site = positions[index]

    kinds: list[str] = ["subtree"]
    if site.params:
        kinds.append("param")
    if _alternatives(site):
        kinds.append("operator")

    kind = kinds[int(rng.integers(len(kinds)))]
    if kind == "param":
        replacement = Node(site.op, site.children, _mutate_params(site, rng))
    elif kind == "operator":
        chosen = _draw(rng, _alternatives(site))
        replacement = Node(chosen.name, site.children, _draw_params(rng, chosen))
    else:
        # The budget is what is left below the site, and it is allowed to fall
        # to a terminal. Flooring it at two levels, so that a subtree mutation
        # always produced a subtree, is what let a mutation at depth 4 of a
        # depth-4 tree return a depth-5 one.
        budget = max_depth - _depth_of(node, index) + 1
        replacement = _grow(rng, budget, site.dtype)

    return node.replace_at(index, replacement)


def _mutate_params(node: Node, rng: np.random.Generator) -> tuple[int, ...]:
    """Step one window to an adjacent rung of its own ladder.

    Adjacent rather than uniform, because the ladder is log spaced: a uniform
    redraw from 250 lands on 2 as often as on 120, which is a new hypothesis
    rather than a refinement of the existing one.
    """
    spec = node.spec
    which = int(rng.integers(len(spec.params)))
    values = spec.params[which].values
    current = values.index(node.params[which])
    if current == 0:
        step = 1
    elif current == len(values) - 1:
        step = -1
    else:
        step = 1 if rng.random() < 0.5 else -1
    params = list(node.params)
    params[which] = values[current + step]
    return tuple(params)


def _alternatives(node: Node) -> tuple[OpSpec, ...]:
    """Operators that could stand in for this one without retyping the tree."""
    spec = node.spec
    return tuple(
        other
        for other in grammar.OPS.values()
        if other.name != spec.name
        and other.in_types == spec.in_types
        and other.out_type is spec.out_type
        and len(other.params) == len(spec.params)
    )


def _depth_of(root: Node, index: int) -> int:
    """Levels from the root down to ``index``, counting the root as 1.

    Walks in the same parents-before-children order as ``nodes()``, so an
    index means the same thing to both.
    """
    position = 0

    def visit(node: Node, level: int) -> int | None:
        nonlocal position
        if position == index:
            return level
        position += 1
        for child in node.children:
            found = visit(child, level + 1)
            if found is not None:
                return found
        return None

    found = visit(root, 1)
    if found is None:
        raise IndexError(index)
    return found


def crossover(
    first: Node, second: Node, rng: np.random.Generator, *, max_depth: int = 6
) -> tuple[Node, Node]:
    """Swap a type-compatible subtree between two trees.

    Returns both offspring. Discarding one halves the information obtained
    from a pair of evaluations that have already been paid for.

    The cut site is chosen by type, so the swap cannot produce an ill-typed
    tree and there is no rejection loop. If a swap would exceed ``max_depth``,
    the parents come back unchanged rather than a truncated tree: trimming
    silently would hand back an expression that is not the crossover of
    anything, and it would still be counted as a trial.
    """
    shared = _shared_types(first, second)
    if not shared:
        return first, second

    dtype = shared[int(rng.integers(len(shared)))]
    first_positions = first.subtrees_of_type(dtype)
    second_positions = second.subtrees_of_type(dtype)
    i = int(first_positions[int(rng.integers(len(first_positions)))])
    j = int(second_positions[int(rng.integers(len(second_positions)))])

    graft_into_first = first.replace_at(i, second.nodes()[j])
    graft_into_second = second.replace_at(j, first.nodes()[i])

    if graft_into_first.depth > max_depth or graft_into_second.depth > max_depth:
        return first, second
    return graft_into_first, graft_into_second


def _shared_types(first: Node, second: Node) -> tuple[DType, ...]:
    """Types present in both trees, so a swap has somewhere to land."""
    return tuple(
        dtype
        for dtype in DType
        if first.subtrees_of_type(dtype) and second.subtrees_of_type(dtype)
    )
