"""Tests for expression trees.

Three properties carry weight here.

Trees are canonical: two spellings of one idea are one object, so the search
cannot spend two trials on it and inflate the N behind the Deflated Sharpe.

Trees are valid by construction: an ill-typed or out-of-range tree cannot be
built, so the ledger never records a candidate that was never a hypothesis.

Warmup is carried, not recomputed: every node knows its own count, and the
counts match the ones hand-computed in the grammar tests.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from alpha.expr import grammar
from alpha.expr.ast import (
    Node,
    ParseError,
    crossover,
    from_string,
    mutate,
    random_tree,
    to_string,
)
from alpha.expr.grammar import DType

CLOSE = Node("close")
MOMENTUM = Node(
    "div",
    (
        Node("ts_delay", (CLOSE,), (20,)),
        Node("ts_delay", (CLOSE,), (250,)),
    ),
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(20260917)


# --------------------------------------------------------------------------
# construction and validity
# --------------------------------------------------------------------------


def test_a_terminal_is_a_tree() -> None:
    assert CLOSE.depth == 1
    assert CLOSE.size == 1
    assert CLOSE.is_terminal
    assert CLOSE.dtype is DType.MATRIX


def test_depth_and_size_are_carried() -> None:
    assert MOMENTUM.depth == 3
    assert MOMENTUM.size == 5


def test_an_ill_typed_tree_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="expects"):
        Node("rank_within", (CLOSE, CLOSE))


def test_a_group_terminal_cannot_stand_where_a_matrix_belongs() -> None:
    with pytest.raises(ValueError, match="expects"):
        Node("rank", (Node("sector"),))


def test_a_window_off_the_ladder_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="admissible ladder"):
        Node("ts_mean", (CLOSE,), (1,))


def test_a_missing_window_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="parameters"):
        Node("ts_mean", (CLOSE,))


def test_wrong_arity_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="expects"):
        Node("rank", (CLOSE, CLOSE))


def test_a_node_is_immutable() -> None:
    with pytest.raises(Exception):
        MOMENTUM.op = "sub"  # type: ignore[misc]


def test_nodes_are_hashable_and_usable_as_keys() -> None:
    """The evaluator's subtree cache depends on this."""
    cache = {MOMENTUM: 1, CLOSE: 2}
    assert cache[Node("close")] == 2


# --------------------------------------------------------------------------
# canonical form
# --------------------------------------------------------------------------


def test_commutative_arguments_canonicalise_to_one_tree() -> None:
    """Two spellings of one idea must not cost two trials."""
    first = Node("add", (Node("close"), Node("open")))
    second = Node("add", (Node("open"), Node("close")))
    assert first == second
    assert first.structural_hash() == second.structural_hash()
    assert to_string(first) == to_string(second)


def test_correlation_canonicalises_too() -> None:
    first = Node("correlation", (Node("close"), Node("volume")), (60,))
    second = Node("correlation", (Node("volume"), Node("close")), (60,))
    assert first.structural_hash() == second.structural_hash()


def test_a_non_commutative_operator_keeps_its_argument_order() -> None:
    first = Node("sub", (Node("close"), Node("open")))
    second = Node("sub", (Node("open"), Node("close")))
    assert first != second
    assert first.structural_hash() != second.structural_hash()


def test_canonicalisation_reaches_inside_nested_trees() -> None:
    first = Node("rank", (Node("mul", (Node("close"), Node("volume"))),))
    second = Node("rank", (Node("mul", (Node("volume"), Node("close"))),))
    assert first.structural_hash() == second.structural_hash()


def test_different_windows_are_different_trees() -> None:
    a = Node("ts_mean", (CLOSE,), (20,))
    b = Node("ts_mean", (CLOSE,), (60,))
    assert a.structural_hash() != b.structural_hash()


def test_different_operators_over_the_same_child_differ() -> None:
    a = Node("ts_mean", (CLOSE,), (20,))
    b = Node("ts_std", (CLOSE,), (20,))
    assert a.structural_hash() != b.structural_hash()


def test_the_hash_is_a_128_bit_hex_digest() -> None:
    """The ledger keys trials on it, so a collision merges two hypotheses."""
    digest = MOMENTUM.structural_hash()
    assert len(digest) == 32
    assert all(c in "0123456789abcdef" for c in digest)


def test_identical_subtrees_share_one_hash() -> None:
    """Which is what makes the evaluator's subtree cache pay for itself."""
    hashes = [node.structural_hash() for node in MOMENTUM.walk() if node.op == "close"]
    assert len(set(hashes)) == 1


# --------------------------------------------------------------------------
# warmup
# --------------------------------------------------------------------------


def test_momentum_warmup_matches_the_hand_computed_value() -> None:
    """The same 250 the grammar tests derive longhand."""
    assert MOMENTUM.warmup == 250


def test_a_terminal_has_no_warmup() -> None:
    assert CLOSE.warmup == 0


def test_warmup_accumulates_through_a_chain() -> None:
    #   decay_linear(ts_rank(delta(close, 250), 20), 5)
    #     delta(., 250)      =   0 + 250     = 250
    #     ts_rank(., 20)     = 250 + 20 - 1  = 269
    #     decay_linear(., 5) = 269 +  5 - 1  = 273
    tree = Node(
        "decay_linear",
        (Node("ts_rank", (Node("delta", (CLOSE,), (250,)),), (20,)),),
        (5,),
    )
    assert tree.warmup == 273


def test_a_parent_is_never_valid_before_its_children() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        tree = random_tree(rng, 4)
        for node in tree.walk():
            for child in node.children:
                assert node.warmup >= child.warmup


# --------------------------------------------------------------------------
# traversal and replacement
# --------------------------------------------------------------------------


def test_walk_visits_every_node_once() -> None:
    assert len(MOMENTUM.nodes()) == MOMENTUM.size


def test_subtrees_of_type_finds_the_group_terminal() -> None:
    tree = Node("demean_by", (CLOSE, Node("sector")))
    assert tree.subtrees_of_type(DType.GROUP) == (2,)
    assert tree.subtrees_of_type(DType.MATRIX) == (0, 1)


def test_replace_at_leaves_the_original_alone() -> None:
    tree = Node("rank", (Node("ts_mean", (CLOSE,), (20,)),))
    replaced = tree.replace_at(2, Node("volume"))
    assert [n.op for n in replaced.nodes()] == ["rank", "ts_mean", "volume"]
    assert [n.op for n in tree.nodes()] == ["rank", "ts_mean", "close"]


def test_replace_at_the_root_returns_the_replacement() -> None:
    assert MOMENTUM.replace_at(0, CLOSE) == CLOSE


def test_replace_at_refuses_a_type_change_at_the_root() -> None:
    with pytest.raises(ValueError, match="cannot replace"):
        MOMENTUM.replace_at(0, Node("sector"))


def test_replace_at_refuses_an_index_off_the_end() -> None:
    with pytest.raises(IndexError):
        MOMENTUM.replace_at(99, CLOSE)


def test_replacement_updates_warmup() -> None:
    """Because warmup is carried, a graft has to rebuild it up the path."""
    tree = Node("rank", (Node("ts_mean", (CLOSE,), (20,)),))
    assert tree.warmup == 19
    swapped = tree.replace_at(1, Node("ts_mean", (CLOSE,), (250,)))
    assert swapped.warmup == 249


# --------------------------------------------------------------------------
# strings
# --------------------------------------------------------------------------


def test_rendering_reads_the_way_it_is_written() -> None:
    assert to_string(MOMENTUM) == "div(ts_delay(close, 20), ts_delay(close, 250))"


def test_a_terminal_renders_bare() -> None:
    assert to_string(CLOSE) == "close"


def test_a_group_argument_renders_by_name() -> None:
    tree = Node("rank_within", (CLOSE, Node("sector")))
    assert to_string(tree) == "rank_within(close, sector)"


@pytest.mark.parametrize(
    "text",
    [
        "close",
        "rank(close)",
        "ts_mean(close, 20)",
        "div(ts_delay(close, 20), ts_delay(close, 250))",
        "correlation(close, volume, 60)",
        "demean_by(rank(returns), sector)",
        "decay_linear(rank_within(ts_rank(volume, 20), sector), 5)",
    ],
)
def test_parsing_round_trips(text: str) -> None:
    tree = from_string(text)
    assert from_string(to_string(tree)) == tree


def test_whitespace_is_insignificant() -> None:
    assert from_string("  ts_mean( close ,  20 )  ") == Node("ts_mean", (CLOSE,), (20,))


def test_a_terminal_with_empty_parentheses_is_accepted() -> None:
    """Lenient about how it was written, strict about what it means."""
    assert from_string("close()") == CLOSE


def test_parsing_a_commutative_node_yields_canonical_order() -> None:
    """The round trip preserves the tree, not the spelling."""
    assert from_string("add(volume, close)") == from_string("add(close, volume)")


@pytest.mark.parametrize(
    "text",
    ["", "ts_meen(close, 20)", "rank(close", "rank(close))", "(close)", "rank()"],
)
def test_malformed_strings_are_refused(text: str) -> None:
    with pytest.raises((ParseError, ValueError)):
        from_string(text)


def test_an_inadmissible_window_fails_at_the_parse() -> None:
    with pytest.raises(ValueError, match="admissible ladder"):
        from_string("ts_mean(close, 1)")


def test_every_random_tree_round_trips(rng: np.random.Generator) -> None:
    """The ledger stores strings; one nobody can parse back is one nobody can use."""
    for _ in range(300):
        tree = random_tree(rng, 4)
        assert from_string(to_string(tree)) == tree


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def test_generated_trees_respect_the_depth_bound(rng: np.random.Generator) -> None:
    for _ in range(300):
        assert random_tree(rng, 4).depth <= 4


def test_the_root_is_never_a_bare_field(rng: np.random.Generator) -> None:
    """A tree that is just close is not a hypothesis."""
    for _ in range(200):
        assert not random_tree(rng, 4).is_terminal


def test_a_one_level_tree_is_refused(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="at least two levels"):
        random_tree(rng, 1)


def test_generation_is_reproducible_from_its_seed() -> None:
    first = [random_tree(np.random.default_rng(11), 4) for _ in range(30)]
    second = [random_tree(np.random.default_rng(11), 4) for _ in range(30)]
    assert [t.structural_hash() for t in first] == [t.structural_hash() for t in second]


def test_a_different_seed_explores_differently() -> None:
    first = {random_tree(np.random.default_rng(11), 4).structural_hash() for _ in range(30)}
    second = {random_tree(np.random.default_rng(12), 4).structural_hash() for _ in range(30)}
    assert first != second


def test_generation_reaches_every_operator(rng: np.random.Generator) -> None:
    """A weight that starves an operator removes it from the search silently."""
    seen = Counter(
        node.op for _ in range(800) for node in random_tree(rng, 4).walk()
    )
    assert set(seen) == set(grammar.OPS)


def test_generated_trees_are_mostly_distinct(rng: np.random.Generator) -> None:
    trees = [random_tree(rng, 4) for _ in range(400)]
    assert len({t.structural_hash() for t in trees}) > 350


def test_a_group_argument_is_always_the_sector_terminal(rng: np.random.Generator) -> None:
    for _ in range(300):
        for node in random_tree(rng, 4).walk():
            if node.dtype is DType.GROUP:
                assert node.op == "sector"
                assert node.is_terminal


# --------------------------------------------------------------------------
# mutation
# --------------------------------------------------------------------------


def test_mutation_respects_the_depth_bound(rng: np.random.Generator) -> None:
    for _ in range(300):
        tree = random_tree(rng, 4)
        assert mutate(tree, rng, max_depth=4).depth <= 4


def test_mutation_leaves_the_parent_alone(rng: np.random.Generator) -> None:
    tree = random_tree(rng, 4)
    before = tree.structural_hash()
    mutate(tree, rng)
    assert tree.structural_hash() == before


def test_mutation_usually_changes_something(rng: np.random.Generator) -> None:
    """It can redraw the same value; it should not usually."""
    trees = [random_tree(rng, 4) for _ in range(300)]
    changed = sum(mutate(t, rng) != t for t in trees)
    assert changed > 250


def test_mutants_are_valid_trees(rng: np.random.Generator) -> None:
    for _ in range(300):
        mutant = mutate(random_tree(rng, 4), rng)
        assert from_string(to_string(mutant)) == mutant


def test_a_window_mutation_steps_to_an_adjacent_rung(rng: np.random.Generator) -> None:
    """A uniform redraw from 250 would land on 2 as often as on 120."""
    tree = Node("ts_mean", (CLOSE,), (20,))
    ladder = grammar.get("ts_mean").params[0].values
    position = ladder.index(20)
    for _ in range(50):
        mutant = mutate(tree, rng, max_depth=2)
        if mutant.op == "ts_mean" and mutant.children == tree.children and mutant.params != tree.params:
            assert abs(ladder.index(mutant.params[0]) - position) == 1


# --------------------------------------------------------------------------
# crossover
# --------------------------------------------------------------------------


def test_crossover_returns_two_offspring(rng: np.random.Generator) -> None:
    first, second = crossover(random_tree(rng, 4), random_tree(rng, 4), rng)
    assert isinstance(first, Node) and isinstance(second, Node)


def test_crossover_respects_the_depth_bound(rng: np.random.Generator) -> None:
    for _ in range(300):
        a, b = crossover(random_tree(rng, 4), random_tree(rng, 4), rng, max_depth=6)
        assert a.depth <= 6 and b.depth <= 6


def test_crossover_offspring_are_valid_trees(rng: np.random.Generator) -> None:
    for _ in range(200):
        a, b = crossover(random_tree(rng, 4), random_tree(rng, 4), rng)
        assert from_string(to_string(a)) == a
        assert from_string(to_string(b)) == b


def test_crossover_leaves_its_parents_alone(rng: np.random.Generator) -> None:
    first, second = random_tree(rng, 4), random_tree(rng, 4)
    before = (first.structural_hash(), second.structural_hash())
    crossover(first, second, rng)
    assert (first.structural_hash(), second.structural_hash()) == before


def test_crossover_with_itself_stays_well_formed(rng: np.random.Generator) -> None:
    tree = random_tree(rng, 4)
    a, b = crossover(tree, tree, rng)
    assert from_string(to_string(a)) == a
    assert from_string(to_string(b)) == b


def test_a_breaching_swap_returns_the_parents_unchanged(rng: np.random.Generator) -> None:
    """Silently trimming would return an expression that is not a crossover."""
    deep = random_tree(np.random.default_rng(2), 6)
    other = random_tree(np.random.default_rng(3), 6)
    a, b = crossover(deep, other, rng, max_depth=2)
    assert (a, b) == (deep, other)
