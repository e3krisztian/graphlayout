import itertools
import random

import numpy as np
import pytest

from bubble_tree import bfs, bubble_tree_locations, center, neighbour_lists
from graphs import (
    cube_stack, eiffel_tower_front, hanoi, petersen, pipe, randomg, star, tree)
from layout import EDGE_LENGTH


def path_edges(n):
    return [[node + step for step in (-1, 1) if 0 <= node + step < n] for node in range(n)]


def tree_edges(edges):
    # the edges of the BFS spanning trees bubble_tree_locations draws
    neighbours = neighbour_lists(edges)
    seen = set()
    result = []
    for node in range(len(edges)):
        if node not in seen:
            order, parents = bfs(neighbours, center(neighbours, node))
            seen.update(order)
            result += [(parent, child) for child, parent in parents.items() if parent is not None]
    return result


def crosses(p1, p2, q1, q2):
    # the segments p1-p2 and q1-q2 cross at a point inside both
    def side(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return (side(p1, p2, q1) * side(p1, p2, q2) < 0
            and side(q1, q2, p1) * side(q1, q2, p2) < 0)


def test_center_is_the_middle_of_a_path():
    assert center(neighbour_lists(path_edges(5)), 0) == 2


def test_a_path_is_drawn_straight_with_edges_of_edge_length():
    locations = bubble_tree_locations(path_edges(5))
    lengths = np.linalg.norm(np.diff(locations, axis=0), axis=1)
    assert lengths == pytest.approx([EDGE_LENGTH] * 4)
    assert np.linalg.norm(locations[4] - locations[0]) == pytest.approx(4 * EDGE_LENGTH)


@pytest.mark.parametrize('nodecount', [0, 1])
def test_tiny_graphs(nodecount):
    assert bubble_tree_locations([[] for _ in range(nodecount)]).shape == (nodecount, 2)


def graphs_to_draw():
    random.seed(1)
    return [
        star(30), tree(100), petersen(), hanoi(3), pipe(10, 10), cube_stack(2, 3, 4),
        eiffel_tower_front(), randomg(40, 30), randomg(60, 200)]


@pytest.mark.parametrize('graph', graphs_to_draw())
def test_no_two_nodes_are_closer_than_edge_length(graph):
    locations = bubble_tree_locations(graph.edges)
    assert np.isfinite(locations).all()
    differences = locations[:, None, :] - locations[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= EDGE_LENGTH * (1 - 1e-6)


@pytest.mark.parametrize('graph', graphs_to_draw())
def test_no_two_spanning_tree_edges_cross(graph):
    locations = bubble_tree_locations(graph.edges)
    edges = tree_edges(graph.edges)
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        if len({a, b, c, d}) == 4:
            assert not crosses(locations[a], locations[b], locations[c], locations[d])


def test_spanning_tree_edges_are_at_least_edge_length():
    random.seed(1)
    graph = tree(100)
    locations = bubble_tree_locations(graph.edges)
    lengths = [np.linalg.norm(locations[a] - locations[b]) for a, b in tree_edges(graph.edges)]
    assert min(lengths) >= EDGE_LENGTH * (1 - 1e-9)
