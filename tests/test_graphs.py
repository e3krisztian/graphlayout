import itertools

import pytest

from graphs import (
    cube, dodecahedron, hanoi, heawood, icosahedron, moser_spindle, octahedron, petersen, pipe,
    tetrahedron)


def degrees(graph):
    return sorted(len(neighbours) for neighbours in graph.edges)


def edge_count(graph):
    return sum(len(neighbours) for neighbours in graph.edges) // 2


def is_simple(graph):
    "no self-loops, no duplicate edges"
    return all(
        node not in neighbours and len(set(neighbours)) == len(neighbours)
        for node, neighbours in enumerate(graph.edges))


def is_connected(graph):
    seen = {0}
    todo = [0]
    while todo:
        for neighbour in graph.edges[todo.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                todo.append(neighbour)
    return len(seen) == graph.nodecount


def test_open_pipe_has_a_node_for_each_position_on_each_circle():
    assert pipe(5, 4).nodecount == 20


def test_open_pipe_end_circles_have_one_neighbour_circle():
    # 5 nodes on each of the 2 end circles: 2 on the circle + 1 link
    # 10 nodes on the 2 inner circles: 2 on the circle + 2 links
    assert degrees(pipe(5, 4)) == [3] * 10 + [4] * 10


def test_single_circle_is_a_ring():
    ring = pipe(6, 1)
    assert degrees(ring) == [2] * 6
    assert is_connected(ring)


def test_pipe_is_connected():
    assert is_connected(pipe(5, 4))


def test_closed_pipe_links_the_last_circle_to_the_first():
    # a torus: every node has 2 neighbours on its circle and 1 on each neighbour circle
    torus = pipe(5, 4, closed=True)
    assert torus.nodecount == 20
    assert degrees(torus) == [4] * 20
    assert is_simple(torus)
    assert is_connected(torus)


def test_closed_pipe_adds_one_link_per_node_of_a_circle():
    assert edge_count(pipe(5, 4, closed=True)) == edge_count(pipe(5, 4)) + 5


def test_two_node_circle_is_a_single_edge():
    # 3 circle edges + 2 links between each of the 2 neighbouring circle pairs
    ladder = pipe(2, 3)
    assert is_simple(ladder)
    assert edge_count(ladder) == 7


def test_one_node_circles_form_a_path():
    path = pipe(1, 4)
    assert is_simple(path)
    assert degrees(path) == [1, 1, 2, 2]


def test_closing_two_circles_adds_no_duplicate_links():
    # the two circles are already linked: 3 + 3 circle edges, 3 links
    graph = pipe(3, 2, closed=True)
    assert is_simple(graph)
    assert edge_count(graph) == 9


def test_closing_a_single_circle_adds_no_self_loops():
    ring = pipe(3, 1, closed=True)
    assert is_simple(ring)
    assert degrees(ring) == [2, 2, 2]


@pytest.mark.parametrize("solid, nodecount, edgecount, degree", [
    (tetrahedron, 4, 6, 3),
    (cube, 8, 12, 3),
    (octahedron, 6, 12, 4),
    (dodecahedron, 20, 30, 3),
    (icosahedron, 12, 30, 5),
])
def test_platonic_solid(solid, nodecount, edgecount, degree):
    graph = solid()
    assert graph.nodecount == nodecount
    assert edge_count(graph) == edgecount
    assert degrees(graph) == [degree] * nodecount
    assert is_simple(graph)
    assert is_connected(graph)


def test_petersen_graph():
    graph = petersen()
    assert graph.nodecount == 10
    assert edge_count(graph) == 15
    assert degrees(graph) == [3] * 10
    assert is_simple(graph)
    assert is_connected(graph)


def test_petersen_graph_pairs_have_one_common_neighbour_unless_linked():
    # what sets it apart from the other 3-regular graphs on 10 nodes, e.g. the pentagonal prism
    graph = petersen()
    for node1 in range(10):
        for node2 in range(node1 + 1, 10):
            common = set(graph.edges[node1]) & set(graph.edges[node2])
            assert len(common) == (0 if node2 in graph.edges[node1] else 1)


def test_heawood_graph():
    graph = heawood()
    assert graph.nodecount == 14
    assert edge_count(graph) == 21
    assert degrees(graph) == [3] * 14
    assert is_simple(graph)
    assert is_connected(graph)


def test_heawood_graph_is_the_fano_plane():
    # even nodes are points, odd nodes are lines: every edge joins a point and a line,
    # and any two points are on exactly one common line, any two lines meet in exactly one point
    graph = heawood()
    for node, neighbours in enumerate(graph.edges):
        assert all((node - neighbour) % 2 == 1 for neighbour in neighbours)
    for node1 in range(14):
        for node2 in range(node1 + 2, 14, 2):
            assert len(set(graph.edges[node1]) & set(graph.edges[node2])) == 1


def distance(graph, node1, node2):
    distances = {node1: 0}
    todo = [node1]
    for node in todo:
        for neighbour in graph.edges[node]:
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                todo.append(neighbour)
    return distances[node2]


@pytest.mark.parametrize("disks", [1, 2, 3, 4])
def test_hanoi_graph(disks):
    # the 3 states with all disks on a single peg have 2 moves, all the others 3
    graph = hanoi(disks)
    assert graph.nodecount == 3 ** disks
    assert degrees(graph) == [2] * 3 + [3] * (3 ** disks - 3)
    assert is_simple(graph)
    assert is_connected(graph)


@pytest.mark.parametrize("disks", [1, 2, 3, 4])
def test_hanoi_graph_moves_the_tower_in_the_known_number_of_steps(disks):
    all_on_first_peg, all_on_last_peg = 0, 3 ** disks - 1
    assert distance(hanoi(disks), all_on_first_peg, all_on_last_peg) == 2 ** disks - 1


def test_moser_spindle():
    graph = moser_spindle()
    assert graph.nodecount == 7
    assert edge_count(graph) == 11
    assert degrees(graph) == [3] * 6 + [4]
    assert is_simple(graph)
    assert is_connected(graph)


def test_moser_spindle_needs_4_colours():
    graph = moser_spindle()

    def proper(colours):
        return all(
            colours[node] != colours[neighbour]
            for node, neighbours in enumerate(graph.edges) for neighbour in neighbours)

    assert not any(proper(colours) for colours in itertools.product(range(3), repeat=7))
    assert any(proper(colours) for colours in itertools.product(range(4), repeat=7))
