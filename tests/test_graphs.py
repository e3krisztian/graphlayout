import pytest

from graphs import cube, dodecahedron, icosahedron, octahedron, petersen, pipe, tetrahedron


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
