import numpy as np

from layout import EDGE_LENGTH, GraphLayout, improveall, randomized_layout, toggle_pin


def two_node_layout(distance):
    return GraphLayout([[1], [0]], [[0, 0], [distance, 0]])


def test_edge_of_ideal_length_does_not_attract():
    layout = two_node_layout(EDGE_LENGTH)
    attraction = layout.attraction(np.array([[EDGE_LENGTH, 0.]]))
    assert attraction.tolist() == [0, 0]


def test_longer_edge_pulls_towards_the_neighbour():
    layout = two_node_layout(EDGE_LENGTH * 3)
    attraction = layout.attraction(np.array([[EDGE_LENGTH * 3, 0.]]))
    assert attraction[0] > 0
    assert attraction[1] == 0


def path_layout(pinned=None):
    # path 0-1-2 with stretched edges, all nodes would move
    return GraphLayout([[1], [0, 2], [1]], [[0, 0], [10, 0], [20, 5]], pinned)


def test_nodes_are_not_pinned_by_default():
    assert path_layout().pinned.tolist() == [False, False, False]


def test_pinned_node_has_no_delta():
    layout = path_layout([True, False, False])
    assert layout.delta[0].tolist() == [0, 0]
    assert np.linalg.norm(layout.delta[1]) > 0


def test_step_does_not_move_pinned_nodes():
    layout = path_layout([True, False, True]).step(0.5)
    assert layout.locations[0].tolist() == [0, 0]
    assert layout.locations[2].tolist() == [20, 5]
    assert layout.pinned.tolist() == [True, False, True]


def test_improveall_does_not_move_pinned_nodes():
    layout = improveall(path_layout([False, True, False]), temperature=1.0)
    assert layout.locations[1].tolist() == [10, 0]
    assert layout.pinned.tolist() == [False, True, False]


def test_pinned_at_moves_and_pins_the_node():
    layout = path_layout().pinned_at(1, [3, 4])
    assert layout.locations[1].tolist() == [3, 4]
    assert layout.pinned.tolist() == [False, True, False]
    assert layout.delta[1].tolist() == [0, 0]


def test_pinned_at_leaves_the_original_layout_unchanged():
    original = path_layout()
    original.pinned_at(1, [3, 4])
    assert original.locations[1].tolist() == [10, 0]
    assert original.pinned.tolist() == [False, False, False]


def test_unpinned_frees_the_node():
    layout = path_layout([True, True, False]).unpinned(0)
    assert layout.pinned.tolist() == [False, True, False]
    assert np.linalg.norm(layout.delta[0]) > 0


def test_unpinned_all_frees_every_node():
    layout = path_layout([True, True, False]).unpinned_all()
    assert layout.pinned.tolist() == [False, False, False]


def test_randomized_layout_keeps_pinned_nodes_in_place():
    layout = randomized_layout(path_layout([False, True, False]))
    assert layout.locations[1].tolist() == [10, 0]
    assert layout.pinned.tolist() == [False, True, False]


def test_toggle_pin_pins_a_free_node_in_place():
    layout = toggle_pin(path_layout(), 2)
    assert layout.pinned.tolist() == [False, False, True]
    assert layout.locations[2].tolist() == [20, 5]


def test_toggle_pin_unpins_a_pinned_node():
    layout = toggle_pin(path_layout([False, False, True]), 2)
    assert layout.pinned.tolist() == [False, False, False]
