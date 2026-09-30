import numpy as np

from layout import EDGE_LENGTH, GraphLayout


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
