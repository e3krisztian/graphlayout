import math

import numpy as np
import pytest

from layout import (
    EDGE_LENGTH, JITTER, BALLOON, DENSE,
    GraphLayout, PowerLaw, improved, jitter_due, jittered, randomized_layout, toggle_pin,
)


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


def test_improved_does_not_move_pinned_nodes():
    layout = improved(path_layout([False, True, False]))
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


def test_energy_of_an_edge_of_ideal_length_is_the_repulsion_energy():
    assert two_node_layout(EDGE_LENGTH).energy == pytest.approx(-2 * math.log(EDGE_LENGTH))


def test_energy_of_a_stretched_edge():
    stretch = 3
    layout = two_node_layout(EDGE_LENGTH + stretch)
    assert layout.energy == pytest.approx(
        stretch ** 2 / (4 * EDGE_LENGTH) - 2 * math.log(EDGE_LENGTH + stretch))


def test_energy_of_unconnected_nodes_is_the_repulsion_energy():
    layout = GraphLayout([[], []], [[0, 0], [3, 4]])
    assert layout.energy == pytest.approx(-2 * math.log(5))


@pytest.mark.parametrize('exponent', [0.5, 1, 1.5, 2, 3])
def test_delta_is_the_negative_gradient_of_the_energy(exponent):
    layout = path_layout().with_model(PowerLaw(strength=2, exponent=exponent))
    h = 1e-6
    for node in range(3):
        for axis in range(2):
            locations = layout.locations.copy()
            locations[node, axis] += h
            moved = layout.moved(locations)
            gradient = (moved.energy - layout.energy) / h
            assert -gradient == pytest.approx(layout.delta[node, axis], abs=1e-4)


def test_power_law_accepts_its_boundary_values():
    PowerLaw(strength=0, exponent=1)
    PowerLaw(strength=1, exponent=1e-6)


@pytest.mark.parametrize('strength, exponent, knob', [
    (-1, 1, 'strength'),
    (math.nan, 1, 'strength'),
    (math.inf, 1, 'strength'),
    (1, 0, 'exponent'),
    (1, -1, 'exponent'),
    (1, math.nan, 'exponent'),
    (1, math.inf, 'exponent'),
])
def test_power_law_rejects_broken_constraints_naming_the_knob(strength, exponent, knob):
    with pytest.raises(ValueError, match='^' + knob + ': '):
        PowerLaw(strength=strength, exponent=exponent)


def test_layouts_are_balloon_by_default():
    assert two_node_layout(EDGE_LENGTH).model == BALLOON


def test_balloon_pushes_with_2_over_d_and_has_the_energy_minus_2_ln_d():
    layout = GraphLayout([[], []], [[0, 0], [3, 4]], model=BALLOON)
    # 2/5, along the unit vector [3, 4] / 5
    assert layout.delta[1].tolist() == pytest.approx([2/5 * 3/5, 2/5 * 4/5])
    assert layout.energy == pytest.approx(-2 * math.log(5))


def test_dense_pushes_with_1_over_d_squared_and_has_the_energy_1_over_d_minus_1():
    layout = GraphLayout([[], []], [[0, 0], [3, 4]], model=DENSE)
    assert layout.delta[1].tolist() == pytest.approx([1/25 * 3/5, 1/25 * 4/5])
    assert layout.energy == pytest.approx(1/5 - 1)


def test_energy_is_continuous_across_exponent_1():
    def energy(exponent):
        return GraphLayout(
            [[], []], [[0, 0], [3, 4]], model=PowerLaw(strength=2, exponent=exponent)).energy
    assert energy(1 - 1e-6) == pytest.approx(energy(1), abs=1e-4)
    assert energy(1 + 1e-6) == pytest.approx(energy(1), abs=1e-4)


@pytest.mark.parametrize('derive', [
    lambda layout: layout.step(0.5),
    lambda layout: layout.pinned_at(1, [3, 4]),
    lambda layout: layout.unpinned(0),
    lambda layout: layout.unpinned_all(),
    lambda layout: layout.moved(layout.locations + 1),
    lambda layout: randomized_layout(layout),
    lambda layout: toggle_pin(layout, 2),
    lambda layout: toggle_pin(layout, 0),
    lambda layout: jittered(layout),
    lambda layout: improved(layout),
])
def test_derived_layouts_keep_the_model(derive):
    layout = path_layout([True, False, False]).with_model(DENSE)
    assert derive(layout).model == DENSE


def test_moved_keeps_the_pins_unless_given():
    layout = path_layout([True, False, False])
    assert layout.moved(layout.locations).pinned.tolist() == [True, False, False]
    assert layout.moved(layout.locations, [False, False, True]).pinned.tolist() == [
        False, False, True]


def test_with_model_keeps_the_locations_and_pins():
    layout = path_layout([True, False, False])
    changed = layout.with_model(DENSE)
    assert changed.model == DENSE
    assert changed.locations.tolist() == layout.locations.tolist()
    assert changed.pinned.tolist() == [True, False, False]


@pytest.mark.parametrize('seed', range(5))
def test_improved_does_not_raise_the_energy_when_a_checked_step_lowers_it(seed):
    # in these layouts no nodes get so close that every checked step overshoots
    np.random.seed(seed)
    layout = GraphLayout(
        [[1, 2], [0, 2, 3], [0, 1], [1]], np.random.random((4, 2)) * 10)
    for _ in range(20):
        improved_layout = improved(layout)
        assert improved_layout.energy <= layout.energy
        layout = improved_layout


def test_improved_takes_a_step_that_lowers_the_energy_but_raises_the_tension():
    # every step improved tries along delta raises the tension of this layout
    layout = GraphLayout([[1], [0, 2], [1]], [[2.0, 2.9], [0.2, 0.9], [4.9, 4.1]])
    assert layout.step(1 / 16).tension > layout.tension
    improved_layout = improved(layout)
    assert improved_layout.energy < layout.energy


def test_jittered_moves_free_coordinates_by_minus_one_zero_or_one_jitter():
    np.random.seed(0)
    layout = GraphLayout([[] for _ in range(50)], np.random.random((50, 2)) * 100)
    moves = (jittered(layout).locations - layout.locations) / JITTER
    assert moves == pytest.approx(np.round(moves))
    assert set(np.round(moves).flatten().tolist()) == {-1, 0, 1}


def test_jittered_does_not_move_pinned_nodes():
    layout = path_layout([False, True, False])
    for _ in range(10):
        jittered_layout = jittered(layout)
        assert jittered_layout.locations[1].tolist() == [10, 0]
        assert jittered_layout.pinned.tolist() == [False, True, False]


def test_jitter_is_due_when_the_square_root_of_the_steps_reaches_a_new_integer():
    assert [steps for steps in range(30) if jitter_due(steps)] == [1, 4, 9, 16, 25]


def test_improved_takes_the_smallest_checked_step_when_every_step_keeps_an_infinite_energy():
    # nodes 1 and 3 are on the same spot and their forces cancel, so they stay
    # together and the energy stays infinite; inf <= inf must not let every step pass
    layout = GraphLayout([[1], [0, 2], [1, 3], [2]], [[0, 0], [2, 0], [4, 0], [2, 0]])
    assert layout.energy == math.inf
    assert improved(layout).locations.tolist() == layout.step(1 / 16).locations.tolist()


def test_improved_takes_a_step_from_an_infinite_to_a_finite_energy():
    # nodes 1 and 3 are on the same spot, node 3 is pulled away from node 1 by node 2
    layout = GraphLayout([[1], [0, 2], [1, 3], [2]], [[0, 0], [3, 0], [6, 0], [3, 0]])
    assert layout.energy == math.inf
    assert math.isfinite(improved(layout).energy)


def test_improved_takes_the_smallest_checked_step_when_no_step_lowers_the_energy():
    # nodes 0 and 2 are almost on top of each other, every checked step overshoots
    layout = GraphLayout([[1], [0, 2], [1]], [[2.39, 3.87], [0.94, 3.5], [2.38, 3.87]])
    smallest_step = layout.step(1 / 16)
    assert smallest_step.energy > layout.energy
    assert improved(layout).locations.tolist() == smallest_step.locations.tolist()
