# an incremental graph layout algorithm - prototype

import dataclasses
import math
from collections import deque

import numpy as np

# Coordinate array conventions used throughout this file:
#   A single point is a (2,) float64 array: [x, y].
#   `locations` and `delta` on GraphLayout are (n, 2) float64 arrays,
#   one row per node - row i is node i's [x, y].
#   A batch of deltas (e.g. loc_deltas in attraction/repulsion) is (k, 2):
#   k vectors, each a [x, y] pair.
# Axis/column name constants, so indices read as names instead of bare 0/1:
X, Y = 0, 1                # column index into a (k, 2) array: point[:, X] / point[:, Y]
NODES, COORDINATES = 0, 1  # axis index for reductions over a (k, 2) array

# the length at which an edge neither pulls nor pushes its nodes
EDGE_LENGTH = 2


def assert_locations_shape(array, *, length=None):
    assert array.ndim == 2 and array.shape[1] == 2
    if length is not None:
        assert array.shape[0] == length


def assert_point_shape(array):
    assert array.shape == (2,)


def attraction(loc_deltas):
    '''
        the (2,) pull of the edge springs on a node, from the (k, 2) loc_deltas to its neighbours
    '''
    assert_locations_shape(loc_deltas)
    distances = np.linalg.norm(loc_deltas, axis=COORDINATES)
    distances_column = np.expand_dims(distances, axis=COORDINATES)
    # a neighbour on the same spot has no direction: its 0/0 is dropped by nansum
    with np.errstate(invalid='ignore'):
        attractions = (
            (distances_column - EDGE_LENGTH) * loc_deltas
            / (2 * distances_column * EDGE_LENGTH)
        )
    result = np.nansum(attractions, axis=NODES)
    assert_point_shape(result)
    return result


# A model is a frozen dataclass of knobs; its bound_to(edges) gives a field: the model
# bound to one graph, holding what is computed once per graph.
# field.delta_and_energy(locations) gives the (n, 2) forces on the nodes and the energy,
# the forces being the negative gradient of the energy; fields know nothing of pins.


@dataclasses.dataclass(frozen=True)
class PowerLaw:
    '''
        repulsion between every pair of nodes, fading with a power of their distance
    '''
    # how hard two nodes push each other apart;
    # 0 or more
    strength: float
    # how fast the push fades with distance: two nodes at distance d push each other
    # apart with strength / d**exponent; a low exponent reaches far and spreads the
    # whole graph, a high one acts mostly between close nodes;
    # above 0
    exponent: float

    def __post_init__(self):
        # written so that nan breaks the constraints
        if not (math.isfinite(self.strength) and self.strength >= 0):
            raise ValueError('strength: a number, 0 or more')
        if not (math.isfinite(self.exponent) and self.exponent > 0):
            raise ValueError('exponent: a number above 0')

    def repulsion(self, loc_deltas, distances):
        '''
            the (2,) force on a node, from the (k, 2) loc_deltas and (k,) distances
            to the other nodes
        '''
        assert_locations_shape(loc_deltas)
        distances_column = np.expand_dims(distances, axis=COORDINATES)
        # loc_deltas / distances is the unit vector towards the other node;
        # a node on the same spot has no direction: its 0/0 is dropped by nansum
        with np.errstate(invalid='ignore'):
            repulsions = -self.strength * loc_deltas / distances_column ** (self.exponent + 1)
        result = np.nansum(repulsions, axis=NODES)
        assert_point_shape(result)
        return result

    def energy(self, distances):
        '''
            the node's share of the pair energies, from the (k,) distances to the other nodes:
            -strength * (d**(1 - exponent) - 1) / (1 - exponent) for a pair,
            -strength * ln(d) at exponent 1, which is the limit of the general form;
            its negative derivative is the force strength / d**exponent
        '''
        # without strength there is no energy, also not for nodes on top of each other
        if self.strength == 0:
            return 0.0
        exponent = self.exponent
        # nodes on top of each other have infinite energy, for exponent >= 1
        with np.errstate(divide='ignore'):
            if exponent == 1:
                pair_energies = -self.strength * np.log(distances)
            else:
                # d**(1 - exponent) - 1 as expm1((1 - exponent) * ln(d)): near exponent 1
                # the subtraction would cancel most of the digits of d**(1 - exponent)
                pair_energies = (
                    -self.strength * np.expm1((1 - exponent) * np.log(distances))
                    / (1 - exponent))
        # every pair is seen from both of its nodes, hence the halving
        return pair_energies.sum() / 2

    def bound_to(self, edges):
        return PowerLawField(self, self, edges)


class PowerLawField:
    '''
        a model bound to a graph: springs along the edges, and the repulsion of a PowerLaw,
        the law, between every pair
    '''
    def __init__(self, model, law, edges):
        self.model = model
        self.law = law
        self.edges = edges

    def delta_and_energy(self, locations):
        '''
            forces: the negative gradient of the energy
            energy: sum over edges of (d - EDGE_LENGTH)**2 / (4 * EDGE_LENGTH)
                plus the law's repulsion energy
        '''
        assert_locations_shape(locations, length=len(self.edges))
        edges = self.edges
        forces = np.zeros_like(locations)
        energy = 0.0

        for node, location in enumerate(locations):
            loc_deltas = locations - location
            distances = np.linalg.norm(loc_deltas, axis=COORDINATES)

            # calculate attraction - along the edges
            pull = attraction(loc_deltas[edges[node]])
            # every edge is seen from both of its ends, hence 8 instead of 4
            energy += ((distances[edges[node]] - EDGE_LENGTH) ** 2).sum() / (8 * EDGE_LENGTH)

            # calculate repulsion - an effect of all other nodes
            # loc_deltas[node] is the node's delta to itself, [0, 0]: a non-zero
            # distance turns its force into 0 instead of 0/0 and its energy into
            # that of distance 1, which is 0
            distances[node] = 1
            repulsion = self.law.repulsion(loc_deltas, distances)
            energy += self.law.energy(distances)

            forces[node] = pull + repulsion
        return forces, energy


@dataclasses.dataclass(frozen=True)
class Balloon:
    '''
        repulsion between every pair of nodes, fading with 1/d, its strength scaled to the graph
    '''
    # how far the edges are stretched: at rest the mean over the edges of
    # d * (d - EDGE_LENGTH) is 2 * EDGE_LENGTH * spread, whatever the size of the graph;
    # 0 or more
    spread: float

    def __post_init__(self):
        # written so that nan breaks the constraint
        if not (math.isfinite(self.spread) and self.spread >= 0):
            raise ValueError('spread: a number, 0 or more')

    def bound_to(self, edges):
        '''
            the 1/d PowerLaw with the strength spread * m / P, for m edges and P node pairs;
            at rest the springs and the repulsion balance:
            sum over edges of d * (d - EDGE_LENGTH) / (2 * EDGE_LENGTH) = strength * P,
            as every pair adds d * strength / d to the right side;
            with a fixed strength the edges would grow with the size of the graph
        '''
        nodecount = len(edges)
        pairs = nodecount * (nodecount - 1) / 2
        # every edge is listed at both of its ends
        edgecount = sum(len(neighbours) for neighbours in edges) / 2
        strength = self.spread * edgecount / pairs if pairs > 0 else 0.0
        return PowerLawField(self, PowerLaw(strength=strength, exponent=1), edges)


# 1/d reaches far: it inflates meshes from the inside, giving wireframe bodies a 3D look
BALLOON = Balloon(spread=2)
# 1/d**2 acts close: it spreads trees into clean branches, but converges slowly
DENSE = PowerLaw(strength=1, exponent=2)


@dataclasses.dataclass(frozen=True)
class Stress:
    '''
        a spring between every pair of nodes, its rest length the hop count of the pair
        times EDGE_LENGTH (Kamada-Kawai); it replaces both the edge springs and the repulsion
    '''
    # how much distant pairs count: a pair h hops apart has the weight h**-weight_exponent;
    # 0 makes every pair count the same, 2 (the usual Kamada-Kawai choice) lets
    # the nearby structure dominate;
    # 0 or more
    weight_exponent: float

    def __post_init__(self):
        # written so that nan breaks the constraint
        if not (math.isfinite(self.weight_exponent) and self.weight_exponent >= 0):
            raise ValueError('weight exponent: a number, 0 or more')

    def bound_to(self, edges):
        return StressField(self, edges)


# every pair is held at its hop count times EDGE_LENGTH, so trees and meshes keep their shape
STRESS = Stress(weight_exponent=2)


def hop_counts(edges):
    '''
        the (n, n) int array of the hop counts between the nodes, by a breadth-first search
        from every node; pairs in different components get one more than the longest
        hop count within a component, 1 in a graph without edges
    '''
    n = len(edges)
    neighbours = [
        sorted(set(int(other) for other in edges[node] if other != node)) for node in range(n)]
    hops = np.full((n, n), -1, dtype=np.int64)
    for source in range(n):
        row = [-1] * n
        row[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for other in neighbours[node]:
                if row[other] < 0:
                    row[other] = row[node] + 1
                    queue.append(other)
        hops[source] = row
    unreachable = hops < 0
    if unreachable.any():
        hops[unreachable] = hops.max() + 1
    return hops


class StressField:
    '''
        a Stress bound to a graph, holding its (n, n) rest lengths and weights
    '''
    def __init__(self, model, edges):
        self.model = model
        hops = hop_counts(edges)
        off_diagonal = ~np.eye(len(edges), dtype=bool)
        self.rest_lengths = hops * EDGE_LENGTH
        # the diagonal, a node with itself, has no weight
        self.weights = np.zeros(hops.shape)
        self.weights[off_diagonal] = hops[off_diagonal] ** -float(model.weight_exponent)

    def delta_and_energy(self, locations):
        '''
            forces: the negative gradient of the energy
            energy: sum over pairs of weights * (d - rest_lengths)**2 / (4 * EDGE_LENGTH)
        '''
        assert_locations_shape(locations, length=len(self.weights))
        # (n, n) coordinate deltas from node i (row) to node j (column)
        x_deltas = locations[:, X] - np.expand_dims(locations[:, X], axis=1)
        y_deltas = locations[:, Y] - np.expand_dims(locations[:, Y], axis=1)
        distances = np.hypot(x_deltas, y_deltas)
        stretches = distances - self.rest_lengths
        # every pair is seen from both of its nodes, hence the halving
        energy = (self.weights * stretches ** 2).sum() / 2 / (4 * EDGE_LENGTH)
        # the pull along each pair per unit of coordinate delta; a pair on the same spot
        # has no direction: its 0/0 is dropped
        with np.errstate(invalid='ignore', divide='ignore'):
            pulls = self.weights * stretches / (2 * EDGE_LENGTH * distances)
        pulls[~np.isfinite(pulls)] = 0
        forces = np.stack(
            [(pulls * x_deltas).sum(axis=1), (pulls * y_deltas).sum(axis=1)], axis=COORDINATES)
        return forces, energy


def circle_locations(nodecount):
    n = nodecount
    locations = [None] * n
    for i in range(n):
        a = 2 * math.pi/n * i
        locations[i] = [n * math.cos(a), n * math.sin(a)]
    return locations


def randomized(locations):
    locations = np.asarray(locations, dtype=np.float64)
    scale = np.random.random(locations.shape) * 2/3 + 0.33
    return locations * scale


class GraphLayout:
    def __init__(self, edges, locations, pinned=None, model=BALLOON, field=None):
        assert len(edges) == len(locations)
        self.edges = [np.array(nodeindices, dtype=np.int64) for nodeindices in edges]
        self.locations = np.array(locations, dtype=np.float64)
        assert_locations_shape(self.locations, length=len(edges))
        # (n,) bool array, pinned nodes keep their location
        if pinned is None:
            pinned = np.zeros(len(edges), dtype=bool)
        self.pinned = np.array(pinned, dtype=bool)
        assert self.pinned.shape == (len(edges),)
        # the model bound to the graph, computing the forces and the energy;
        # given by the layouts derived from this one, so it is bound once per graph
        if field is None:
            field = model.bound_to(self.edges)
        self.field = field
        self.delta, self.energy = self.calculate_delta_and_energy()
        self.tension = self.calculate_tension()

    @property
    def model(self):
        return self.field.model

    def __str__(self):
        return 'Graph: ' + str(self.edges) + '\n' + 'Layout: ' + str(self.locations)

    def calculate_tension(self):
        return np.linalg.norm(self.delta, axis=COORDINATES).sum()

    def calculate_delta_and_energy(self):
        '''
            delta: the forces on the nodes
            energy: the potential of these forces, delta is its negative gradient,
                so a small enough step along delta always lowers it
        '''
        forces, energy = self.field.delta_and_energy(self.locations)
        result = np.array(forces, dtype=np.float64)
        assert_locations_shape(result, length=len(self.edges))
        # pinned nodes do not move, and their unrelievable forces are left out of the tension
        result[self.pinned] = 0
        return result, energy

    def moved(self, locations, pinned=None):
        '''
            create a layout of the same graph and model at locations,
            keeping the pins unless pinned is given
        '''
        if pinned is None:
            pinned = self.pinned
        return GraphLayout(self.edges, locations, pinned, field=self.field)

    def with_model(self, model):
        '''
            create a layout of the same graph, locations and pins with another model
        '''
        return GraphLayout(self.edges, self.locations, self.pinned, model)

    def step(self, t):
        '''
            create a new layout by applying delta to the current layout t times
        '''
        new_locations = self.locations + self.delta * t
        return self.moved(new_locations)

    def pinned_at(self, node, location):
        '''
            create a new layout with node moved to location and pinned there
        '''
        locations = self.locations.copy()
        locations[node] = location
        pinned = self.pinned.copy()
        pinned[node] = True
        return self.moved(locations, pinned)

    def unpinned(self, node):
        pinned = self.pinned.copy()
        pinned[node] = False
        return self.moved(self.locations, pinned)

    def unpinned_all(self):
        return self.moved(self.locations, np.zeros(len(self.edges), dtype=bool))


def randomized_layout(layout):
    # pinned nodes stay in place
    pinned_column = np.expand_dims(layout.pinned, axis=COORDINATES)
    locations = np.where(pinned_column, layout.locations, randomized(layout.locations))
    return layout.moved(locations)


def toggle_pin(layout, node):
    # a free node is pinned where it is
    if layout.pinned[node]:
        return layout.unpinned(node)
    return layout.pinned_at(node, layout.locations[node])


# the size of the jitter, in layout units
JITTER = 0.001


def jitter_due(steps):
    '''
        jitter often after an input event, then less and less often:
        when the square root of the steps since then reaches a new integer
    '''
    return steps > 0 and math.isqrt(steps) ** 2 == steps


def jittered(layout):
    '''
        create a new layout with every coordinate of the free nodes
        moved by -JITTER, 0 or JITTER, at random
    '''
    jitter = np.random.randint(-1, 2, layout.locations.shape) * JITTER
    jitter[layout.pinned] = 0
    return layout.moved(layout.locations + jitter)


def lowers_or_keeps_energy(layout_new, layout_old):
    # a step to an infinite energy is never taken: inf <= inf would take every step
    # of a layout with nodes on top of each other, and the largest checked one at that
    return math.isfinite(layout_new.energy) and layout_new.energy <= layout_old.energy


# when even the first step of improved raises the energy, it is halved at most this many times
MAX_HALVINGS = 30


def step_move_limit(layout):
    '''
        the farthest a step of improved moves a node: the median edge length, so the limit
        shrinks with the layout, but at least EDGE_LENGTH
    '''
    sources = np.repeat(np.arange(len(layout.edges)), [len(e) for e in layout.edges])
    if len(sources) == 0:
        return EDGE_LENGTH
    targets = np.concatenate(layout.edges)
    # every edge is listed at both of its ends
    once = sources < targets
    lengths = np.linalg.norm(
        layout.locations[targets[once]] - layout.locations[sources[once]], axis=COORDINATES)
    return max(float(np.median(lengths)), EDGE_LENGTH)


def improved(layout):
    '''
        create a new layout one step along delta, searching for a step size that lowers the energy:
        doubling it, then bisecting; no step moves a node farther than step_move_limit;
        when even the first step raises the energy, it is halved until a step does not,
        and in the rare case that none does, the smallest one is taken
    '''
    largest_force = np.linalg.norm(layout.delta, axis=COORDINATES).max(initial=0)
    t_limit = step_move_limit(layout) / largest_force if largest_force > 0 else math.inf

    # double t while the step does not raise the energy
    n = 4
    t_curr = 0
    layout_curr = layout
    t_next = min(1.0, t_limit)
    while n > 0:
        layout_next = layout.step(t_next)

        if not lowers_or_keeps_energy(layout_next, layout_curr):
            break

        t_curr = t_next
        layout_curr = layout_next
        if t_curr == t_limit:
            # the farthest step allowed does not raise the energy
            return layout_curr
        t_next = min(t_next + t_next, t_limit)
        n -= 1

    if layout_curr is layout:
        # even the first step raises the energy: a small enough step along delta lowers it
        for _ in range(MAX_HALVINGS):
            t_next /= 2
            layout_next = layout.step(t_next)
            if lowers_or_keeps_energy(layout_next, layout):
                return layout_next
        # no checked step lowers the energy: take the smallest one, raising the energy,
        # rather than getting stuck with the unchanged layout
        return layout_next

    # bisect-find the best layout between
    n = 4
    while n > 0:
        t_mid = (t_curr + t_next) / 2
        layout_mid = layout.step(t_mid)
        if lowers_or_keeps_energy(layout_mid, layout_curr):
            layout_curr = layout_mid
            t_curr = t_mid
        else:
            t_next = t_mid
        n -= 1

    return layout_curr
