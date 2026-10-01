# an incremental graph layout algorithm - prototype

import dataclasses
import math

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
                pair_energies = (
                    -self.strength * (distances ** (1 - exponent) - 1) / (1 - exponent))
        # every pair is seen from both of its nodes, hence the halving
        return pair_energies.sum() / 2


# 1/d reaches far: it inflates meshes from the inside, giving wireframe bodies a 3D look
BALLOON = PowerLaw(strength=2, exponent=1)
# 1/d**2 acts close: it spreads trees into clean branches, but converges slowly
DENSE = PowerLaw(strength=1, exponent=2)


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
    def __init__(self, edges, locations, pinned=None, model=BALLOON):
        assert len(edges) == len(locations)
        self.edges = [np.array(nodeindices, dtype=np.int64) for nodeindices in edges]
        self.locations = np.array(locations, dtype=np.float64)
        assert_locations_shape(self.locations, length=len(edges))
        # (n,) bool array, pinned nodes keep their location
        if pinned is None:
            pinned = np.zeros(len(edges), dtype=bool)
        self.pinned = np.array(pinned, dtype=bool)
        assert self.pinned.shape == (len(edges),)
        # the repulsion between the nodes, e.g. a PowerLaw
        self.model = model
        self.delta, self.energy = self.calculate_delta_and_energy()
        self.tension = self.calculate_tension()

    def __str__(self):
        return 'Graph: ' + str(self.edges) + '\n' + 'Layout: ' + str(self.locations)

    def calculate_tension(self):
        return np.linalg.norm(self.delta, axis=COORDINATES).sum()

    def calculate_delta_and_energy(self):
        '''
            delta: the forces on the nodes
            energy: the potential of these forces, delta is its negative gradient,
                so a small enough step along delta always lowers it:
                sum over edges of (d - EDGE_LENGTH)**2 / (4 * EDGE_LENGTH)
                plus the model's repulsion energy
        '''
        locations = self.locations
        assert_locations_shape(locations, length=len(self.edges))
        edges = self.edges
        delta = [None] * len(locations)
        energy = 0.0

        for node, location in enumerate(locations):
            loc_deltas = locations - location
            distances = np.linalg.norm(loc_deltas, axis=COORDINATES)

            # calculate attraction - along the edges
            attraction = self.attraction(loc_deltas[edges[node]])
            # every edge is seen from both of its ends, hence 8 instead of 4
            energy += ((distances[edges[node]] - EDGE_LENGTH) ** 2).sum() / (8 * EDGE_LENGTH)

            # calculate repulsion - an effect of all other nodes
            # loc_deltas[node] is the node's delta to itself, [0, 0]: a non-zero
            # distance turns its force into 0 instead of 0/0 and its energy into
            # that of distance 1, which is 0
            distances[node] = 1
            repulsion = self.model.repulsion(loc_deltas, distances)
            energy += self.model.energy(distances)

            # set the new location
            delta[node] = attraction + repulsion
        result = np.array(delta, dtype=np.float64)
        assert_locations_shape(result, length=len(self.edges))
        # pinned nodes do not move, and their unrelievable forces are left out of the tension
        result[self.pinned] = 0
        return result, energy

    def attraction(self, loc_deltas):
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

    def moved(self, locations, pinned=None):
        '''
            create a layout of the same graph and model at locations,
            keeping the pins unless pinned is given
        '''
        if pinned is None:
            pinned = self.pinned
        return GraphLayout(self.edges, locations, pinned, self.model)

    def with_model(self, model):
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
