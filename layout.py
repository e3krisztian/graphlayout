# an incremental graph layout algorithm - prototype

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
    def __init__(self, edges, locations, pinned=None):
        assert len(edges) == len(locations)
        self.edges = [np.array(nodeindices, dtype=np.int64) for nodeindices in edges]
        self.locations = np.array(locations, dtype=np.float64)
        assert_locations_shape(self.locations, length=len(edges))
        # (n,) bool array, pinned nodes keep their location
        if pinned is None:
            pinned = np.zeros(len(edges), dtype=bool)
        self.pinned = np.array(pinned, dtype=bool)
        assert self.pinned.shape == (len(edges),)
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
                minus sum over node pairs of 2 * ln(d)
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
            # distance turns its force into 0 instead of 0/0 and its energy into ln(1) = 0
            distances[node] = 1
            repulsion = self.repulsion(loc_deltas, distances)
            # every pair is seen from both of its nodes, hence 1 instead of 2;
            # nodes on top of each other have infinite energy
            with np.errstate(divide='ignore'):
                energy -= np.log(distances).sum()

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
        attractions = (
            (distances_column - EDGE_LENGTH) * loc_deltas
            / (2 * distances_column * EDGE_LENGTH)
        )
        result = np.nansum(attractions, axis=NODES)
        assert_point_shape(result)
        return result

    def repulsion(self, loc_deltas, distances):
        assert_locations_shape(loc_deltas)
        distances_column = np.expand_dims(distances, axis=COORDINATES)
        repulsions = -2 * loc_deltas / distances_column ** 2
        result = np.nansum(repulsions, axis=NODES)
        assert_point_shape(result)
        return result

    def step(self, t):
        '''
            create a new layout by applying delta to the current layout t times
        '''
        new_locations = self.locations + self.delta * t
        return GraphLayout(self.edges, new_locations, self.pinned)

    def pinned_at(self, node, location):
        '''
            create a new layout with node moved to location and pinned there
        '''
        locations = self.locations.copy()
        locations[node] = location
        pinned = self.pinned.copy()
        pinned[node] = True
        return GraphLayout(self.edges, locations, pinned)

    def unpinned(self, node):
        pinned = self.pinned.copy()
        pinned[node] = False
        return GraphLayout(self.edges, self.locations, pinned)

    def unpinned_all(self):
        return GraphLayout(self.edges, self.locations)


def randomized_layout(layout):
    # pinned nodes stay in place
    pinned_column = np.expand_dims(layout.pinned, axis=COORDINATES)
    locations = np.where(pinned_column, layout.locations, randomized(layout.locations))
    return GraphLayout(layout.edges, locations, layout.pinned)


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
    return GraphLayout(layout.edges, layout.locations + jitter, layout.pinned)


def improveall(layout):
    # double t while the step does not raise the energy
    n = 4
    t_curr = 0
    layout_curr = layout
    t_next = 1.0
    while n > 0:
        layout_next = layout.step(t_next)

        if layout_next.energy > layout_curr.energy:
            break

        t_curr = t_next
        t_next = t_next + t_next
        layout_curr = layout_next
        n -= 1

    # bisect-find the best layout between
    n = 4
    while n > 0:
        t_mid = (t_curr + t_next) / 2
        layout_mid = layout.step(t_mid)
        if layout_mid.energy <= layout_curr.energy:
            layout_curr = layout_mid
            t_curr = t_mid
        else:
            t_next = t_mid
        n -= 1

    return layout_curr
