# initial locations: a BFS spanning tree of the graph drawn as a bubble tree

import math
from collections import deque

import numpy as np

from layout import EDGE_LENGTH

# every node keeps a disk of this radius free of other nodes,
# so no two nodes are closer than EDGE_LENGTH
NODE_RADIUS = EDGE_LENGTH / 2
# the angle around a node, towards its parent, that its subtrees leave free,
# so the edge to the parent crosses none of them
PARENT_GAP = math.pi / 2
# iterations of the search for the center of the circle enclosing a subtree
ENCLOSING_ITERATIONS = 64


def neighbour_lists(edges):
    # the sorted distinct neighbours of each node, without self-loops
    return [
        sorted(set(int(other) for other in edges[node] if other != node))
        for node in range(len(edges))]


def bfs(neighbours, source):
    '''
        the nodes reachable from source in BFS order, and the parent of each in the BFS tree
        as a dict, the source's parent being None
    '''
    parents = {source: None}
    order = [source]
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for other in neighbours[node]:
            if other not in parents:
                parents[other] = node
                order.append(other)
                queue.append(other)
    return order, parents


def center(neighbours, node):
    '''
        a node near the center of node's component: the middle of a long path in it,
        found from the farthest node from node and the farthest node from that
    '''
    order, parents = bfs(neighbours, node)
    order, parents = bfs(neighbours, order[-1])
    path = [order[-1]]
    while parents[path[-1]] is not None:
        path.append(parents[path[-1]])
    return path[len(path) // 2]


def rotation(angle):
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])


class Bubble:
    '''
        a subtree laid out around its root at the origin, with its parent towards -x:
        nodes, their (k, 2) points, and the circle enclosing the disks of the nodes
    '''
    def __init__(self, nodes, points, circle_center, radius):
        self.nodes = nodes
        self.points = points
        self.circle_center = circle_center
        self.radius = radius

    def parent_distance(self, extra):
        '''
            the length of the edge to the parent: at least EDGE_LENGTH,
            with the parent's disk outside the circle, plus extra
        '''
        ox, oy = self.circle_center
        reach = self.radius + NODE_RADIUS
        outside = -ox + math.sqrt(max(reach ** 2 - oy ** 2, 0))
        return max(EDGE_LENGTH, outside) + extra

    def seen_from_parent(self, extra):
        '''
            for the parent at parent_distance(extra): the length of the edge, the angle
            of the circle's center off the edge, and half the angle the circle takes up
        '''
        length = self.parent_distance(extra)
        ox, oy = self.circle_center
        x, y = ox + length, oy
        distance = math.hypot(x, y)
        return length, math.atan2(y, x), math.asin(min(self.radius / distance, 1))


def enclosing_circle(centers, radii):
    '''
        a circle enclosing the given circles, near the smallest one:
        Badoiu and Clarkson's iteration moves the center towards the farthest circle
    '''
    circle_center = centers.mean(axis=0)
    for i in range(1, ENCLOSING_ITERATIONS + 1):
        offsets = centers - circle_center
        reaches = np.linalg.norm(offsets, axis=1) + radii
        farthest = int(reaches.argmax())
        distance = reaches[farthest] - radii[farthest]
        direction = offsets[farthest] / distance if distance > 0 else np.zeros(2)
        far_point = centers[farthest] + direction * radii[farthest]
        circle_center = circle_center + (far_point - circle_center) / (i + 1)
    radius = (np.linalg.norm(centers - circle_center, axis=1) + radii).max()
    return circle_center, float(radius)


def fitting_extra(children, available):
    '''
        the smallest extra length of the edges to the children (found to a 1e-6 share
        of EDGE_LENGTH) for which their circles fit into the available angle
    '''
    def taken(extra):
        return sum(2 * child.seen_from_parent(extra)[2] for child in children)

    if taken(0) <= available:
        return 0.0
    low, high = 0.0, EDGE_LENGTH
    while taken(high) > available:
        low, high = high, 2 * high
    while high - low > 1e-6 * EDGE_LENGTH:
        middle = (low + high) / 2
        if taken(middle) > available:
            low = middle
        else:
            high = middle
    return high


def bubble(node, children, available):
    '''
        the Bubble of node, its children's Bubbles spread over the available angle around +x
    '''
    if not children:
        return Bubble([node], np.zeros((1, 2)), np.zeros(2), NODE_RADIUS)
    extra = fitting_extra(children, available)
    seen = [child.seen_from_parent(extra) for child in children]
    gap = (available - sum(2 * half for length, off_edge, half in seen)) / len(children)
    # the widest subtree goes in the middle, pointing away from the parent, each next one
    # to the side that takes up less of the angle so far
    by_width = sorted(range(len(children)), key=lambda i: -seen[i][2])
    widest = by_width[0]
    # the edge to the widest subtree points straight away from the parent, at direction 0,
    # so a chain of wide subtrees goes on straight instead of curling;
    # directions: of the centers of the subtrees' circles
    length, off_edge, half = seen[widest]
    directions = {widest: off_edge}
    # side (-1 or 1): the angle taken up on that side of the direction 0
    extents = {-1: half - off_edge, 1: half + off_edge}
    for i in by_width[1:]:
        side = -1 if extents[-1] < extents[1] else 1
        half = seen[i][2]
        directions[i] = side * (extents[side] + gap + half)
        extents[side] += gap + 2 * half
    # turned as little as needed for both sides to stay within the available angle
    turn_all = min(max(0.0, extents[-1] - available / 2), available / 2 - extents[1])

    nodes = [node]
    points = [np.zeros((1, 2))]
    centers = [np.zeros(2)]
    radii = [NODE_RADIUS]
    for i, child in enumerate(children):
        length, off_edge, half = seen[i]
        # the child's frame turned so that its circle's center is seen at its direction
        turn = rotation(directions[i] + turn_all - off_edge)
        location = turn @ np.array([length, 0.0])
        nodes += child.nodes
        points.append(location + child.points @ turn.T)
        centers.append(location + turn @ child.circle_center)
        radii.append(child.radius)
    circle_center, radius = enclosing_circle(np.array(centers), np.array(radii))
    return Bubble(nodes, np.concatenate(points), circle_center, radius)


def component_bubble(neighbours, node):
    # the Bubble of the BFS tree of node's component, rooted near its center
    root = center(neighbours, node)
    order, parents = bfs(neighbours, root)
    children = {node: [] for node in order}
    for node in order[1:]:
        children[parents[node]].append(node)
    bubbles = {}
    for node in reversed(order):
        available = 2 * math.pi if node == root else 2 * math.pi - PARENT_GAP
        bubbles[node] = bubble(node, [bubbles.pop(child) for child in children[node]], available)
    return bubbles[root]


def bubble_tree_locations(edges):
    '''
        the (n, 2) locations of a BFS spanning tree of each component drawn as a bubble tree,
        the components packed in rows, the largest first
    '''
    neighbours = neighbour_lists(edges)
    seen = set()
    bubbles = []
    for node in range(len(edges)):
        if node not in seen:
            component = component_bubble(neighbours, node)
            seen.update(component.nodes)
            bubbles.append(component)
    bubbles.sort(key=lambda bubble: -bubble.radius)

    locations = np.zeros((len(edges), 2))
    # rows about as wide as the components would be in a square
    row_width = math.sqrt(sum((2 * bubble.radius) ** 2 for bubble in bubbles))
    x = y = row_height = 0.0
    for component in bubbles:
        size = 2 * component.radius
        if x > 0 and x + size > row_width:
            x, y, row_height = 0.0, y + row_height, 0.0
        # the circle's lowest left point at (x, y)
        offset = np.array([x, y]) + component.radius - component.circle_center
        locations[component.nodes] = component.points + offset
        x += size
        row_height = max(row_height, size)
    return locations
