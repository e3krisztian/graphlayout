# graphs to lay out: the Graph class and its creators

import math
import random


class Graph:
    def __init__(self, n):
        self.nodecount = n
        self.edges = [[] for _ in range(n)]

    def add_edge(self, node1, node2):
        self.edges[node1].append(node2)
        self.edges[node2].append(node1)


def completegraph(n):
    g = Graph(n)
    for i in range(n):
        for j in range(i+1, n):
            g.add_edge(i, j)
    return g

def tree(n):
    g = Graph(n)
    for i in range(1, n):
        g.add_edge(i, int(i * random.random()))
    return g

def permutation(n):
    x = [0] * n
    for i in range(n):
        x[i] = i
    for step in range(n//2, 0, -1):
        for i in range(n):
            if random.random() > 0.5:
                i2 = (i+step)%n
                z = x[i]
                x[i] = x[i2]
                x[i2] = z
    return x

def randomg(n, e):
    g = Graph(n)
    for i in range(e):
        g.add_edge(int(n * random.random()), int(n * random.random()))
    return g

def g1():
    g = Graph(10)
    g.add_edge(1,5)
    g.add_edge(2,5)
    g.add_edge(3,7)
    g.add_edge(3,8)
    g.add_edge(4,6)
    g.add_edge(4,9)
    g.add_edge(4,5)
    g.add_edge(5,6)
    g.add_edge(1,8)
    g.add_edge(0,8)
    return g

def g2():
    g = g1()
    g.add_edge(9, 7)
    return g

def star(n):
    g = Graph(n)
    for i in range(n-1):
        g.add_edge(n-1, i)
    return g

def star2(n):
    g = star(n)
    for i in range(n-2):
        g.add_edge(n-2, i)
    return g

def pipe(nodes_per_circle, length, closed=False):
    '''
        length circles of nodes_per_circle nodes each, every circle linked to the next one

        closed links the last circle to the first one as well, making it a torus
    '''
    n, m = nodes_per_circle, length
    g = Graph(n * m)
    p = permutation(n * m)
    # a circle of 2 nodes is a single edge, of 1 node is no edge at all
    circle_edges = n if n > 2 else n - 1
    # 2 circles are already linked, 1 circle would be linked to itself
    linked_circles = m if closed and m > 2 else m - 1
    for r in range(m):
        r0 = r*n
        for i in range(circle_edges):
            g.add_edge(p[r0 + i], p[r0 + (i+1) % n])
    for r in range(linked_circles):
        r0 = r*n
        r1 = (r+1) % m * n
        for i in range(n):
            g.add_edge(p[r0 + i], p[r1 + i])
    return g

# the graphs of the platonic solids: the vertices and the edges of the solid

def tetrahedron():
    return completegraph(4)

def cube():
    'the 3 dimensional hypercube: nodes differing in a single bit are linked'
    g = Graph(8)
    for i in range(8):
        for bit in range(3):
            j = i ^ (1 << bit)
            if i < j:
                g.add_edge(i, j)
    return g

def octahedron():
    'every node linked to all the others, except to its opposite'
    g = Graph(6)
    for i in range(6):
        for j in range(i+1, 6):
            if j != i + 3:
                g.add_edge(i, j)
    return g

def generalized_petersen(n, k):
    '''
        an outer ring of n nodes, each linked to a node of an inner ring,
        where the inner nodes are linked to the ones k steps further
    '''
    g = Graph(2 * n)
    for i in range(n):
        g.add_edge(i, (i+1) % n)
        g.add_edge(i, n + i)
        g.add_edge(n + i, n + (i+k) % n)
    return g

def petersen():
    return generalized_petersen(5, 2)

def heawood():
    '''
        the points and the lines of the Fano plane, each point linked to the 3 lines through it:
        a ring of 14 nodes, the even ones also linked to the node 5 steps further
    '''
    g = Graph(14)
    for i in range(14):
        g.add_edge(i, (i+1) % 14)
        if i % 2 == 0:
            g.add_edge(i, (i+5) % 14)
    return g

def cube_stack(width, depth, height):
    '''
        the corners and the edges of width x depth x height unit cubes stacked into a block

        the corner at (x, y, z) is the node x + (width+1) * (y + (depth+1) * z)
    '''
    sizes = (width + 1, depth + 1, height + 1)
    g = Graph(sizes[0] * sizes[1] * sizes[2])
    steps = (1, sizes[0], sizes[0] * sizes[1])
    for z in range(sizes[2]):
        for y in range(sizes[1]):
            for x in range(sizes[0]):
                node = x + steps[1] * y + steps[2] * z
                for coordinate, size, step in zip((x, y, z), sizes, steps):
                    if coordinate + 1 < size:
                        g.add_edge(node, node + step)
    return g

def eiffel_tower():
    '''
        a wireframe of the Eiffel tower, built of square lattice trusses

        4 legs rise from the ground, linked by girders at the first and the second platform;
        above the second platform the legs join into a single shaft, topped by a spire

        in a leg's square, corner 0 faces the centre of the tower, corner 1 the next leg,
        corner 2 the outside, corner 3 the previous leg
    '''
    legs = 4
    # levels of the legs; the platforms are at these levels
    leg_levels = 18
    platforms = [9, 17]
    # nodes between two legs on a platform girder rail
    girder_nodes = 4
    shaft_levels = 16
    spire_nodes = 4

    edges = []
    nodecount = 0

    def new_nodes(count):
        nonlocal nodecount
        nodecount += count
        return list(range(nodecount - count, nodecount))

    def truss(squares):
        '''
            squares of 4 nodes, one above the other: the sides of each square,
            the verticals between neighbouring squares and a zigzag of diagonals on each face
        '''
        for level, square in enumerate(squares):
            for corner in range(4):
                edges.append((square[corner], square[(corner+1) % 4]))
            if level > 0:
                below = squares[level - 1]
                for corner in range(4):
                    next_corner = (corner+1) % 4
                    edges.append((below[corner], square[corner]))
                    if level % 2:
                        edges.append((below[corner], square[next_corner]))
                    else:
                        edges.append((below[next_corner], square[corner]))

    def rail(start, end):
        '''a path of girder_nodes new nodes from start to end, returned with its ends'''
        path = [start] + new_nodes(girder_nodes) + [end]
        edges.extend(zip(path, path[1:]))
        return path

    leg_squares = [[new_nodes(4) for _ in range(leg_levels)] for _ in range(legs)]
    for squares in leg_squares:
        truss(squares)

    for level in platforms:
        for leg in range(legs):
            here = leg_squares[leg][level]
            there = leg_squares[(leg+1) % legs][level]
            # the inner rail links the inner corners, the outer rail the facing side corners
            inner = rail(here[0], there[0])
            outer = rail(here[1], there[3])
            for node1, node2 in zip(inner[1:-1], outer[1:-1]):
                edges.append((node1, node2))

    shaft_squares = [new_nodes(4) for _ in range(shaft_levels)]
    truss(shaft_squares)
    # each corner of the shaft's bottom square stands on the top square of a leg
    for leg, node in enumerate(shaft_squares[0]):
        for leg_node in leg_squares[leg][-1]:
            edges.append((node, leg_node))

    spire = new_nodes(spire_nodes)
    for node in shaft_squares[-1]:
        edges.append((node, spire[0]))
    edges.extend(zip(spire, spire[1:]))

    g = Graph(nodecount)
    for node1, node2 in edges:
        g.add_edge(node1, node2)
    return g

def eiffel_tower_front():
    '''
        the Eiffel tower seen from the front, cut out of a lattice of equilateral triangles

        2 legs rising from the ground, an arch between them below the first platform,
        the legs joining above the second platform, the third platform near the top, a spire

        the proportions follow the real tower; heights and widths are in edge lengths
    '''
    height = 60
    platforms = [0.19 * height, 0.38 * height, 0.92 * height]
    platform_depth = 1.8
    overhang = 0.6
    # the legs join at this height
    merge = 0.55 * height
    top = 0.96 * height
    arch_rib = 1.0
    spire_nodes = 3
    row_height = math.sqrt(3) / 2

    def half_width(y):
        return 0.21 * height * math.exp(-y / (0.33 * height)) + 0.5

    def leg_width(y):
        # a leg narrower than 2 would break into triangles linked only at their corners
        return max(0.4 * half_width(y), 2.0)

    # the arch is a circle through the inner sides of the legs on the ground and its apex
    arch_x = half_width(0) - leg_width(0)
    arch_apex = platforms[0] - platform_depth - 2.2
    arch_radius = (arch_x ** 2 + arch_apex ** 2) / (2 * arch_apex)
    arch_centre = arch_apex - arch_radius

    def inside(x, y):
        if any(platform - platform_depth <= y <= platform for platform in platforms):
            return abs(x) <= half_width(y) + overhang
        if abs(x) > half_width(y):
            return False
        # between the legs: the inner sides of the legs close in towards merge
        between = (half_width(y) - leg_width(y)) * min(1, (merge - y) / 4)
        if y < merge and abs(x) < between:
            on_arch = abs(math.hypot(x, y - arch_centre) - arch_radius) < arch_rib
            return y < platforms[0] and on_arch
        return True

    # a point is (row, x in half edge lengths); the odd rows are shifted by half an edge
    rows = int(top / row_height) + 1
    points = {
        (row, column) for row in range(rows) for column in range(-80, 81)
        if column % 2 == row % 2 and inside(column / 2, row * row_height)}

    def neighbours(point):
        row, column = point
        candidates = [
            (row, column - 2), (row, column + 2), (row - 1, column - 1), (row - 1, column + 1),
            (row + 1, column - 1), (row + 1, column + 1)]
        return [candidate for candidate in candidates if candidate in points]

    # points not on a triangle would hang loose: drop them, until every point is on one
    while True:
        loose = {
            point for point in points
            if not any(set(neighbours(point)) & set(neighbours(other))
                       for other in neighbours(point))}
        if not loose:
            break
        points -= loose

    nodes = {point: node for node, point in enumerate(sorted(points))}
    g = Graph(len(nodes) + spire_nodes)
    for point, node in nodes.items():
        for neighbour in neighbours(point):
            if nodes[neighbour] > node:
                g.add_edge(node, nodes[neighbour])
    spire = list(range(len(nodes), len(nodes) + spire_nodes))
    top_row = max(row for row, _ in points)
    for (row, _), node in nodes.items():
        if row == top_row:
            g.add_edge(node, spire[0])
    for node1, node2 in zip(spire, spire[1:]):
        g.add_edge(node1, node2)
    return g

def moser_spindle():
    '''
        two rhombi of 2 triangles each, sharing a node at one end, their other ends linked

        all its edges can be drawn with unit length, but it needs 4 colours
    '''
    g = Graph(7)
    for a, b, tip in [(1, 2, 3), (4, 5, 6)]:
        g.add_edge(0, a)
        g.add_edge(0, b)
        g.add_edge(a, b)
        g.add_edge(a, tip)
        g.add_edge(b, tip)
    g.add_edge(3, 6)
    return g

def hanoi(disks):
    '''
        the states of the Tower of Hanoi with 3 pegs, linked by the legal moves

        a state is the pegs of the disks, from the smallest disk up,
        its node is the pegs read as a number in base 3
    '''
    g = Graph(3 ** disks)
    for node in range(3 ** disks):
        pegs = [node // 3 ** disk % 3 for disk in range(disks)]
        # the top of each peg: the smallest disk on it, or disks if it is empty
        tops = [pegs.index(peg) if peg in pegs else disks for peg in range(3)]
        for source in range(3):
            for target in range(3):
                # a move and its reverse are the same edge: add it from the lower peg
                if tops[source] < tops[target] and target > source:
                    g.add_edge(node, node + (target - source) * 3 ** tops[source])
    return g

def dodecahedron():
    return generalized_petersen(10, 2)

def icosahedron():
    '''
        a top node above an upper ring of 5, a lower ring of 5 below it, a bottom node below that;
        each upper ring node is linked to the 2 closest lower ring nodes
    '''
    top, upper, lower, bottom = 0, 1, 6, 11
    g = Graph(12)
    for i in range(5):
        g.add_edge(top, upper + i)
        g.add_edge(upper + i, upper + (i+1) % 5)
        g.add_edge(upper + i, lower + i)
        g.add_edge(upper + i, lower + (i+1) % 5)
        g.add_edge(lower + i, lower + (i+1) % 5)
        g.add_edge(lower + i, bottom)
    return g
