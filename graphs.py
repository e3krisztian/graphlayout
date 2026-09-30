# graphs to lay out: the Graph class and its creators

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
