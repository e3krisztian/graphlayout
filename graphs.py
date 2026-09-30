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

def rings(n, m):
    "m rings each constructed of n sections"
    g = Graph(n * m)
    p = permutation(n * m)
    for r in range(m):
        r0 = r*n
        for i in range(n):
            g.add_edge(p[r0 + i], p[r0 + (i+1) % n])
        if r > 0:
            for i in range(n):
                g.add_edge(p[r0-n +i], p[r0 + i])
    return g
