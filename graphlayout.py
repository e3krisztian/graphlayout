from __future__ import print_function

import numpy as np

debug = print

# an incremental graph layout algorithm - prototype
class Graph:
    def __init__(self, n):
        self.nodecount = n
        self.edges = [[] for _ in range(n)]

    def add_edge(self, node1, node2):
        self.edges[node1].append(node2)
        self.edges[node2].append(node1)


import math


def circle_locations(graph):
    n = graph.nodecount
    locations = [None] * n
    for i in range(n):
        a = 2 * math.pi/n * i
        locations[i] = n*complex(math.cos(a), math.sin(a))
    return locations


import random


def randomized(locations):
    return [
        c * complex(random.random()*2/3 + 0.33, random.random()*2/3 + 0.33)
        for c in locations]


class GraphLayout:
    def __init__(self, edges, locations):
        assert len(edges) == len(locations)
        self.edges = [np.array(nodeindices, dtype=np.int64) for nodeindices in edges]
        self.locations = np.array(locations, dtype=np.complex128)
        self.delta = self.calculate_delta()
        self.tension = self.calculate_tension()

    def __str__(self):
        return 'Graph: ' + str(self.edges) + '\n' + 'Layout: ' + str(self.locations)

    def calculate_tension(self):
        return np.abs(self.delta).sum()

    def calculate_delta(self):
        locations = self.locations
        edges = self.edges
        delta = [None] * len(locations)

        for node, location in enumerate(locations):
            # calculate attraction - along the edges
            connected_locations = locations[edges[node]]
            attraction = self.attraction(connected_locations - location)

            # calculate repulsion - an effect of all other nodes
            repulsion = self.repulsion(locations - location)

            # set the new location
            delta[node] = attraction + repulsion
        return np.array(delta, dtype=np.complex128)

    def attraction(self, loc_deltas):
        edge_length = 2
        # edge_length = np.random.randint(2, 40, size=len(loc_deltas))
        # edge_length = 10
        distances = np.abs(loc_deltas)
        attractions = (distances - edge_length) * loc_deltas / (2 * distances * edge_length)
        return np.nansum(attractions)

    def repulsion(self, loc_deltas):
        distances = np.abs(loc_deltas)
        repulsions = -2 * loc_deltas / distances ** 2
        return np.nansum(repulsions)

    @property
    def approx_diameter(self):
        rmin = np.min(self.locations.real)
        rmax = np.max(self.locations.real)
        imin = np.min(self.locations.imag)
        imax = np.max(self.locations.imag)
        return math.sqrt((rmax - rmin) ** 2 + (imax - imin) ** 2)

    def step(self, t):
        '''
            create a new layout by applying delta to the current layout t times
        '''
        new_locations = self.locations + self.delta * t
        return GraphLayout(self.edges, new_locations)


def improveall(layout):
    # find largest t that has less tension than the current layout, but greater than the previous one
    n = 4
    t_curr = 0
    layout_curr = layout
    t_next = 1.0
    while n > 0:
        layout_next = layout.step(t_next)

        if layout_curr.tension <= layout_next.tension:
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
        if layout_mid.tension <= layout_curr.tension:
            layout_curr = layout_mid
            t_curr = t_mid
        else:
            layout_next = layout_mid
            t_next = t_mid
        n -= 1

    if t_curr == 0:
        return layout_mid
    return layout_curr


#--- Graph creators

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

#--- GUI app code

try:
    import tkinter
    from tkinter.constants import *
except ImportError:
    import Tkinter as tkinter
    from Tkconstants import *

tk = tkinter.Tk()
frame = tkinter.Frame(tk, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
canvas = tkinter.Canvas(frame, width=800, height=500)
canvas.pack(fill=BOTH, expand=1)

canvas.create_line(0,0,800,500, fill = "red")
canvas.create_line(0,500,800,0, fill = "red")

def button(side, text, command):
    button = tkinter.Button(frame,text=text,command=command).pack(side=side)

g = None

def new_graph(graph):
    global g, t, n
    g = GraphLayout(graph.edges, randomized(circle_locations(graph)))
    t = 1
    n = 1

def exit_gui():
    global g
    g = None

button(LEFT, "Exit", exit_gui)

def randomize():
    global g
    g = GraphLayout(g.edges, randomized(g.locations))

button(LEFT, "Randomize", randomize)

def new_tree():
    new_graph(tree(400))

def new_tree40():
    new_graph(tree(40))

def new_tree100():
    new_graph(tree(100))

def new_tree200():
    new_graph(tree(200))

def new_random():
    new_graph(randomg(20, 50))

def new_complete():
    new_graph(completegraph(40))

def new_g1():
    new_graph(g1())

def new_g2():
    new_graph(g2())

def new_star():
    new_graph(star(50))

def new_star2():
    new_graph(star2(50))

def new_ring():
    new_graph(rings(20, 5))

def new_ring_400():
    new_graph(rings(40, 10))

def new_pipe():
    new_graph(rings(10, 10))

def new_pipe_400():
    new_graph(rings(20, 20))

def new_pipe_2000():
    new_graph(rings(20, 100))

button(RIGHT, "g2", new_g2)
button(RIGHT, "g1", new_g1)
button(RIGHT, "Star", new_star)
button(RIGHT, "Star2", new_star2)
button(RIGHT, "T 200", new_tree200)
button(RIGHT, "T 100", new_tree100)
button(RIGHT, "T 40", new_tree40)
button(RIGHT, "Random", new_random)
button(RIGHT, "Complete", new_complete)
button(RIGHT, "Ring", new_ring)
button(RIGHT, "Ring400", new_ring_400)
button(RIGHT, "Pipe", new_pipe)
button(RIGHT, "Pipe400", new_pipe_400)
button(RIGHT, "Pipe2000", new_pipe_2000)

frame.pack()

class GraphCanvas:
    def __init__(self, canvas):
        self.canvas = canvas
        self.canvasitems = []
        self.magnification = 10

    def drawcircle(self, x, y, r):
        m = self.magnification
        x = x * m
        y = y * m
        r = r * m
        self.canvasitems.append(self.canvas.create_oval(x-r/2, y-r/2, x+r/2, y+r/2))

    def drawline(self, x1, y1, x2, y2):
        m = self.magnification
        x1 = x1 * m
        y1 = y1 * m
        x2 = x2 * m
        y2 = y2 * m
        self.canvasitems.append(self.canvas.create_line(x1,y1,x2,y2))

    def write(self, x, y, text):
        m = self.magnification
        item = self.canvas.create_text(0,0,text=text,anchor="sw",font=('Courier', int(5*m)))
        x0, y0, x1, y1 = self.canvas.bbox(item)
        x = x*m-(x1-x0)/2
        y = y*m-(y0-y1)/2
        self.canvas.move(item, x, y)
        self.canvasitems.append(item)
        x0, y0, x1, y1 = self.canvas.bbox(item)
        # debug(int(5*m), x1-x0, y1-y0)
        self.canvasitems.append(self.canvas.create_rectangle(x0, y0, x1, y1, fill='yellow'))
        self.canvas.tag_raise(item)

    def draw(self, glayout):
        # recalculate magnification - to be able to draw the whole graph
        self.magnification = 500.0 / glayout.approx_diameter
        self.magnification = max(1., self.magnification)
        self.magnification = min(50., self.magnification)
        self.clear()

        # draw graph centered around its first node
        c = glayout.locations[0]
        x, y = c.real, c.imag
        offx = 400/self.magnification-x
        offy = 250/self.magnification-y
        locations = glayout.locations
        # draw nodes
        for c, n in zip(locations, range(len(locations))):
            x, y = c.real, c.imag
            self.drawcircle(x + offx, y + offy, 2)
            self.write(x + offx, y + offy, n)
        # draw edges
        for i in range(len(locations)):
            c1 = locations[i]
            x1, y1 = c1.real, c1.imag
            for dest in glayout.edges[i]:
                if i < dest:
                    c2 = locations[dest]
                    x2, y2 = c2.real, c2.imag
                    self.drawline(x1+offx, y1+offy, x2+offx, y2+offy)

    def clear(self):
        #  clear previous screen
        for ci in self.canvasitems:
            self.canvas.delete(ci)
        self.canvasitems = []

gcanvas = GraphCanvas(canvas)
new_pipe()

from time import sleep
# main loop
while g:
    print('n= %4d, magnification=%.5f, tension=%.5f' % (n, gcanvas.magnification, g.tension))
    g = improveall(g)
    n = n + 1
    gcanvas.draw(g)
    sleep(0.01)
    # update native window & process events
    canvas.update()

# EOF
