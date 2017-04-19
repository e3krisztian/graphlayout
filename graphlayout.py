from __future__ import print_function

debug = print

# an incremental graph layout algorithm - prototype
class Graph:
    def __init__(self, n):
        self.nodecount = n
        self.edges = [[] for _ in range(n)]

    def addedge(self, node1, node2):
        self.edges[node1].append(node2)
        self.edges[node2].append(node1)


def csum(list_of_complex_numbers):
    assert isinstance(list_of_complex_numbers, list)
    real = math.fsum(n.real for n in list_of_complex_numbers)
    imag = math.fsum(n.imag for n in list_of_complex_numbers)
    return complex(real, imag)


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
    def __init__(self, graph, locations):
        assert graph.nodecount == len(locations)
        self.graph = graph
        self.locations = locations
        self.delta = self.calculate_delta()
        self.tension = self.calculate_tension()

    def __str__(self):
        return 'Graph: ' + str(self.graph) + '\n' + 'Layout: ' + str(self.locations)

    def calculate_tension(self):
        return math.fsum(abs(d) for d in self.delta)

    def calculate_delta(self):
        locations = self.locations
        delta = [None] * len(locations)
        for node, c in enumerate(locations):
            d = []
            # calculate attraction - along the edges
            for i in self.graph.edges[node]:
                oc = locations[i]
                # attrx, attry = self.attraction(ox - x, oy - y, edge_length=10)
                attr = self.attraction(oc - c, edge_length=2)
                # attrx, attry = self.attraction(ox - x, oy - y, edge_length=random.randint(2, 40))
                d.append(attr)

            # calculate repulsion - an effect of all other nodes
            for i in range(len(locations)):
                if i != node:
                    oc = locations[i]
                    rep = self.repulsion(oc - c)
                    d.append(rep)

            # set the new location
            delta[node] = csum(d)
        return delta

    def attraction(self, dc, edge_length):
        d = abs(dc)
        try:
            m = (d - edge_length) / edge_length / 2
            return m * dc / d
        except:
            return 0.
            return complex(random.random(), random.random())

    def repulsion(self, dc):
        try:
            d2 = abs(dc) ** 2
            div = d2 / 2
            # for trees:
            # :either:
            # d3 = math.pow(d2, 1.5)
            # div = d3 / 4
            # :or:
            # el2 = 400 # (2 * edge length) ^ 2
            # if d2 > el2:
            #   return (0, 0)
            return -dc / div
            return 0.
        except:
            return complex(-random.random(), -random.random())

    @property
    def approx_diameter(self):
        rmin = min(c.real for c in self.locations)
        rmax = max(c.real for c in self.locations)
        imin = min(c.imag for c in self.locations)
        imax = max(c.imag for c in self.locations)
        return math.sqrt((rmax - rmin) ** 2 + (imax - imin) ** 2)

    def step(self, t):
        '''
            create a new layout by applying delta to the current layout t times
        '''
        new_locations = [c + d * t for c, d in zip(self.locations, self.delta)]
        return GraphLayout(self.graph, new_locations)


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
            g.addedge(i, j)
    return g

def tree(n):
    g = Graph(n)
    for i in range(1, n):
        g.addedge(i, int(i * random.random()))
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
        g.addedge(int(n * random.random()), int(n * random.random()))
    return g

def g1():
    g = Graph(10)
    g.addedge(1,5)
    g.addedge(2,5)
    g.addedge(3,7)
    g.addedge(3,8)
    g.addedge(4,6)
    g.addedge(4,9)
    g.addedge(4,5)
    g.addedge(5,6)
    g.addedge(1,8)
    g.addedge(0,8)
    return g

def g2():
    g = g1()
    g.addedge(9, 7)
    return g

def star(n):
    g = Graph(n)
    for i in range(n-1):
        g.addedge(n-1, i)
    return g

def star2(n):
    g = star(n)
    for i in range(n-2):
        g.addedge(n-2, i)
    return g

def rings(n, m):
    "m rings each constructed of n sections"
    g = Graph(n * m)
    p = permutation(n * m)
    for r in range(m):
        r0 = r*n
        for i in range(n):
            g.addedge(p[r0 + i], p[r0 + (i+1) % n])
        if r > 0:
            for i in range(n):
                g.addedge(p[r0-n +i], p[r0 + i])
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
    g = GraphLayout(graph, randomized(circle_locations(graph)))
    t = 1
    n = 1

def exit_gui():
    global g
    g = None

button(LEFT, "Exit", exit_gui)

def randomize():
    global g
    g = GraphLayout(g.graph, randomized(g.locations))

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

def new_pipe():
    new_graph(rings(10, 10))

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
button(RIGHT, "Pipe", new_pipe)

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
            for dest in glayout.graph.edges[i]:
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
