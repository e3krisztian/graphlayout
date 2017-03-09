from __future__ import print_function

debug = print

# an incremental graph layout algorithm - prototype
class Graph:
    def __init__(self, n):
        self.nodecount = n
        self.edges = [[] for dummy in range(n)]

    def addedge(self, node1, node2):
        self.edges[node1].append(node2)
        self.edges[node2].append(node1)


import math


def circle_locations(graph):
    n = graph.nodecount
    locations = [None] * n
    for i in range(n):
        a = 2 * math.pi/n * i
        locations[i] = (n*math.cos(a), n*math.sin(a))
    return locations


import random


def randomized(locations):
    return [
        (x * (random.random()*2/3 + 0.33), y * (random.random()*2/3 + 0.33))
        for x, y in locations]


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
        return math.fsum(abs(dx) for dx, _ in self.delta) + math.fsum(abs(dy) for _, dy in self.delta)

    def calculate_delta(self):
        locations = self.locations
        delta = [None] * len(locations)
        for node, (x, y) in enumerate(locations):
            dx, dy = [], []
            # calculate attraction - along the edges
            for i in self.graph.edges[node]:
                ox, oy = locations[i]
                attrx, attry = self.attraction(ox - x, oy - y)
                dx.append(attrx)
                dy.append(attry)

            # calculate repulsion - an effect of all other nodes
            for i in range(len(locations)):
                if i != node:
                    ox, oy = locations[i]
                    repx, repy = self.repulsion(ox - x, oy - y)
                    dx.append(repx)
                    dy.append(repy)

            # set the new location
            delta[node] = (math.fsum(dx), math.fsum(dy))
        return delta

    def distance2(self, n1, n2):
        x1, y1 = self.locations[n1]
        x2, y2 = self.locations[n2]
        dx = x1-x2
        dy = y1-y2
        return dx * dx + dy * dy

    def distance(self, n1, n2):
        return math.sqrt(self.distance2(n1, n2))

    def attraction(self, dx, dy):
        d = math.sqrt(dx * dx + dy * dy)
        el = 10 # edge length
        try:
            m = (d - el) / el / 2
            return (m * dx / d, m * dy / d)
        except:
            return (random.random(), random.random())

    def repulsion(self, dx, dy):
        try:
            d2 = dx * dx + dy * dy
            div = d2 / 2
            # for trees:
            # :either:
            # d3 = math.pow(d2, 1.5)
            # div = d3 / 4
            # :or:
            # el2 = 400 # (2 * edge length) ^ 2
            # if d2 > el2:
            #   return (0, 0)
            return (-dx / div, -dy / div)
        except:
            return (-random.random(), -random.random())


def adddelta(locations, delta, t):
    return [(x + dx * t, y + dy * t) for (x, y), (dx, dy) in zip(locations, delta)]


def improveall(layout, t):
    # avg distance2 between nodes
    # approximation: avg is calculated from the distances of the first node
    x0, y0 = layout.locations[0]
    avg = math.fsum([(x0 - x) * (x0 - x) + (y0 - y) * (y0 - y)
        for x, y in layout.locations]) / len(layout.locations)
    # t = scale so that every delta < avg / 2
    maxdelta = max([x * x + y * y for x, y in layout.delta])
    if maxdelta > avg / 2:
        ot = t
        t = t * avg / maxdelta
        debug('t was overridden: %f -> %f ' % (ot, t))
    new_locations = adddelta(layout.locations, layout.delta, t)
    return GraphLayout(layout.graph, new_locations)

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
    new_graph(tree(40))

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
button(RIGHT, "Tree", new_tree)
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
        # draw graph centered around its first node
        # recalculate magnification - to be able to draw the whole graph
        x, y = glayout.locations[0]
        maxdist = 0
        for i in range(1, len(glayout.locations)):
            dist = glayout.distance(0, i)
            if dist > maxdist:
                maxdist = dist
        self.magnification = 250.0 / maxdist

        self.clear()

        offx = 400/self.magnification-x
        offy = 250/self.magnification-y
        locations = glayout.locations
        # draw nodes
        for (x, y), n in zip(locations, range(len(locations))):
            self.drawcircle(x + offx, y + offy, 2)
            self.write(x + offx, y + offy, n)
        # draw edges
        for i in range(len(locations)):
            x1, y1 = locations[i]
            for dest in glayout.graph.edges[i]:
                if i < dest:
                    x2, y2 = locations[dest]
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
t = 1
n = 1
while g:
    if n < 1200:
        print('n= %4d, t=%.5f, magnification=%.5f, tension=%.5f' % (n, t, gcanvas.magnification, g.tension))
        g = improveall(g, t)
        if n > 100:
            t = max(0, t - 0.001)
        n = n + 1
        gcanvas.draw(g)
    else:
        sleep(0.1)
    # update native window & process events
    canvas.update()

# EOF
