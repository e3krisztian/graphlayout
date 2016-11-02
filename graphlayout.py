# an incremental graph layout algorithm - prototype
class Graph:
    def __init__(self, n):
        self.n = n
        self._edges = [[] for dummy in xrange(n)]

    def addedge(self, node1, node2):
        self._edges[node1].append(node2)
        self._edges[node2].append(node1)

    def edges(self, nodeindex):
        return self._edges[nodeindex]

    def nodecount(self):
        return self.n


import math, random

class GraphLayout:
    def __init__(self, graph, locations=None):
        self.graph = graph
        self.attrstrength = 1
        self.repustrength = 1
        if locations:
            assert self.graph.nodecount() == len(locations)
            self._locations = locations
        else:
            self._locations = [None] * graph.nodecount()
            self._initloc()

    def clone(self):
        return GraphLayout(self.graph, self._locations[:])

    def _initloc(self):
        n = self.nodecount()
        for i in xrange(n):
            a = 2 * math.pi/n * i
            self._locations[i] = (n*math.cos(a), n*math.sin(a))

    def randomize(self):
        for i in xrange(self.nodecount()):
            x, y = self._locations[i]
            x = x * (random.random()*2/3 + 0.33)
            y = y * (random.random()*2/3 + 0.33)
            self._locations[i] = (x, y)

    def nodecount(self):
        return len(self._locations)

    def location(self, i):
        return self._locations[i]

    def __str__(self):
        return 'Graph: ' + str(self.graph) + '\n' + 'Layout: ' + str(self._locations)

    def draw(self, canvas, offx, offy):
        # draw nodes
        for (x, y), n in zip(self._locations, xrange(len(self._locations))):
            canvas.drawcircle(x + offx, y + offy, 2)
            canvas.write(x + offx, y + offy, n)
        # draw edges
        for i in range(self.nodecount()):
            x1, y1 = self._locations[i]
            for dest in self.graph.edges(i):
                if i < dest:
                    x2, y2 = self._locations[dest]
                    canvas.drawline(x1+offx, y1+offy, x2+offx, y2+offy)

    def delta(self, locations):
        """locations -> change vectors"""
        delta = [None] * len(locations)
        for node in xrange(len(locations)):
            x, y = locations[node]
            dx, dy = ([], [])
            # calculate attraction - along the edges
            for i in self.graph.edges(node):
                ox, oy = locations[i]
                attrx, attry = self.attraction(ox - x, oy - y)
                dx.append(attrx)
                dy.append(attry)

            # calculate repulsion - an effect of all other nodes
            for i in xrange(len(locations)):
                if i <> node:
                    ox, oy = locations[i]
                    repx, repy = self.repulsion(ox - x, oy - y)
                    dx.append(repx)
                    dy.append(repy)

            # set the new location
            dx.sort(key=abs)
            dy.sort(key=abs)
            delta[node] = (sum(dx), sum(dy))
        return delta

    def adddelta(self, locations, delta, t):
        newlocations = [(x + dx * t, y + dy * t)
            for (x, y), (dx, dy)
            in zip(locations, delta)]
        return newlocations

    def improveall(self, t):
        delta = self.delta(self._locations)
        # avg distance2 between nodes
        # approximation: avg is calculated from the distances of the first node
        x0, y0 = self._locations[0]
        avg = sum([(x0 - x) * (x0 - x) + (y0 - y) * (y0 - y)
            for x, y in self._locations]) / len(self._locations)
        # t = scale so that every delta < avg / 2
        maxdelta = max([x * x + y * y for x, y in delta])
        if maxdelta > avg / 2:
            ot = t
            t = t * avg / maxdelta
            print 't was overridden: %f -> %f ' % (ot, t)
        self._locations = self.adddelta(self._locations, delta, t)

    def distance2(self, n1, n2):
        x1, y1 = self.location(n1)
        x2, y2 = self.location(n2)
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
    for step in range(n/2, 0, -1):
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
    for i in xrange(n-1):
        g.addedge(n-1, i)
    return g

def star2(n):
    g = star(n)
    for i in xrange(n-2):
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

import Tkinter
from Tkconstants import *

tk = Tkinter.Tk()
frame = Tkinter.Frame(tk, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
canvas = Tkinter.Canvas(frame, width=800, height=500)
canvas.pack(fill=BOTH, expand=1)

canvas.create_line(0,0,800,500, fill = "red")
canvas.create_line(0,500,800,0, fill = "red")

def button(side, text, command):
    button = Tkinter.Button(frame,text=text,command=command).pack(side=side)

def new_graph(graph):
    global g, t, n
    g = GraphLayout(graph)
    g.randomize()
    t = 1
    n = 1

g = GraphLayout(rings(10, 10))
g.randomize()

def exit_gui():
    global g
    g = None

button(LEFT, "Exit", exit_gui)

def randomize():
    global g
    g.randomize()

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
        print int(5*m), x1-x0, y1-y0
        self.canvasitems.append(self.canvas.create_rectangle(x0, y0, x1, y1, fill='yellow'))
        self.canvas.tag_raise(item)


    def draw(self, glayout):
        # recalculate magnification - to be able to draw the whole graph
        x, y = glayout.location(0)
        maxdist = 0
        for i in range(1, glayout.nodecount()):
            dist = glayout.distance(0, i)
            if dist > maxdist:
                maxdist = dist
        self.magnification = 250.0 / maxdist

        # draw graph centered around its first node
        #  clear previous screen
        for ci in self.canvasitems:
            self.canvas.delete(ci)
        self.canvasitems = []

        #  draw new content
        glayout.draw(self, 400/self.magnification-x, 250/self.magnification-y)

gcanvas = GraphCanvas(canvas)

from time import sleep
# main loop
t = 1
n = 1
while g:
    if n < 1200:
        print 'n= %4d, t=%.5f, magnification=%.5f' % (n, t, gcanvas.magnification)
        g.improveall(t)
        if n > 100:
            t = max(0, t - 0.001)
        n = n + 1
        gcanvas.draw(g)
    else:
        sleep(0.1)
    # update native window & process events
    canvas.update()

# EOF
