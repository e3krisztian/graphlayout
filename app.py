# GUI app: shows a graph while its layout is being improved

import tkinter
from time import sleep

from graphs import completegraph, tree, randomg, g1, g2, star, star2, rings
from layout import (
    X, Y, NODES,
    GraphLayout, circle_locations, randomized, target_temperature, improveall,
)


def window_geometry(root):
    # 2/3 of the screen width at the right edge, leaving some room for panels/taskbars
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    width, height = screen_width * 2 // 3, int(screen_height * 0.95)
    x, y = screen_width - width, 0
    return '%dx%d+%d+%d' % (width, height, x, y)


def raise_and_focus(root):
    # window managers tend to ignore a plain focus request from a new window,
    # briefly making it topmost gets it in front
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.focus_force()


class GraphCanvas:
    # space around the outermost nodes, in layout units, for the node circles and labels
    PADDING = 5
    MAX_MAGNIFICATION = 50.

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
        self.canvasitems.append(self.canvas.create_rectangle(x0, y0, x1, y1, fill='yellow'))
        self.canvas.tag_raise(item)

    def draw(self, glayout):
        # fit the layout's bounding box into the canvas, keeping the aspect ratio
        locations = glayout.locations
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        low = locations.min(axis=NODES) - self.PADDING
        high = locations.max(axis=NODES) + self.PADDING
        extent = high - low
        self.magnification = min(
            canvas_width / extent[X], canvas_height / extent[Y], self.MAX_MAGNIFICATION)
        self.clear()

        # draw graph centered in the canvas
        center = (low + high) / 2
        offx = canvas_width / 2 / self.magnification - center[X]
        offy = canvas_height / 2 / self.magnification - center[Y]
        # draw nodes
        for c, n in zip(locations, range(len(locations))):
            x, y = c[X], c[Y]
            self.drawcircle(x + offx, y + offy, 2)
            self.write(x + offx, y + offy, n)
        # draw edges
        for i in range(len(locations)):
            c1 = locations[i]
            x1, y1 = c1[X], c1[Y]
            for dest in glayout.edges[i]:
                if i < dest:
                    c2 = locations[dest]
                    x2, y2 = c2[X], c2[Y]
                    self.drawline(x1+offx, y1+offy, x2+offx, y2+offy)

    def clear(self):
        #  clear previous screen
        for ci in self.canvasitems:
            self.canvas.delete(ci)
        self.canvasitems = []


# (button text, graph creator), the buttons are placed from the right in this order
GRAPHS = [
    ("g2", g2),
    ("g1", g1),
    ("Star", lambda: star(50)),
    ("Star2", lambda: star2(50)),
    ("T 200", lambda: tree(200)),
    ("T 100", lambda: tree(100)),
    ("T 40", lambda: tree(40)),
    ("Random", lambda: randomg(20, 50)),
    ("Complete", lambda: completegraph(40)),
    ("Ring", lambda: rings(20, 5)),
    ("Ring400", lambda: rings(40, 10)),
    ("Pipe", lambda: rings(10, 10)),
    ("Pipe400", lambda: rings(20, 20)),
    ("Pipe2000", lambda: rings(20, 100)),
]


class App:
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.geometry(window_geometry(self.root))
        # pack the toolbar first, so it keeps its space when the window shrinks
        self.toolbar = tkinter.Frame(self.root)
        self.toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        frame = tkinter.Frame(self.root, relief=tkinter.RIDGE, borderwidth=2)
        frame.pack(fill=tkinter.BOTH, expand=1)
        self.canvas = tkinter.Canvas(frame, highlightthickness=0)
        self.canvas.pack(fill=tkinter.BOTH, expand=1)
        self.gcanvas = GraphCanvas(self.canvas)

        self.button(tkinter.LEFT, "Exit", self.exit)
        self.button(tkinter.LEFT, "Randomize", self.randomize)
        for text, create_graph in GRAPHS:
            self.button(
                tkinter.RIGHT, text,
                lambda create_graph=create_graph: self.new_graph(create_graph()))

        self.running = True
        self.new_graph(rings(10, 10))

    def button(self, side, text, command):
        tkinter.Button(self.toolbar, text=text, command=command).pack(side=side)

    def set_layout(self, layout):
        # a new layout restarts the iteration count and the cooling
        self.layout = layout
        self.iteration = 1
        self.temperature = target_temperature(layout, self.iteration)

    def new_graph(self, graph):
        locations = randomized(circle_locations(graph.nodecount))
        self.set_layout(GraphLayout(graph.edges, locations))

    def randomize(self):
        self.set_layout(GraphLayout(self.layout.edges, randomized(self.layout.locations)))

    def exit(self):
        self.running = False

    def run(self):
        # map the window, so the canvas has its real size before the first draw
        self.root.update()
        raise_and_focus(self.root)

        while self.running:
            self.temperature = min(
                self.temperature, target_temperature(self.layout, self.iteration))
            print('n= %4d, magnification=%.5f, tension=%.5f, temperature=%.5f' % (
                self.iteration, self.gcanvas.magnification, self.layout.tension, self.temperature))
            self.layout = improveall(self.layout, self.temperature)
            self.iteration += 1
            self.gcanvas.draw(self.layout)
            sleep(0.01)
            # update native window & process events
            self.canvas.update()


def main():
    App().run()


if __name__ == '__main__':
    main()
