# GUI app: shows a graph while its layout is being improved

import math
import statistics
import tkinter
import tkinter.font
from time import sleep

import numpy as np

from graphs import completegraph, tree, randomg, g1, g2, star, star2, pipe
from layout import (
    X, Y, NODES, COORDINATES,
    GraphLayout, circle_locations, randomized, randomized_layout, toggle_pin,
    improveall, jitter_due, jittered,
)
from themes import DARK, LIGHT, strain_color


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


def centered_position(outer, width, height):
    # top left corner for a width x height window centered over outer: (x, y, width, height)
    outer_x, outer_y, outer_width, outer_height = outer
    return outer_x + (outer_width - width) // 2, outer_y + (outer_height - height) // 2


# how the node labels are drawn
LABELS_OFF, LABELS_BELOW, LABELS_ABOVE = 'off', 'below', 'above'


class GraphCanvas:
    # space around the outermost nodes, in layout units, for the node dots and labels
    PADDING = 5
    MAX_MAGNIFICATION = 50.
    # sizes in layout units, with their minimum in pixels
    DOT_SIZE, MIN_DOT_SIZE = 0.6, 3
    LABEL_HEIGHT, MIN_LABEL_HEIGHT = 0.7, 6
    LABEL_MARGIN = 2
    # a node can be grabbed within this many pixels, or its label height if larger
    MIN_GRAB_RADIUS = 8

    def __init__(self, canvas):
        self.canvas = canvas
        self.magnification = 10
        # the layout point drawn at the center of the canvas
        self.center = np.zeros(2)
        # a frozen view keeps its center and magnification, e.g. while a node is dragged
        self.frozen = False
        self.fonts = {}

    def canvas_center(self):
        return np.array([self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2])

    def to_canvas(self, location):
        return ((np.asarray(location) - self.center) * self.magnification
                + self.canvas_center()).tolist()

    def to_layout(self, x, y):
        return (np.array([x, y]) - self.canvas_center()) / self.magnification + self.center

    def node_at(self, glayout, x, y):
        # the node nearest to canvas point (x, y) if it is within grabbing distance, else None
        distances = np.linalg.norm(
            glayout.locations - self.to_layout(x, y), axis=COORDINATES) * self.magnification
        node = int(distances.argmin())
        radius = max(self.LABEL_HEIGHT * self.magnification, self.MIN_GRAB_RADIUS)
        return node if distances[node] <= radius else None

    def fit(self, locations):
        # fit the layout's bounding box into the canvas, keeping the aspect ratio
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        low = locations.min(axis=NODES) - self.PADDING
        high = locations.max(axis=NODES) + self.PADDING
        extent = high - low
        self.magnification = min(
            canvas_width / extent[X], canvas_height / extent[Y], self.MAX_MAGNIFICATION)
        self.center = (low + high) / 2

    def draw(self, glayout, theme, labels):
        if not self.frozen:
            self.fit(glayout.locations)
        self.clear()
        self.canvas.configure(background=theme['background'])

        # canvas coordinates of the nodes
        points = self.to_canvas(glayout.locations)
        pinned = glayout.pinned.tolist()

        # canvas items drawn later cover the ones drawn earlier
        if labels == LABELS_BELOW:
            self.draw_labels(points, pinned, theme)
            self.draw_edges(glayout.edges, points, theme)
        else:
            self.draw_edges(glayout.edges, points, theme)
            if labels == LABELS_ABOVE:
                self.draw_labels(points, pinned, theme)
            else:
                self.draw_dots(points, pinned, theme)

    def draw_edges(self, edges, points, theme):
        width = 2 if self.magnification > 10 else 1
        lines = [
            (points[node], points[dest])
            for node in range(len(points)) for dest in edges[node] if node < dest]
        # canvas lengths: the magnification cancels out in length / reference
        lengths = [math.dist(p1, p2) for p1, p2 in lines]
        if not lengths:
            return
        # colored relative to the median edge, as repulsion keeps all edges
        # longer than EDGE_LENGTH even in a settled layout
        # (at least a pixel, for when most nodes are on top of each other)
        reference = max(statistics.median(lengths), 1)
        for ((x1, y1), (x2, y2)), length in zip(lines, lengths):
            self.canvas.create_line(
                x1, y1, x2, y2, fill=strain_color(length, reference, theme), width=width)

    def draw_dots(self, points, pinned, theme):
        r = max(self.DOT_SIZE * self.magnification, self.MIN_DOT_SIZE) / 2
        for (x, y), is_pinned in zip(points, pinned):
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=theme['pinned'] if is_pinned else theme['node_fill'],
                outline=theme['node_outline'])

    def draw_labels(self, points, pinned, theme):
        # a node is drawn as its number in a box
        height = max(int(self.LABEL_HEIGHT * self.magnification), self.MIN_LABEL_HEIGHT)
        font = self.label_font(height)
        char_width = font.measure('0')
        h = height / 2 + self.LABEL_MARGIN
        for node, ((x, y), is_pinned) in enumerate(zip(points, pinned)):
            text = str(node)
            w = char_width * len(text) / 2 + self.LABEL_MARGIN
            if is_pinned:
                outline, width = theme['pinned'], 2
            else:
                outline, width = theme['node_outline'], 1
            self.canvas.create_rectangle(
                x - w, y - h, x + w, y + h,
                fill=theme['label_box'], outline=outline, width=width)
            self.canvas.create_text(x, y, text=text, font=font, fill=theme['label_text'])

    def label_font(self, height):
        # negative font size is in pixels
        if height not in self.fonts:
            self.fonts[height] = tkinter.font.Font(
                root=self.canvas, family='Courier', size=-height)
        return self.fonts[height]

    def clear(self):
        self.canvas.delete('all')


class PipeDialog:
    # asks for the parameters of a pipe, OK passes them to on_ok
    MIN_NODES_PER_CIRCLE = 3
    MIN_LENGTH = 1

    def __init__(self, parent, parameters, on_ok):
        nodes_per_circle, length, closed = parameters
        self.on_ok = on_ok
        self.window = tkinter.Toplevel(parent)
        # hidden until it is placed, so it does not jump from its default position
        self.window.withdraw()
        self.window.title("Custom pipe")
        self.window.transient(parent)
        self.window.resizable(False, False)

        self.nodes_per_circle = tkinter.StringVar(value=nodes_per_circle)
        self.length = tkinter.StringVar(value=length)
        self.closed = tkinter.BooleanVar(value=closed)
        self.summary = tkinter.StringVar()

        frame = tkinter.Frame(self.window, padx=10, pady=10)
        frame.pack()
        first = self.spinbox(
            frame, 0, "Nodes per circle", self.nodes_per_circle, self.MIN_NODES_PER_CIRCLE)
        self.spinbox(frame, 1, "Length (circles)", self.length, self.MIN_LENGTH)
        tkinter.Checkbutton(frame, text="Closed (torus)", variable=self.closed).grid(
            row=2, column=1, sticky=tkinter.W)
        tkinter.Label(frame, textvariable=self.summary).grid(
            row=3, column=0, columnspan=2, pady=5)
        buttons = tkinter.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=2)
        self.ok_button = tkinter.Button(buttons, text="OK", command=self.ok)
        self.ok_button.pack(side=tkinter.LEFT)
        tkinter.Button(buttons, text="Cancel", command=self.cancel).pack(side=tkinter.LEFT)

        self.window.bind('<Return>', lambda event: self.ok())
        self.window.bind('<Escape>', lambda event: self.cancel())
        self.nodes_per_circle.trace_add('write', self.update_summary)
        self.length.trace_add('write', self.update_summary)
        self.update_summary()

        # center over the parent, the requested size is known after the idle tasks
        self.window.update_idletasks()
        outer = (
            parent.winfo_rootx(), parent.winfo_rooty(),
            parent.winfo_width(), parent.winfo_height())
        self.window.geometry('+%d+%d' % centered_position(
            outer, self.window.winfo_reqwidth(), self.window.winfo_reqheight()))
        self.window.deiconify()

        # the main window does not take input while the dialog is open
        self.window.wait_visibility()
        self.window.grab_set()
        first.focus_set()
        first.selection_range(0, tkinter.END)

    def spinbox(self, frame, row, text, variable, minimum):
        tkinter.Label(frame, text=text).grid(row=row, column=0, sticky=tkinter.E, padx=5)
        spinbox = tkinter.Spinbox(frame, from_=minimum, to=1000, textvariable=variable, width=6)
        spinbox.grid(row=row, column=1, sticky=tkinter.W, pady=2)
        return spinbox

    def sizes(self):
        # (nodes per circle, length) or None if they are not valid
        try:
            nodes_per_circle = int(self.nodes_per_circle.get())
            length = int(self.length.get())
        except ValueError:
            return None
        if nodes_per_circle < self.MIN_NODES_PER_CIRCLE or length < self.MIN_LENGTH:
            return None
        return nodes_per_circle, length

    def update_summary(self, *args):
        sizes = self.sizes()
        if sizes:
            nodes_per_circle, length = sizes
            self.summary.set("%d nodes" % (nodes_per_circle * length))
            self.ok_button.configure(state=tkinter.NORMAL)
        else:
            self.summary.set("nodes per circle: %d or more, length: %d or more" % (
                self.MIN_NODES_PER_CIRCLE, self.MIN_LENGTH))
            self.ok_button.configure(state=tkinter.DISABLED)

    def ok(self):
        sizes = self.sizes()
        if sizes:
            self.window.destroy()
            self.on_ok(*sizes, self.closed.get())

    def cancel(self):
        self.window.destroy()


PIPES = "Pipes"

# (group name, buttons in a row, [(button text, graph creator)])
GRAPH_GROUPS = [
    ("Small", 3, [
        ("g1", g1),
        ("g2", g2),
        ("Random", lambda: randomg(20, 50)),
        ("Star", lambda: star(50)),
        ("Star2", lambda: star2(50)),
        ("Complete", lambda: completegraph(40)),
    ]),
    ("Trees", 2, [
        ("T 40", lambda: tree(40)),
        ("T 100", lambda: tree(100)),
        ("T 200", lambda: tree(200)),
    ]),
    (PIPES, 4, [
        ("Ring", lambda: pipe(20, 5)),
        ("Ring400", lambda: pipe(40, 10)),
        ("Pipe", lambda: pipe(10, 10)),
        ("Pipe400", lambda: pipe(20, 20)),
        ("Pipe2000", lambda: pipe(20, 100)),
        ("Torus", lambda: pipe(10, 10, closed=True)),
        ("Torus400", lambda: pipe(20, 20, closed=True)),
    ]),
]


class App:
    # graphs with more nodes than this start with their labels off
    LABELS_MAX_NODES = 100
    # moving the pointer more than this many pixels with a node grabbed drags it
    DRAG_DISTANCE = 3

    def __init__(self):
        self.root = tkinter.Tk()
        self.root.geometry(window_geometry(self.root))
        # pack the toolbar and the status line first, so they keep their space
        # when the window shrinks
        self.toolbar = tkinter.Frame(self.root)
        self.toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        statusline = tkinter.Frame(self.root)
        statusline.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        frame = tkinter.Frame(self.root, relief=tkinter.RIDGE, borderwidth=2)
        frame.pack(fill=tkinter.BOTH, expand=1)
        self.canvas = tkinter.Canvas(frame, highlightthickness=0)
        self.canvas.pack(fill=tkinter.BOTH, expand=1)
        self.gcanvas = GraphCanvas(self.canvas)

        self.status = tkinter.StringVar()
        self.dark = tkinter.BooleanVar(value=True)
        self.labels = tkinter.StringVar(value=LABELS_ABOVE)
        # the last choice of drawing the labels, for graphs that start with labels
        self.label_order = LABELS_ABOVE
        self.pipe_parameters = (10, 10, False)
        # written to end the wait for events while the animation is stopped
        self.wakeup = tkinter.IntVar()
        self.animating = True
        # a single step of the stopped animation is to be done
        self.step_requested = False
        self.running = True
        # (node, x, y) of the node grabbed by the mouse and where it was grabbed, or None
        self.grabbed = None
        self.dragging = False
        self.create_statusline(statusline)
        self.create_toolbar()
        self.canvas.bind('<ButtonPress-1>', self.grab)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        # the stopped animation has to follow the size of the window
        self.canvas.bind('<Configure>', lambda event: self.wake())
        # closing the window while waiting for events would leave the process waiting forever
        self.root.protocol('WM_DELETE_WINDOW', self.exit)

        self.new_graph(pipe(10, 10))

    def create_statusline(self, statusline):
        tkinter.Label(statusline, textvariable=self.status, font='TkFixedFont').pack(
            side=tkinter.LEFT)
        tkinter.Checkbutton(
            statusline, text="Dark", variable=self.dark, command=self.wake).pack(
                side=tkinter.RIGHT, padx=10)
        for text, value in [
                ("above edges", LABELS_ABOVE), ("below edges", LABELS_BELOW), ("off", LABELS_OFF)]:
            tkinter.Radiobutton(
                statusline, text=text, value=value, variable=self.labels,
                command=self.labels_selected).pack(side=tkinter.RIGHT)
        tkinter.Label(statusline, text="Labels:").pack(side=tkinter.RIGHT)

    def create_toolbar(self):
        tkinter.Button(self.toolbar, text="Exit", command=self.exit).pack(side=tkinter.LEFT)
        tkinter.Button(self.toolbar, text="Randomize", command=self.randomize).pack(
            side=tkinter.LEFT)
        # wide enough for both of its texts
        self.animation_button = tkinter.Button(
            self.toolbar, text="Stop", width=5, command=self.toggle_animation)
        self.animation_button.pack(side=tkinter.LEFT)
        tkinter.Button(self.toolbar, text="Step", command=self.step).pack(side=tkinter.LEFT)
        tkinter.Button(self.toolbar, text="Unpin all", command=self.unpin_all).pack(
            side=tkinter.LEFT)
        for name, columns, graphs in reversed(GRAPH_GROUPS):
            buttons = [
                (text, lambda create_graph=create_graph: self.new_graph(create_graph()))
                for text, create_graph in graphs]
            if name == PIPES:
                buttons.append(("Custom...", self.open_pipe_dialog))
            group = tkinter.LabelFrame(self.toolbar, text=name)
            group.pack(side=tkinter.RIGHT, anchor=tkinter.N, padx=2)
            for i, (text, command) in enumerate(buttons):
                tkinter.Button(group, text=text, command=command).grid(
                    row=i // columns, column=i % columns, sticky=tkinter.EW)

    def set_layout(self, layout):
        # a new layout restarts the iteration count and the jitter
        self.layout = layout
        self.iteration = 1
        self.steps_since_input = 0
        self.wake()

    def new_graph(self, graph):
        locations = randomized(circle_locations(graph.nodecount))
        self.set_layout(GraphLayout(graph.edges, locations))
        self.edgecount = sum(len(neighbours) for neighbours in graph.edges) // 2
        if graph.nodecount > self.LABELS_MAX_NODES:
            self.labels.set(LABELS_OFF)
        else:
            self.labels.set(self.label_order)

    def randomize(self):
        self.set_layout(randomized_layout(self.layout))

    def set_pins(self, layout):
        # a pin change keeps the iteration count, but restarts the jitter
        self.layout = layout
        self.steps_since_input = 0
        self.wake()

    def unpin_all(self):
        self.set_pins(self.layout.unpinned_all())

    def grab(self, event):
        node = self.gcanvas.node_at(self.layout, event.x, event.y)
        if node is None:
            return
        self.grabbed = (node, event.x, event.y)
        self.dragging = False
        # the dragged node stays under the pointer
        self.gcanvas.frozen = True

    def drag(self, event):
        if self.grabbed is None:
            return
        node, x, y = self.grabbed
        if not self.dragging and math.dist((x, y), (event.x, event.y)) <= self.DRAG_DISTANCE:
            return
        self.dragging = True
        self.set_pins(self.layout.pinned_at(node, self.gcanvas.to_layout(event.x, event.y)))

    def release(self, event):
        if self.grabbed is None:
            return
        node, x, y = self.grabbed
        # a dragged node stays pinned, a clicked one is toggled
        self.gcanvas.frozen = False
        if not self.dragging:
            self.set_pins(toggle_pin(self.layout, node))
        else:
            self.wake()
        self.grabbed = None
        self.dragging = False

    def labels_selected(self):
        if self.labels.get() != LABELS_OFF:
            self.label_order = self.labels.get()
        self.wake()

    def open_pipe_dialog(self):
        PipeDialog(self.root, self.pipe_parameters, self.new_pipe)

    def new_pipe(self, nodes_per_circle, length, closed):
        self.pipe_parameters = (nodes_per_circle, length, closed)
        self.new_graph(pipe(nodes_per_circle, length, closed))

    def toggle_animation(self):
        self.set_animating(not self.animating)

    def set_animating(self, animating):
        self.animating = animating
        self.animation_button.configure(text="Stop" if animating else "Start")
        self.wake()

    def step(self):
        # one step only, then stop
        self.step_requested = True
        self.set_animating(False)

    def wake(self):
        self.wakeup.set(0)

    def exit(self):
        self.running = False
        self.wake()

    def run(self):
        # map the window, so the canvas has its real size before the first draw
        self.root.update()
        raise_and_focus(self.root)

        while self.running:
            if self.animating or self.step_requested:
                self.step_requested = False
                self.steps_since_input += 1
                if jitter_due(self.steps_since_input):
                    self.layout = jittered(self.layout)
                self.layout = improveall(self.layout)
                self.iteration += 1
            self.gcanvas.draw(
                self.layout, DARK if self.dark.get() else LIGHT, self.labels.get())
            self.status.set(
                'nodes=%d edges=%d pinned=%d  n=%4d  energy=%.5f  tension=%.5f'
                '  magnification=%.2f' % (
                    len(self.layout.edges), self.edgecount, self.layout.pinned.sum(),
                    self.iteration, self.layout.energy,
                    self.layout.tension, self.gcanvas.magnification))
            if self.animating:
                sleep(0.01)
                # update native window & process events
                self.canvas.update()
            else:
                # no work until an event changes what is to be drawn
                self.root.wait_variable(self.wakeup)


def main():
    App().run()


if __name__ == '__main__':
    main()
