from app import GraphCanvas, LABELS_ABOVE, LABELS_BELOW, LABELS_OFF, centered_position
from layout import EDGE_LENGTH, GraphLayout
from themes import DARK, LIGHT


class RecordingCanvas:
    "stands in for a tkinter canvas of 1000x800 pixels, records the kind and options of the drawn items"

    def __init__(self):
        self.items = []
        self.options = {}

    def winfo_width(self):
        return 1000

    def winfo_height(self):
        return 800

    def configure(self, **options):
        self.options.update(options)

    def delete(self, tag):
        assert tag == 'all'
        self.items = []

    def create_line(self, *coordinates, **options):
        self.items.append(('line', options))

    def create_oval(self, *coordinates, **options):
        self.items.append(('oval', options))

    def create_rectangle(self, *coordinates, **options):
        self.items.append(('rectangle', options))

    def create_text(self, *coordinates, **options):
        self.items.append(('text', options))


class FixedWidthFont:
    def measure(self, text):
        return 10 * len(text)


class DisplaylessGraphCanvas(GraphCanvas):
    # a tkinter font needs a display
    def label_font(self, height):
        return FixedWidthFont()


def draw(layout, theme, labels):
    canvas = RecordingCanvas()
    DisplaylessGraphCanvas(canvas).draw(layout, theme, labels)
    return canvas


def kinds(canvas):
    return [kind for kind, options in canvas.items]


def triangle():
    return GraphLayout([[1, 2], [0, 2], [0, 1]], [[0, 0], [2, 0], [1, 2]])


def test_labels_above_are_drawn_after_the_edges():
    canvas = draw(triangle(), DARK, LABELS_ABOVE)
    assert kinds(canvas) == ['line'] * 3 + ['rectangle', 'text'] * 3


def test_labels_below_are_drawn_before_the_edges():
    canvas = draw(triangle(), DARK, LABELS_BELOW)
    assert kinds(canvas) == ['rectangle', 'text'] * 3 + ['line'] * 3


def test_without_labels_nodes_are_dots_drawn_after_the_edges():
    canvas = draw(triangle(), DARK, LABELS_OFF)
    assert kinds(canvas) == ['line'] * 3 + ['oval'] * 3


def test_labels_are_the_node_numbers():
    canvas = draw(triangle(), DARK, LABELS_ABOVE)
    texts = [options['text'] for kind, options in canvas.items if kind == 'text']
    assert texts == ['0', '1', '2']


def test_edge_of_ideal_length_is_drawn_in_the_ideal_color():
    layout = GraphLayout([[1], [0]], [[0, 0], [EDGE_LENGTH, 0]])
    canvas = draw(layout, LIGHT, LABELS_OFF)
    assert canvas.items[0] == ('line', dict(fill=LIGHT['ideal'], width=2))


def test_edge_of_4_times_the_ideal_length_is_drawn_in_the_stretched_color():
    layout = GraphLayout([[1], [0]], [[0, 0], [0, EDGE_LENGTH * 4]])
    canvas = draw(layout, DARK, LABELS_OFF)
    assert canvas.items[0][1]['fill'] == DARK['stretched']


def test_background_follows_the_theme():
    assert draw(triangle(), DARK, LABELS_OFF).options['background'] == DARK['background']
    assert draw(triangle(), LIGHT, LABELS_OFF).options['background'] == LIGHT['background']


def test_dialog_is_centered_over_the_window():
    # window at (100, 50) of 800x600: its center is (500, 350)
    assert centered_position((100, 50, 800, 600), 200, 100) == (400, 300)
