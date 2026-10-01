import pytest

from app import (
    GraphCanvas, LABELS_ABOVE, LABELS_BELOW, LABELS_OFF, centered_position,
    model_with_knob, preset_name,
)
from layout import EDGE_LENGTH, BALLOON, DENSE, GraphLayout, PowerLaw
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


def edge_fills(canvas):
    return [options['fill'] for kind, options in canvas.items if kind == 'line']


def test_edges_are_colored_relative_to_the_median_edge_length():
    # path 0-1-2-3 with edges of length 1, 1 and 4 - the median is 1
    layout = GraphLayout([[1], [0, 2], [1, 3], [2]], [[0, 0], [1, 0], [2, 0], [6, 0]])
    canvas = draw(layout, DARK, LABELS_OFF)
    assert edge_fills(canvas) == [DARK['ideal'], DARK['ideal'], DARK['stretched']]


def test_edges_longer_than_edge_length_are_ideal_when_all_are_equal():
    # a settled layout has all edges longer than EDGE_LENGTH, due to repulsion
    layout = GraphLayout([[1], [0]], [[0, 0], [0, EDGE_LENGTH * 4]])
    canvas = draw(layout, LIGHT, LABELS_OFF)
    assert edge_fills(canvas) == [LIGHT['ideal']]


def test_graph_without_edges_is_drawn():
    layout = GraphLayout([[], []], [[0, 0], [1, 1]])
    assert kinds(draw(layout, DARK, LABELS_OFF)) == ['oval'] * 2


def test_background_follows_the_theme():
    assert draw(triangle(), DARK, LABELS_OFF).options['background'] == DARK['background']
    assert draw(triangle(), LIGHT, LABELS_OFF).options['background'] == LIGHT['background']


def test_dialog_is_centered_over_the_window():
    # window at (100, 50) of 800x600: its center is (500, 350)
    assert centered_position((100, 50, 800, 600), 200, 100) == (400, 300)


def drawn_canvas(layout, labels=LABELS_OFF):
    gcanvas = DisplaylessGraphCanvas(RecordingCanvas())
    gcanvas.draw(layout, DARK, labels)
    return gcanvas


def test_node_at_finds_the_node_under_the_pointer():
    layout = triangle()
    gcanvas = drawn_canvas(layout)
    for node, location in enumerate(layout.locations):
        x, y = gcanvas.to_canvas(location)
        assert gcanvas.node_at(layout, x + 1, y - 1) == node


def test_node_at_is_none_away_from_the_nodes():
    assert drawn_canvas(triangle()).node_at(triangle(), 0, 0) is None


def test_to_layout_reverses_to_canvas():
    gcanvas = drawn_canvas(triangle())
    x, y = gcanvas.to_canvas([1.5, -2])
    assert gcanvas.to_layout(x, y).tolist() == pytest.approx([1.5, -2])


def test_frozen_view_keeps_its_transform_when_the_layout_changes():
    gcanvas = drawn_canvas(triangle())
    before = gcanvas.to_canvas([1, 1])
    gcanvas.frozen = True
    gcanvas.draw(triangle().pinned_at(2, [100, 100]), DARK, LABELS_OFF)
    assert gcanvas.to_canvas([1, 1]) == pytest.approx(before)


def test_view_is_fitted_again_when_unfrozen():
    gcanvas = drawn_canvas(triangle())
    before = gcanvas.to_canvas([1, 1])
    gcanvas.draw(triangle().pinned_at(2, [100, 100]), DARK, LABELS_OFF)
    assert gcanvas.to_canvas([1, 1]) != pytest.approx(before)


def test_pinned_dots_have_the_pinned_color():
    canvas = draw(triangle().pinned_at(1, [2, 0]), DARK, LABELS_OFF)
    fills = [options['fill'] for kind, options in canvas.items if kind == 'oval']
    assert fills == [DARK['node_fill'], DARK['pinned'], DARK['node_fill']]


@pytest.mark.parametrize('labels', [LABELS_ABOVE, LABELS_BELOW])
def test_pinned_label_boxes_have_the_pinned_outline(labels):
    canvas = draw(triangle().pinned_at(1, [2, 0]), LIGHT, labels)
    outlines = [options['outline'] for kind, options in canvas.items if kind == 'rectangle']
    assert outlines == [LIGHT['node_outline'], LIGHT['pinned'], LIGHT['node_outline']]


def test_model_with_knob_sets_the_field_from_text_and_keeps_the_other():
    assert model_with_knob(BALLOON, 'strength', ' 1.5 ') == (
        PowerLaw(strength=1.5, exponent=BALLOON.exponent), None)
    assert model_with_knob(BALLOON, 'exponent', '2.5') == (
        PowerLaw(strength=BALLOON.strength, exponent=2.5), None)


@pytest.mark.parametrize('field, text, message', [
    ('strength', '', 'strength: a number, 0 or more'),
    ('strength', 'abc', 'strength: a number, 0 or more'),
    ('strength', '-1', 'strength: a number, 0 or more'),
    ('strength', 'nan', 'strength: a number, 0 or more'),
    ('exponent', '', 'exponent: a number above 0'),
    ('exponent', 'abc', 'exponent: a number above 0'),
    ('exponent', '0', 'exponent: a number above 0'),
    ('exponent', 'nan', 'exponent: a number above 0'),
])
def test_model_with_knob_rejects_text_breaking_the_constraint(field, text, message):
    assert model_with_knob(BALLOON, field, text) == (None, message)


def test_preset_name_names_the_presets():
    assert preset_name(BALLOON) == "Balloon (1/d)"
    assert preset_name(DENSE) == "Dense (1/d²)"


def test_preset_name_is_custom_for_other_models():
    assert preset_name(PowerLaw(strength=2, exponent=1.5)) == "Custom"
