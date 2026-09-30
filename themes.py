# colors of the drawing: the themes and the coloring of edges by their strain

import math

DARK = dict(
    background='#000000',
    node_fill='#d4d4d4',
    node_outline='#ffffff',
    label_text='#f0f0f0',
    label_box='#2b2b2b',
    compressed='#3b82f6',
    ideal='#8a8a8a',
    stretched='#ef4444',
)

LIGHT = dict(
    background='#ffffff',
    node_fill='#3a3a3a',
    node_outline='#000000',
    label_text='#1a1a1a',
    label_box='#fff7d6',
    compressed='#1d4ed8',
    ideal='#9a9a9a',
    stretched='#dc2626',
)

# strain is log2(length / reference), edges 4 times longer or shorter than
# the reference get the full stretched or compressed color
MAX_STRAIN = 2


def rgb(color):
    return [int(color[i:i+2], 16) for i in (1, 3, 5)]


def mix(color1, color2, ratio):
    "color at ratio of the way from color1 to color2"
    return '#%02x%02x%02x' % tuple(
        round(c1 + (c2 - c1) * ratio) for c1, c2 in zip(rgb(color1), rgb(color2)))


def strain_color(length, reference, theme):
    if length <= 0:
        strain = -MAX_STRAIN
    else:
        strain = max(-MAX_STRAIN, min(MAX_STRAIN, math.log2(length / reference)))
    end_color = theme['stretched'] if strain > 0 else theme['compressed']
    return mix(theme['ideal'], end_color, abs(strain) / MAX_STRAIN)
