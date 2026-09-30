import pytest

from layout import EDGE_LENGTH
from themes import DARK, LIGHT, strain_color

# channel values chosen so that the halfway colors are whole numbers
THEME = dict(compressed='#0000fe', ideal='#808080', stretched='#fe0000')


@pytest.mark.parametrize('theme', [DARK, LIGHT, THEME])
def test_edge_of_ideal_length_has_the_ideal_color(theme):
    assert strain_color(EDGE_LENGTH, theme) == theme['ideal']


@pytest.mark.parametrize('theme', [DARK, LIGHT, THEME])
@pytest.mark.parametrize('ideal_lengths', [4, 5, 1000])
def test_edge_stretched_to_4_times_or_more_has_the_stretched_color(theme, ideal_lengths):
    assert strain_color(EDGE_LENGTH * ideal_lengths, theme) == theme['stretched']


@pytest.mark.parametrize('theme', [DARK, LIGHT, THEME])
@pytest.mark.parametrize('ideal_lengths', [1/4, 1/5, 0])
def test_edge_compressed_to_a_quarter_or_less_has_the_compressed_color(theme, ideal_lengths):
    assert strain_color(EDGE_LENGTH * ideal_lengths, theme) == theme['compressed']


def test_edge_of_double_length_is_halfway_to_the_stretched_color():
    assert strain_color(EDGE_LENGTH * 2, THEME) == '#bf4040'


def test_edge_of_half_length_is_halfway_to_the_compressed_color():
    assert strain_color(EDGE_LENGTH / 2, THEME) == '#4040bf'

