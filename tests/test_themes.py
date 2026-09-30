import pytest

from themes import DARK, LIGHT, strain_color

REFERENCE = 3

# channel values chosen so that the halfway colors are whole numbers
THEME = dict(compressed='#0000fe', ideal='#808080', stretched='#fe0000')


@pytest.mark.parametrize('theme', [DARK, LIGHT, THEME])
def test_edge_of_reference_length_has_the_ideal_color(theme):
    assert strain_color(REFERENCE, REFERENCE, theme) == theme['ideal']


@pytest.mark.parametrize('theme', [DARK, LIGHT, THEME])
@pytest.mark.parametrize('reference_lengths', [4, 5, 1000])
def test_edge_stretched_to_4_times_or_more_has_the_stretched_color(theme, reference_lengths):
    assert strain_color(REFERENCE * reference_lengths, REFERENCE, theme) == theme['stretched']


@pytest.mark.parametrize('theme', [DARK, LIGHT, THEME])
@pytest.mark.parametrize('reference_lengths', [1/4, 1/5, 0])
def test_edge_compressed_to_a_quarter_or_less_has_the_compressed_color(theme, reference_lengths):
    assert strain_color(REFERENCE * reference_lengths, REFERENCE, theme) == theme['compressed']


def test_edge_of_double_length_is_halfway_to_the_stretched_color():
    assert strain_color(REFERENCE * 2, REFERENCE, THEME) == '#bf4040'


def test_edge_of_half_length_is_halfway_to_the_compressed_color():
    assert strain_color(REFERENCE / 2, REFERENCE, THEME) == '#4040bf'

