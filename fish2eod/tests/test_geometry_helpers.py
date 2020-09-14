import numpy as np
import pytest
import shapely.geometry as shp
from hypothesis import given
from hypothesis.strategies import floats, integers

from fish2eod.geometry.operations import (
    cut_line_between_fractions,
    extend_line,
    filter_line_length,
    measure_and_interpolate,
    parallel_curves,
    uniform_spline_interpolation,
)
from fish2eod.tests.testing_helpers import same_len_lists


@pytest.mark.quick
@given(same_len_lists(), integers(min_value=0, max_value=1000))
def test_measure_and_interpolate(xy, additional):
    x, y = xy
    m = len(x) + additional
    d, xi, yi = measure_and_interpolate(x, y, m)

    assert len(xi) >= m
    assert len(yi) >= m

    d_true = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    if d_true == 0:
        err = d_true - np.sum(d)
    else:
        err = abs(d_true - np.sum(d)) / d_true

    assert np.round(d_true, 5) >= np.round(np.sum(d), 5)
    assert err < 1e-5


@pytest.mark.quick
@given(same_len_lists(), integers(min_value=0, max_value=100))
def test_uniform_spline_interpolation(xy, additional):
    x, y = xy
    n = len(x) + additional
    xi, yi = uniform_spline_interpolation(x, y, n)

    total_length = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    assert np.allclose(np.diff(xi), total_length / (n - 1), 3)
    assert np.allclose(np.diff(yi), total_length / (n - 1), 3)
    assert len(xi) == n == len(yi)


@pytest.mark.quick
def test_uniform_spline_sine():
    x = np.linspace(0, 10, 20)
    y = np.exp(x)

    n = 40
    xi, yi = uniform_spline_interpolation(x, y, n)

    target_distance, *_ = measure_and_interpolate(x, y, 100)
    target_distance = sum(target_distance) / (n - 1)

    measured_distance = np.sqrt(np.diff(xi) ** 2 + np.diff(yi) ** 2)

    assert (
        np.std(measured_distance) / np.mean(measured_distance) <= 0.01
    )  # < 1% variance
    assert (
        abs(np.mean(measured_distance) - target_distance) / target_distance
    ), 0.01  # <1%error


@pytest.mark.quick
def test_cut_line_between_fractions():
    line = shp.LineString(zip([0, 1], [0, 0]))

    cut_line = cut_line_between_fractions(line, 1 / 3, 2 / 3)

    x_t, y_t = np.array(cut_line.coords).T

    d = np.sqrt((max(x_t) - min(x_t)) ** 2 - (max(y_t) - min(y_t)) ** 2)

    assert np.isclose(d, 1 / 3)
    assert np.isclose(max(x_t), 2 / 3)
    assert np.isclose(min(x_t), 1 / 3)


@pytest.mark.quick
@pytest.mark.parametrize(
    ("f1", "f2", "good_bad"),
    [
        (0.6, 0.7, True),
        (0.6, 0.75, False),
        (0.6, 0.707, False),
        (0.605, 0.7, False),
        (0.4, 0.6, False),
        (1.4, 1.5, False),
    ],
)
def test_filter_line_length(f1, f2, good_bad):
    original = shp.LineString(zip([0, 1], [0, 0]))
    slice = shp.LineString(zip([f1, f2], [0, 0]))
    assert filter_line_length(original, 0.6, 0.7, slice) == good_bad


@pytest.mark.quick
def test_offset_curve_straight_line():
    x = [0, 1]
    y = [0, 0]

    offset_dict = parallel_curves(x, y, 1)

    x_inner = offset_dict["x_inner"]
    x_outer = offset_dict["x_outer"]

    y_inner = offset_dict["y_inner"]
    y_outer = offset_dict["y_outer"]

    assert np.allclose(x, x_inner)
    assert np.allclose(x, x_outer)
    assert np.allclose(y_inner, 1)
    assert np.allclose(y_outer, -1)


@pytest.mark.quick
@given(same_len_lists(), floats(min_value=0, max_value=1))
def test_extend_curve(l, f):
    x, y = l

    e_x, e_y = extend_line(x, y, f)

    length_1 = np.sqrt(np.sum(np.diff(x) ** 2 + np.diff(y) ** 2))
    length_2 = np.sqrt(np.sum(np.diff(e_x) ** 2 + np.diff(e_y) ** 2))

    assert np.round(length_1, 6) <= np.round(length_2, 6)
