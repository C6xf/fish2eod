import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.strategies import floats

from fish2eod.geometry.primitives import Circle, Polygon, Rectangle
from fish2eod.tests.testing_helpers import (
    circle_inputs,
    rectangle_inputs,
    rotation_inputs,
    translation_inputs,
)


@pytest.mark.quick
@given(*rectangle_inputs)
def test_create_rectangle(corner_x, corner_y, width, height, use_height):
    if use_height:
        r = Rectangle([corner_x, corner_y], width, height)
    else:
        r = Rectangle([corner_x, corner_y], width)

    assert np.isclose(r.center[0], corner_x + width / 2)
    if use_height:
        assert np.isclose(r.center[1], corner_y + height / 2)
    else:
        assert np.isclose(r.center[1], corner_y + width / 2)
    assert np.isclose(width, r.width)
    if use_height:
        assert np.isclose(height, r.height)

    if use_height:
        check_represented_area(r, width * height)
    else:
        check_represented_area(r, width * width)


@pytest.mark.quick
@given(*rectangle_inputs)
def test_create_rectangle_center(center_x, center_y, width, height, use_height):
    if use_height:
        r = Rectangle.from_center([center_x, center_y], width, height)
    else:
        r = Rectangle.from_center([center_x, center_y], width)

    assert np.isclose(r.center[0], center_x)
    assert np.isclose(r.center[1], center_y)
    assert np.isclose(width, r.width)
    if use_height:
        assert np.isclose(height, r.height)

    if use_height:
        check_represented_area(r, width * height)
    else:
        check_represented_area(r, width * width)


@pytest.mark.quick
@given(*circle_inputs)
def test_create_circle(center_x, center_y, r):
    c = Circle([center_x, center_y], r)

    # Check center and radius
    assert np.isclose(center_x, c.center[0])
    assert np.isclose(center_y, c.center[1])
    assert np.isclose(r, c.radius)

    check_represented_area(c, np.pi * r * r)


@pytest.mark.quick
@given(*translation_inputs)
def test_translate(dx, dy):
    c = Circle([0, 0], 1)
    re_done = c.translate(dx=dx, dy=dy).translate(dx=-dx, dy=-dy)

    check_overlap_equal(c, re_done)


@pytest.mark.quick
@given(*rotation_inputs)
def test_rotate(angle, degrees, center):
    r = Rectangle.from_center([0, 0], 5, 5)
    re_done = r.rotate(angle, degrees, center).rotate(-angle, degrees, center)

    check_overlap_equal(r, re_done)


@pytest.mark.quick
@given(
    *circle_inputs,
    floats(min_value=-1e6, max_value=1e6),
    floats(min_value=-1e6, max_value=1e6)
)
def test_inside_circle(center_x, center_y, r, px, py):
    c = Circle([center_x, center_y], r)
    expr = (
        c.inside(None, None, buffer=0).replace("x[0]", str(px)).replace("x[1]", str(py))
    )

    if (center_x - px) ** 2 + (center_y - py) ** 2 <= r ** 2:
        assert eval(expr)
    else:
        assert not eval(expr)


@pytest.mark.quick
@given(*circle_inputs, floats(min_value=1e-6, max_value=1e6))
def test_offset_circle(center_x, center_y, r, o):
    assume(abs(center_x) + r + o <= 2e6)
    assume(abs(center_y) + r + o <= 2e6)

    c = Circle([center_x, center_y], r)
    new_c = c.expand(o)

    theory_overlap = (np.pi * (r + o) ** 2 - np.pi * r ** 2) / (np.pi * (r + o) ** 2)
    measured_overlap = get_overlap(new_c, c)

    err = abs(theory_overlap - measured_overlap) / theory_overlap
    assert err < 0.005


@pytest.mark.quick
def test_polygon():
    x = [0, 1, 1]  # triangle
    y = [0, 0, 1]
    p = Polygon(x, y)
    check_represented_area(p, 1 / 2 * 1 * 1)
    x_in, y_in = zip(*p._shapely_representation.exterior.coords)

    assert np.all(np.array(x) == x_in[:-1])
    assert np.all(np.array(y) == y_in[:-1])


@pytest.mark.quick
@given(
    *rectangle_inputs[:-1],
    floats(min_value=-1e6, max_value=1e6),
    floats(min_value=-1e6, max_value=1e6)
)
def test_inside_square(corner_x, corner_y, w, h, px, py):
    r = Rectangle([corner_x, corner_y], w, h)
    expr = (
        r.inside(None, None, buffer=0)
        .replace("x[0]", str(px))
        .replace("x[1]", str(py))
        .replace("&&", "and")
    )

    if corner_x < px < corner_x + w and corner_y < py < corner_y + h:
        assert eval(expr)
    else:
        assert not eval(expr)


@pytest.mark.quick
def test_invalid_polygon():
    with pytest.raises(ValueError):
        Polygon([0, 1, 0, 1], [0, 1, 1, 0])


def check_overlap_equal(shape1, shape2):
    err = get_overlap(shape1, shape2)
    assert err < 1e-9


def check_represented_area(s, true_area):
    shapely_area = s._shapely_representation.area

    err = (true_area - shapely_area) / shapely_area
    assert err < 0.01  # tests shapely representation < 1% err


def get_overlap(shape1, shape2):
    true_area = shape1._shapely_representation.area

    overlapping_area = shape1._shapely_representation.intersection(
        shape2._shapely_representation
    ).area
    return (true_area - overlapping_area) / true_area
