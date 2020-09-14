import dolfin as df
import numpy as np
import pytest
import shapely.geometry as shp

from fish2eod.properties import Property, SplineExpression
from fish2eod.tests.test_boundary import make_square_in_square_in_square


@pytest.mark.quick
def test_property():
    domains, mg = make_square_in_square_in_square(complex=True)
    p = Property(domains, {0: 0, 1: 100, 2: 200, 3: 300})

    class Test(object):  # todo mock this in
        index = 0

    a = Test()
    v = [0, 15]
    for c in df.cells(domains.mesh()):
        a.index = c.index()
        p.eval_cell(v, tuple(c.midpoint())[:-1], a)

        assert v[0] == 100 * domains[c]


@pytest.mark.quick
def test_functional_expression():
    domains, mg = make_square_in_square_in_square(complex=True)
    p = Property(domains, {k: lambda x, y: x * y for k in range(4)})

    class Test(object):  # todo mock this in
        index = 0

    a = Test()
    v = [0, 15]
    for c in df.cells(domains.mesh()):
        a.index = c.index()
        p.eval_cell(v, tuple(c.midpoint())[:-1], a)

        assert v[0] == c.midpoint()[0] * c.midpoint()[1]


@pytest.mark.quick
def test_spline_property():
    domains, mg = make_square_in_square_in_square(complex=True)
    line = shp.LineString([shp.Point(-3, 0), shp.Point(3, 0)])
    p = SplineExpression(line, lambda x: x)

    v = [0, 15]
    for c in df.cells(domains.mesh()):
        p.eval(v, tuple(c.midpoint())[:-1])

        assert np.isclose(v[0], (c.midpoint()[0] + 3) / 6)
