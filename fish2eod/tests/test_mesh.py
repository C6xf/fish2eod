import numpy as np
from hypothesis import assume, given, settings

from fish2eod.geometry.primitives import Rectangle
from fish2eod.mesh.mesh import create_mesh
from fish2eod.mesh.model_geometry import ModelGeometry
from fish2eod.tests.testing_helpers import rectangle_inputs


@given(*rectangle_inputs)
@settings(deadline=None, max_examples=5)
def test_square_mesh(corner_x, corner_y, width, height, _):
    assume(width > 1e-3)
    assume(height > 1e-3)  # todo limit aspect ratio
    bg = Rectangle.from_center([0, 0], 2e6, 2e6)
    r = Rectangle([corner_x, corner_y], width, height)

    mg = ModelGeometry(allow_overlaps=True)
    mg.add_domain("bg", bg)
    mg.add_domain("r", r)

    points = [
        (-1e6, -1e6),
        (1e-6, -1e6),
        (1e6, 1e6),
        (-1e6, 1e6),
        (corner_x, corner_y),
        (corner_x + width, corner_y),
        (corner_x + width, corner_y + height),
        (corner_x, corner_y + height),
    ]

    mesh = create_mesh(mg)
    coordinates = mesh.coordinates()
    for p in points:
        err = coordinates - p
        assert np.any(np.isclose(err, 0, atol=1e-6))
