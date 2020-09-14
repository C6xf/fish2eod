import matplotlib.tri as tri
import numpy as np
import pytest
import shapely.geometry as shp

from fish2eod.geometry.primitives import Polygon, Rectangle
from fish2eod.mesh.domain import mark_domains
from fish2eod.mesh.mesh import create_mesh
from fish2eod.mesh.model_geometry import ModelGeometry


@pytest.mark.quick
def test_domain_single():
    mg = ModelGeometry()
    bg = Rectangle.from_center([0, 0], 1)
    mg.add_domain("bg", bg)
    mesh = create_mesh(mg)
    domains = mark_domains(mesh, mg).array()

    assert np.all(domains == mg.domain_names["bg"])


@pytest.mark.quick
def test_domain_nested():
    mg = ModelGeometry()
    bg = Rectangle.from_center([0, 0], 1)
    r = Rectangle.from_center([0, 0], 0.5)
    mg.add_domain("bg", bg)
    mg.add_domain("r", r)
    mesh = create_mesh(mg)
    domains = mark_domains(mesh, mg).array()

    checker = r.expand(1e-3)._shapely_representation

    xy = mesh.coordinates()
    tri_f = tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())
    triangles = tri_f.get_masked_triangles()
    verts = np.stack((tri_f.x[triangles], tri_f.y[triangles]), axis=-1)

    for ix, (v1, v2, v3) in enumerate(verts):
        if (
            checker.contains(shp.Point(v1))
            and checker.contains(shp.Point(v2))
            and checker.contains(shp.Point(v3))
        ):
            assert domains[ix] == mg.domain_names["r"]
        else:
            assert domains[ix] == mg.domain_names["bg"]


@pytest.mark.quick
def test_domains_polygon():
    mg = ModelGeometry()
    bg = Rectangle.from_center([0, 0], 1)
    r = Polygon([-0.25, 0.25, 0.25, -0.25], [-0.25, -0.25, 0.25, 0.25])
    mg.add_domain("bg", bg)
    mg.add_domain("r", r)
    mesh = create_mesh(mg)
    domains = mark_domains(mesh, mg).array()

    checker = r.expand(1e-3)._shapely_representation

    xy = mesh.coordinates()
    tri_f = tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())
    triangles = tri_f.get_masked_triangles()
    verts = np.stack((tri_f.x[triangles], tri_f.y[triangles]), axis=-1)

    for ix, (v1, v2, v3) in enumerate(verts):
        if (
            checker.contains(shp.Point(v1))
            and checker.contains(shp.Point(v2))
            and checker.contains(shp.Point(v3))
        ):
            assert domains[ix] == mg.domain_names["r"]
        else:
            assert domains[ix] == mg.domain_names["bg"]
