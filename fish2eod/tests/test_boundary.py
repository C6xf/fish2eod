"""Test components related to boundaries and labeling them."""
import dolfin as df
import numpy as np
import pytest

from fish2eod import BoundaryCondition
from fish2eod.helpers.dolfin_tools import bc_to_expression
from fish2eod.geometry.primitives import Rectangle
from fish2eod.mesh.boundary import (
    boundary_iterator,
    mark_boundaries,
    mark_edge_by_rules,
)
from fish2eod.helpers.type_helpers import BOUNDARY_MARKER
from fish2eod.mesh.domain import mark_domains
from fish2eod.mesh.mesh import create_mesh
from fish2eod.mesh.model_geometry import ModelGeometry
from fish2eod.tests.testing_helpers import FullModel


def simple_domain(size=15):
    """Construct a simple meshfunction representing domains.

    Simple is defined as 0 defined on the left half (x < 0.5) and 1 on the right side (x > 0.5)

    :param size: Resolution of the mesh
    :return: Domain function
    """
    mesh = df.UnitSquareMesh(size, size)
    domain = df.MeshFunction("size_t", mesh, 2)

    # make MeshFunction with right = 1 and left = 0
    for c in df.cells(mesh):
        if c.midpoint()[0] > 0.5:
            domain[c] = 1
    return domain


def complex_domain(size=15):
    """Construct a complex meshfunction representing domains.

    Simple is defined as 0 defined on the left half (x < 0.5)
    If on the right half the function is 1 if y < 0.5 and 2 if y > 0.5

    :param size: Resolution of the mesh
    :return: Domain function
    """
    mesh = df.UnitSquareMesh(size, size)
    domain = df.MeshFunction("size_t", mesh, 2)

    # make MeshFunction with left = 0 and right = 1 if lower and 2 if upper
    for c in df.cells(mesh):
        if c.midpoint()[0] > 0.5:
            if c.midpoint()[1] > 0.5:
                domain[c] = 2
            else:
                domain[c] = 1
    return domain


@pytest.mark.quick
def test_boundary_iterator():
    """Test iterating over a domain meshfunction.

    The simple meshfunction has a single boundary at x = 0.5. The bordering domains are values 0, 1
    """
    domains = simple_domain()
    for edge, neighbouring_domains in boundary_iterator(domains):
        assert edge.midpoint()[0] == 0.5  # TODO float comparison
        assert neighbouring_domains == (0, 1)


@pytest.mark.quick
def test_complicated_boundary_iterator():
    r"""Test iterating over a domain meshfunction.

    The complex meshfunction has 2 boundaries

    There is a boundary at x=0.5
    There is a boundary at y=0.5 for x\in [0.5, 1]

    The first boundary has mating domains of 0,1 or 0,2 depending on where it is
    The second has the domains 1, 2
    """
    domains = complex_domain()
    for edge, neighbouring_domains in boundary_iterator(domains):
        if edge.midpoint()[0] == 0.5:  # todo float comparison
            assert neighbouring_domains == (0, 1) or neighbouring_domains == (0, 2)
        else:
            assert edge.midpoint()[1] == 0.5  # todo float comparison
            assert neighbouring_domains == (1, 2)


@pytest.mark.quick
def test_mark_edge_by_rules():
    """Test marking boundaries with a rule.

    The rule used is only when the mating domains are 1 and 2. If this is valid then the boundary value is
    100 if x < 0.75 otherwise its 200
    """
    bm = lambda b1, b2: (b1 == 1 and b2 == 2, 100)

    mf = complex_domain()
    for edge, neighbouring_domains in boundary_iterator(mf):
        label = mark_edge_by_rules(neighbouring_domains, (bm,))
        if neighbouring_domains == (1, 2):
            assert label == 100
        else:  # Boundary was not labeled so None
            assert label is None


def make_square_in_square_in_square(complex=False):
    """Make a complex nested square structure.

    Default behaviour is a nested square
    Complex behaviour will nest two other squares inside the center square

    :param complex: If the complex regime will be used
    :return: Model geometry of the nested squares
    """
    sq1 = Rectangle.from_center([0, 0], 6)
    sq2 = Rectangle.from_center([0, 0], 4)
    mg = ModelGeometry(allow_overlaps=True)
    mg.add_domain("bg", sq1)
    mg.add_domain("fg", sq2)
    if complex:
        sq3 = Rectangle.from_center([0, 0], 2)
        sq4 = Rectangle.from_center([0, 0], 0.5)  # not labeled

        mg.add_domain("fg2", sq3)
        mg.add_domain("fake", sq4)
    mesh = create_mesh(mg)
    domains = mark_domains(mesh, mg)

    return domains, mg


@pytest.mark.quick
def test_mark_boundaries():
    """Test the marking of boundaries.

    Mark the boundaries on the complex 4-nested square
    """
    domains, mg = make_square_in_square_in_square(complex=True)
    bm = lambda b1, b2: (
        b1 == mg.domain_names["fg"] and b2 == mg.domain_names["fg2"],
        200,
    )  # mark boundary between fg and fg2 as 200

    boundaries, outline = mark_boundaries(domains, mg, (bm,), 9999)

    # manually check edges (facets)
    for f in df.facets(boundaries.mesh()):
        mp = f.midpoint()

        checked = False  # todo refactor checks into fcns
        if np.isclose(abs(mp[0]), 3) or np.isclose(abs(mp[1]), 3):
            assert boundaries[f] == 9999
            checked = True

        if (
            np.isclose(abs(mp[0]), 2)
            and abs(mp[1]) <= 2
            or np.isclose(abs(mp[1]), 2)
            and abs(mp[0]) <= 2
        ):
            assert boundaries[f] == 1
            checked = True

        if (
            np.isclose(abs(mp[0]), 1)
            and abs(mp[1]) <= 1
            or np.isclose(abs(mp[1]), 1)
            and abs(mp[0]) <= 1
        ):
            assert boundaries[f] == 200
            checked = True

        if (
            np.isclose(abs(mp[0]), 0.25)
            and abs(mp[1]) <= 0.25
            or np.isclose(abs(mp[1]), 0.25)
            and abs(mp[0]) <= 0.25
        ):
            assert (
                boundaries[f] == 0
            )  # explicitly check this is bg todo is this needed?
            checked = True

        if not checked:
            assert boundaries[f] == 0


@pytest.mark.parametrize("val", [-100, 0, 100, 1e6])
def test_boundary_condition_representation(val):
    """Test implementation of boundary conditions and the various methods are equivalent.

    :param val: Value to set bc
    """
    b1 = bc_to_expression(BoundaryCondition(str(val), 1))  # string def
    b2 = bc_to_expression(BoundaryCondition(val, 1))  # numeric def
    b3 = bc_to_expression(BoundaryCondition(df.Expression(str(val), degree=2), 1))

    model = FullModel()
    model.compile()

    b1 = b1.compute_vertex_values(model.mesh)
    b2 = b2.compute_vertex_values(model.mesh)
    b3 = b3.compute_vertex_values(model.mesh)

    assert np.all(b1 == b2)
    assert np.all(b1 == b3)

    assert np.allclose(b1[2:], val)  # for some reason not defined on nodes for source?
