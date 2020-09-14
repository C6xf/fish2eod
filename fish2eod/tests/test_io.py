from tempfile import TemporaryDirectory
from typing import List

import numpy as np

from fish2eod import BoundaryCondition
from fish2eod.geometry.primitives import Rectangle, Circle
from fish2eod.models import QESModel
from fish2eod.xdmf.load import load_from_file
from fish2eod.xdmf.save import Saver


def test_write_read():
    class mod(QESModel):
        def create_geometry(self, **kwargs):
            sq = Rectangle.from_center([0, 0], 3)
            circ = Circle([0, 0], 0.5)

            self.model_geometry.add_domain("bg", sq, sigma=1)
            self.model_geometry.add_domain("act", circ, sigma=2)

        def get_neumann_conditions(self, **kwargs) -> List[BoundaryCondition]:
            return [BoundaryCondition(value=1, label=self.model_geometry["act"])]

    model = mod()
    model.compile()
    model.solve()

    with TemporaryDirectory() as tmp_dir:
        s = Saver("test", tmp_dir)
        s.save_model(model)

        load_structure = load_from_file(tmp_dir + "/test/")
        check_solution(model, load_structure)
        check_sigma(model, load_structure)
        check_domain(model, load_structure)
        check_boundary(model, load_structure)
        check_outline(model, load_structure)


def check_solution(model, l):
    true_topology = model.topology_2d
    true_geometry = model.topology_coordinates
    true_sol = model._fem_solution.compute_vertex_values()

    top, geom, sol = l["solution"].load_data()

    check_parts(true_sol, sol, true_topology, top, true_geometry, geom)


def check_sigma(model, l):
    true_topology = model.topology_2d
    true_geometry = model.topology_coordinates
    true_sol = model.get_property("sigma").compute_vertex_values(model.mesh)

    top, geom, sol = l["sigma"].load_data()

    check_parts(true_sol, sol, true_topology, top, true_geometry, geom)


def check_domain(model, l):
    true_topology = model.topology_2d
    true_geometry = model.topology_coordinates
    true_sol = model.domains.array()

    top, geom, sol = l["domain"].load_data()

    check_parts(true_sol, sol, true_topology, top, true_geometry, geom)


def check_boundary(model, l):
    true_topology = model.topology_1d
    true_geometry = model.topology_coordinates
    true_sol = model.boundaries.array()

    top, geom, sol = l["boundary"].load_data()

    check_parts(true_sol, sol, true_topology, top, true_geometry, geom)


def check_outline(model, l):
    true_topology = model.topology_1d
    true_geometry = model.topology_coordinates
    true_sol = model.outline.array()

    top, geom, sol = l["outline"].load_data()

    check_parts(true_sol, sol, true_topology, top, true_geometry, geom)


def check_parts(true_sol, sol, true_top, top, true_geom, geom):
    assert np.all(true_top == top)
    assert np.linalg.norm(true_geom - geom) < 1e-15
    assert np.linalg.norm(true_sol - sol) < 1e-15
