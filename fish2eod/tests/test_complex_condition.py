import pytest

from fish2eod.geometry.primitives import Circle, Rectangle
from fish2eod.models import QESModel
from fish2eod import BoundaryCondition
from fish2eod.tests.testing_helpers import compare_comsol


@pytest.mark.integration
def test_complex_bc():
    class TestModel(QESModel):
        def create_geometry(self, **kwargs) -> None:
            background = Rectangle([-0.5, -0.5], 1, 1, lcar=0.01)
            source = Circle(
                [0, 0], 0.2, lcar=0.01
            )  # particularly sensitive to mesh size

            self.model_geometry.add_domain("bkg", background, sigma=1)
            self.model_geometry.add_domain("source", source, sigma=4)

        def get_neumann_conditions(self, **kwargs):
            return [
                BoundaryCondition("atan2(x[0], x[1])", self.model_geometry["source"])
            ]

        def get_dirichlet_conditions(self, **kwargs):
            return [BoundaryCondition(0, self._EXTERNAL_BOUNDARY)]

    model = TestModel()
    model.compile()
    model.solve()

    assert (
        compare_comsol(model._fem_solution, "complex_bc") <= 0.025
    )  # extra half point for mesh sensitivity


@pytest.mark.integration
def test_complex_conductance():
    class TestModel(QESModel):
        def create_geometry(self, **kwargs) -> None:
            background = Rectangle([-0.5, -0.5], 1, 1)
            source = Circle([0, 0], 0.2)

            self.model_geometry.add_domain("bkg", background, sigma=lambda x, y: 1 + x)
            self.model_geometry.add_domain("source", source, sigma=lambda x, y: 1 + y)

        def get_neumann_conditions(self, **kwargs):
            return [BoundaryCondition("1", self.model_geometry["source"])]

        def get_dirichlet_conditions(self, **kwargs):
            return [BoundaryCondition(0, self._EXTERNAL_BOUNDARY)]

    model = TestModel()
    model.compile()
    model.solve()

    assert compare_comsol(model._fem_solution, "complex_cond") <= 0.02


@pytest.mark.integration
def test_complex_geometry():
    class TestModel(QESModel):
        def create_geometry(self, **kwargs) -> None:
            background = Rectangle([-0.5, -0.5], 1, 1)
            ground = Circle([-0.4, 0.2], 0.02)
            bottom_source = Circle([0, -0.2], 0.1)
            top_source = Circle([0.2, 0.3], 0.05)

            self.model_geometry.add_domain("bkg", background, sigma=1)
            self.model_geometry.add_domain("ground", ground, sigma=2)
            self.model_geometry.add_domain("bottom_source", bottom_source, sigma=3)
            self.model_geometry.add_domain("top_source", top_source, sigma=4)

        def get_neumann_conditions(self, **kwargs):
            return [
                BoundaryCondition(1, self.model_geometry["bottom_source"]),
                BoundaryCondition(-2, self.model_geometry["top_source"]),
            ]

        def get_dirichlet_conditions(self, **kwargs):
            return [BoundaryCondition(0, self.model_geometry["ground"])]

    model = TestModel()
    model.compile()
    model.solve()

    assert compare_comsol(model._fem_solution, "multi_geometry") <= 0.02
