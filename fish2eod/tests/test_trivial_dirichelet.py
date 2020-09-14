import numpy as np

from fish2eod.geometry.primitives import Rectangle
from fish2eod.models import QESModel
from fish2eod import BoundaryCondition


class BasicModel(QESModel):
    def create_geometry(self, **kwargs) -> None:
        background = Rectangle([-0.5, -0.5], 1, 1)
        self.model_geometry.add_domain("bkg", background, sigma=1)


def test_empty_model():
    model = BasicModel()
    model.compile()
    model.solve()

    assert np.isclose(model._fem_solution(-0.5, -0.5), 0)
    assert np.isclose(model._fem_solution(-0.5, 0.5), 0)
    assert np.isclose(model._fem_solution(0.5, -0.5), 0)
    assert np.isclose(model._fem_solution(0.5, 0.5), 0)


def test_grounded_extremity():
    class BasicModel2(BasicModel):
        def get_voltage_sources(self):
            return [BoundaryCondition(0, self._EXTERNAL_BOUNDARY)]

    model = BasicModel2()
    model.compile()
    model.solve()

    assert np.isclose(model._fem_solution(-0.5, -0.5), 0)
    assert np.isclose(model._fem_solution(-0.5, 0.5), 0)
    assert np.isclose(model._fem_solution(0.5, -0.5), 0)
    assert np.isclose(model._fem_solution(0.5, 0.5), 0)


def test_constant_extremity():
    class BasicModel2(BasicModel):
        def get_dirichlet_conditions(self, **kwargs):
            return [BoundaryCondition(1, self._EXTERNAL_BOUNDARY)]

    model = BasicModel2()
    model.compile()
    model.solve()

    assert np.isclose(model._fem_solution(-0.5, -0.5), 1)
    assert np.isclose(model._fem_solution(-0.5, 0.5), 1)
    assert np.isclose(model._fem_solution(0.5, -0.5), 1)
    assert np.isclose(model._fem_solution(0.5, 0.5), 1)


def test_interior_boundary():
    class TestModel2(QESModel):
        def create_geometry(self, **kwargs) -> None:
            background = Rectangle([-0.5, -0.5], 1, 1)
            square = Rectangle([-0.25, -0.25], 0.5, 0.5)

            self.model_geometry.add_domain("bkg", background, sigma=1)
            self.model_geometry.add_domain("sq", square, sigma=1)

        def get_dirichlet_conditions(self, **kwargs):
            return [BoundaryCondition(1, self._EXTERNAL_BOUNDARY)]

    model = TestModel2()
    model.compile()
    model.solve()

    assert np.isclose(model._fem_solution(-0.25, -0.25), 1)
    assert np.isclose(model._fem_solution(-0.25, 0.25), 1)
    assert np.isclose(model._fem_solution(0.25, -0.25), 1)
    assert np.isclose(model._fem_solution(0.25, 0.25), 1)


def test_logical_dirichelet():
    class TestModel2(QESModel):
        def create_geometry(self, **kwargs) -> None:
            background = Rectangle([-0.5, -0.5], 1, 1)
            square = Rectangle([-0.25, -0.25], 0.5, 0.5)

            self.model_geometry.add_domain("bkg", background, sigma=1)
            self.model_geometry.add_domain("sq", square, sigma=1)

        def get_dirichlet_conditions(self, **kwargs):
            return [
                BoundaryCondition("1*x[0]<0", self.model_geometry["sq"])
            ]  # todo optional not list

    model = TestModel2()
    model.compile()
    model.solve()

    assert np.isclose(model._fem_solution(-0.25, -0.25), 1)
    assert np.isclose(model._fem_solution(-0.25, 0.25), 1)
    assert np.isclose(model._fem_solution(0.25, -0.25), 0)
    assert np.isclose(model._fem_solution(0.25, 0.25), 0)
