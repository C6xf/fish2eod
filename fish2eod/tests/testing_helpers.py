"""Helper module for running tests.

Defines default classes and convenience functions.
"""
from os.path import abspath, join, split

import numpy as np
import pandas as pd
from hypothesis import strategies as st
from hypothesis.strategies import booleans, floats, tuples
from scipy.interpolate import griddata

from fish2eod.geometry.primitives import Circle, Rectangle
from fish2eod.models import QESModel
from fish2eod import BoundaryCondition


class FullModel(QESModel):
    """Stock complex models with complex sources and sinks for testing."""

    BG_ID = 0
    BG_COND = 1

    SOURCE_ID_1 = 1
    SOURCE_ID_2 = 2

    SOURCE_LOCATION_1 = [0.2, 0.2]
    SOURCE_LOCATION_2 = [0.4, 0.4]

    SOURCE_RADIUS = 0.1

    SOURCE_AMPLITUDE_1 = 1
    SOURCE_AMPLITUDE_2 = -SOURCE_AMPLITUDE_1

    def __init__(self, source_1=(0.2, 0.2), source_2=(0.8, 0.8), amp_1=1):
        super().__init__()
        self.SOURCE_LOCATION_1 = source_1
        self.SOURCE_LOCATION_2 = source_2

        self.SOURCE_AMPLITUDE_1 = amp_1
        self.SOURCE_AMPLITUDE_2 = -amp_1

    def create_geometry(self, **kwargs):
        """Add geometry to the model."""
        bg = Rectangle([0, 0], 1, 1)
        self.model_geometry.add_domain("bkg", bg, sigma=self.BG_COND)

        source_1 = Circle(self.SOURCE_LOCATION_1, self.SOURCE_RADIUS)
        source_2 = Circle(self.SOURCE_LOCATION_2, self.SOURCE_RADIUS)

        self.model_geometry.add_domain("source_1", source_1, sigma=self.BG_COND)
        self.model_geometry.add_domain("source_2", source_2, sigma=self.BG_COND)

    def define_current_sources(self):
        """Create source and sink for the model."""
        yield BoundaryCondition(
            self.SOURCE_AMPLITUDE_1, self.model_geometry["source_1"]
        )
        yield BoundaryCondition(
            self.SOURCE_AMPLITUDE_2, self.model_geometry["source_2"]
        )


@st.composite
def same_len_lists(draw):
    """Create two lists of the same but random length for hypothesis.

    :param draw: Hypotheis draw object
    :returns: valid lists
    """
    n = draw(st.integers(min_value=3, max_value=50))
    fixed_length_list = st.lists(
        st.floats(min_value=-1e6, max_value=1e6), min_size=n, max_size=n
    )

    return draw(fixed_length_list), draw(fixed_length_list)


def compare_comsol(u, filename: str) -> float:
    """Interpolate and compare Comsol and fish2eod solutions.

    :param u: fish2eod solution object
    :param filename: Name of the comsol file
    :return: NRMSE between two solutions
    """
    true_data = pd.read_csv(
        join(split(abspath(__file__))[0], "data", filename + ".txt"),
        delim_whitespace=True,
        skiprows=9,
        names=["x", "y", "v"],
    )

    x = np.linspace(true_data.x.values.min(), true_data.x.values.max(), 100)
    y = np.linspace(true_data.y.values.min(), true_data.y.values.max(), 100)

    comsol = griddata(
        (true_data.x.values, true_data.y.values),
        true_data.v,
        (x[None, :], y[:, None]),
        method="cubic",
    )

    if "fish" in filename:
        fenics_data = [
            u(a, b) for a, b in zip(true_data.x.values * 100, true_data.y.values * 100)
        ]
    else:
        fenics_data = [u(a, b) for a, b in zip(true_data.x.values, true_data.y.values)]

    fenics = griddata(
        (true_data.x.values, true_data.y.values),
        fenics_data,
        (x[None, :], y[:, None]),
        method="cubic",
    )

    err = comsol - fenics

    return np.sqrt(np.mean(err ** 2)) / (np.max(comsol) - np.min(comsol))


rectangle_inputs = (
    floats(min_value=-1e6, max_value=1e6),
    floats(min_value=-1e6, max_value=1e6),
    floats(min_value=1e-6, max_value=1e6),
    floats(min_value=1e-6, max_value=1e6),
    booleans(),
)
circle_inputs = (
    floats(min_value=-1e6, max_value=1e6),
    floats(min_value=-1e6, max_value=1e6),
    floats(min_value=1e-6, max_value=1e6),
)
translation_inputs = (
    floats(min_value=-1e6, max_value=1e6),
    floats(min_value=-1e6, max_value=1e6),
)
rotation_inputs = (
    floats(min_value=-1e6, max_value=1e6),
    booleans(),
    tuples(
        floats(min_value=-0.5e6, max_value=0.5e6),
        floats(min_value=-0.5e6, max_value=0.5e6),
    ),
)
