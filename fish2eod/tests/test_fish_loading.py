from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from fish2eod.geometry.fish import Fish, get_body, get_eod


class MockFish(Fish):
    def __init__(self, skeleton_x, skeleton_y):
        super().__init__(skeleton_x, skeleton_y)
        self.name = "mock"
        self.normal_length = 10
        self.head_distance = 1
        self.tail_distance = 9
        self.head_conductance = 1
        self.tail_conductance = 2
        self.organ_start = 2
        self.organ_length = 6
        self.organ_width = 0.01  # todo get these numbers?

        self.create_fish_components()


def mock_load_body():
    x = [0, 1, 2]
    y = [0, 0, 0]

    return pd.DataFrame({"x": x, "y": y})


def mock_load_eod():
    phase = [0, 0, 0, 1, 1, 1]
    x = [0, 1, 2, 0, 1, 2]
    eod = [5, 6, 7, 10, 11, 12]

    return pd.DataFrame({"phase": phase, "x": x, "eod": eod})


def mock_rectangle_body():
    return pd.DataFrame({"x": [0, 10], "y": [0.1, 0.1]})


@pytest.mark.quick
@mock.patch("fish2eod.geometry.fish.pd.read_csv", return_value=mock_load_body())
def test_get_body(_):
    body = get_body(Path().home())
    assert np.all(body.x == [0, 100, 200])
    assert np.all(body.y == [0, 0, 0])


@pytest.mark.quick
@pytest.mark.parametrize(("phase", "expected"), [(0, [5, 6, 7]), (1, [10, 11, 12])])
@mock.patch("fish2eod.geometry.fish.pd.read_csv", return_value=mock_load_eod())
def test_get_eod(_, phase, expected):
    eod = get_eod(phase, Path().home())

    for ix, x in enumerate([0, 0.5, 1]):
        assert np.isclose(expected[ix], eod(x) * 100 * 100)


@pytest.mark.quick
@mock.patch("fish2eod.geometry.fish.get_body", return_value=mock_rectangle_body())
def test_fish_creation(_):
    fish = MockFish([0, 10], [0, 0])

    assert np.isclose(
        fish.organ._shapely_representation.area,
        fish.organ_length * 2 * fish.organ_width,
    )
    assert np.isclose(
        fish.body._shapely_representation.area, fish.normal_length * 2 * 0.1, rtol=0.01
    )
    assert (
        fish.body._shapely_representation.area
        < fish.outer_body._shapely_representation.area
    )

    assert fish.skin_conductance(0.1, 0.1) == fish.head_conductance
    assert fish.skin_conductance(9.9, 0.1) == fish.tail_conductance
    assert (
        fish.skin_conductance(5, 0.1)
        == 0.5 * fish.head_conductance + 0.5 * fish.tail_conductance
    )


@pytest.mark.parametrize(
    ("x", "y", "var", "sign"),
    [
        ([0, 10], [0, 0], 1, 1),
        ([10, 0], [0, 0], 1, -1),
        ([0, 0], [0, 10], 0, -1),
        ([0, 0], [10, 0], 0, 1),
    ],
)
@mock.patch("fish2eod.geometry.fish.get_body", return_value=mock_rectangle_body())
def test_fish_sides(_, x, y, var, sign):  # todo fix this up
    fish = MockFish(x, y)

    data_l = sign * fish.sides["body"].left[:, var]
    data_r = sign * fish.sides["body"].right[:, var]
    assert np.mean(data_l) < 0
    assert np.mean(data_r) > 0
