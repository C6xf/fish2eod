import pytest

from fish2eod.models import BaseFishModel
from fish2eod.tests.testing_helpers import compare_comsol


@pytest.mark.integration
@pytest.mark.parametrize(
    ("species", "fish_length", "name"),
    [("Apteronotus", 21, "fish"), ("Eigenmannia", 26, "fish_eig")],
)
def test_fish(species, fish_length, name):
    skeleton_x = [0, fish_length]
    skeleton_y = [0, 0]
    p = {"fish_x": skeleton_x, "fish_y": skeleton_y, "phase": 0.24, "species": species}

    model = BaseFishModel()
    model.MIRROR_GROUND = True
    model.compile(**p)
    model.solve(**p)

    assert compare_comsol(model._fem_solution, name) <= 0.02
