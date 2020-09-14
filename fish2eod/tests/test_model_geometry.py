import pytest

from fish2eod.geometry.primitives import Circle, Rectangle
from fish2eod.mesh.model_geometry import ModelGeometry

overlaps_c1 = Circle([0, 0], 0.5)
not_on_bg = Circle([15, 15], 1)
bg = Rectangle.from_center([0, 0], 10)
c1 = Circle([0, 0], 1)
c2 = Circle([2, 2], 1)


@pytest.mark.quick
def test_create_only_bg():
    ModelGeometry()


@pytest.mark.quick
def test_create_seperate_domains():
    mg = ModelGeometry()
    mg.add_domain("bg", bg)
    mg.add_domain("c1", c1)
    mg.add_domain("c2", c2)


@pytest.mark.quick
def test_not_on_bg():
    mg = ModelGeometry()
    mg.add_domain("bg", bg)
    with pytest.raises(ValueError) as _:
        mg.add_domain("off", not_on_bg)


@pytest.mark.quick
def test_multiple_on_domain():
    mg = ModelGeometry()
    mg.add_domain("bg", bg)
    mg.add_domain("c", c1, c2)


@pytest.mark.quick
def test_reuse_name():
    mg = ModelGeometry()
    mg.add_domain("bg", bg)
    mg.add_domain("c1", c1)
    with pytest.raises(ValueError) as _:
        mg.add_domain("c1", c2)


@pytest.mark.quick
def test_overlap_different_domain():
    mg = ModelGeometry()
    mg.add_domain("bg", bg)
    mg.add_domain("c1", c1)
    with pytest.raises(ValueError) as _:
        mg.add_domain("o1", overlaps_c1)


@pytest.mark.quick
def test_overlap_same_domain():
    mg = ModelGeometry()
    mg.add_domain("bg", bg)
    with pytest.raises(ValueError) as _:
        mg.add_domain("o1", c1, overlaps_c1)
