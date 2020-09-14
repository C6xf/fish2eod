# coding=UTF-8
"""Geometry primitives for constructing a model.

Polygon is the base class which can be used to define abstract shapes
Circle defines circles
Rectangle defines rectangles
"""

from typing import Iterable, Optional, Sequence, Tuple, NamedTuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shp
from dolfin import SubDomain
from shapely import affinity

_SMALLEST_DIM = 1e-6  # smallest workable dimension
_LARGEST_DIM = 2e6  # largest workable dimension
_DOLFIN_X = "x[0]"  # name of internal dolfin x variable
_DOLFIN_Y = "x[1]"  # name of internal dolfin y variable


class Polygon(SubDomain):
    """Polygon object: all geometry objects are represented as polygons.

    Provides a convenient interface between shapely and dolfin SubDomains

    :param x: Sequence of floats specifying x coordinates
    :param y: Sequence of floats specifying y coordinates
    :param lcar: Optional overwrite for characteristic length
    """

    name = "Polygon"

    def __init__(
        self, x: Sequence[float], y: Sequence[float], lcar: Optional[float] = None
    ):
        """Instantiate Polygon."""
        err_msg = (
            f"Please ensure coordinates are between {_SMALLEST_DIM} and {_LARGEST_DIM}"
        )
        # points should be between +/-2e6 for sanity/stability reasons
        assert np.all(-_LARGEST_DIM <= np.round(x)) and np.all(
            np.round(x) <= _LARGEST_DIM
        ), err_msg

        assert np.all(-_LARGEST_DIM <= np.round(y)) and np.all(
            np.round(y) <= _LARGEST_DIM
        ), err_msg

        super().__init__()

        self.x, self.y, = x, y
        self._shapely_representation = shp.Polygon(list(zip(x, y)))

        self.lcar = lcar

        if not self._shapely_representation.is_valid:
            raise ValueError("Polygon is invalid - Likely self intersection")

    def simplify(self, threshold: float):
        """Simplify a geometry.

        Complex geometries can have too many micro verteces which make meshing complex. Strip out edges shorter than
        simp

        :param threshold: Threshold length to simplify
        :return: None
        """
        self._shapely_representation = self.shapely_representation.simplify(threshold)

    @property
    def shapely_representation(self) -> shp.Polygon:
        """Get representation of the shape as a geometry object."""
        return self._shapely_representation

    @property
    def mesh_representation(self) -> Iterable[Tuple[float, float, int]]:
        """Get the coordinates for the mesh.

        pygmsh requires surfaces to be open and implicitly closes them so x and y return [:-1]

        :return: Iterable of (x,y,z) pairs: z=0
        """
        x, y = zip(*self._shapely_representation.exterior.coords)  # unpack
        return zip(
            x[:-1],  # all but last x point - must be open curve
            y[:-1],  # all but last y point - must be open curve
            (len(x) - 1) * [0],
        )  # embed plane in 3D by setting z=0

    def expand(self, distance: float) -> "Polygon":
        """Blow up a geometry to widen it.

        Buffer or expand slightly a geometry object.

        Note: This is exact for circles but angles will round the more the shape is offset

        :param distance: The distance to offset: try to keep this small
        :return: The offset object
        """
        assert (
            _SMALLEST_DIM <= distance <= _LARGEST_DIM
        ), f"Please ensure d is between {_SMALLEST_DIM} and {_LARGEST_DIM}"

        new_shapely_representation = self._shapely_representation.buffer(distance)
        polygon_x, polygon_y = zip(*new_shapely_representation.exterior.coords)  # unzip
        return Polygon(polygon_x, polygon_y, lcar=self.lcar)  # todo becomes polygon

    def draw(self, color="k") -> None:
        """Plot the geometry object.

        :param color: Valid matplotlib color
        :return: None
        """
        coords = np.array(self._shapely_representation.exterior.coords).T
        plt.plot(*coords, color=color)
        plt.gca().set_aspect("equal")

    def inside(self, x, *_, buffer=1e-6) -> bool:
        """Is the point x inside the shape.

        :param x: Point to check
        :param _: Ignored
        :param buffer: Edge buffer to include edge points - should be small, enough to catch rounding errors
        """
        p = shp.Point(*x)
        return self._shapely_representation.buffer(buffer).contains(p)

    def __repr__(self) -> str:
        """Get helpful representation of class name and position."""
        try:
            # todo overwrite center for circ and square
            return self.name + " Geometry Object at: " + str(self.center)
        except AttributeError:
            return self.name + "Geometry Object"

    @property
    def center(self) -> Tuple[float, float]:
        """Get coordinates of the centroid of the object.

        Should be close but not exact for circles and rectangles

        :return: Tuple of center
        """
        # coords is a list of 1 centroid hence [0]
        centroid = self._shapely_representation.centroid.coords[0]
        return centroid[0], centroid[1]

    def __str__(self) -> str:
        """Pretty name."""
        return self.name + " Geometry Object"

    def intersects(self, other: "Polygon") -> bool:
        """Determine if geometry intersect another.

        :param other: Another geometry object to check
        :return: Whether or not the two geometries intersect
        """
        return self.shapely_representation.intersects(other.shapely_representation)

    def rotate(
        self,
        angle: float,
        degrees: bool = True,
        center: Optional[Sequence[float]] = None,
    ) -> "Polygon":
        """Rotate an object.

        :param angle: Angle to rotate object in
        :param degrees: Is number in degrees or radians. Defaults to degrees
        :param center: Optional center to rotate about: sequence of 2 floats
        """
        # Use center of object is center not specified
        center = self.center if center is None else center
        center = shp.Point(center[0], center[1])

        # Perform rotation
        new_shapely_representation = affinity.rotate(
            self._shapely_representation, angle, use_radians=not degrees, origin=center
        )

        polygon_x, polygon_y = zip(*new_shapely_representation.exterior.coords)  # unzip
        return Polygon(polygon_x, polygon_y, lcar=self.lcar)

    def translate(self, dx: float = 0, dy: float = 0) -> "Polygon":
        """Translate an object.

        :param dx: Shift in x
        :param dy: Shift in y
        """
        err_msg = f"Please ensure shift is between {_SMALLEST_DIM} and {_LARGEST_DIM}"
        assert -_LARGEST_DIM <= dx <= _LARGEST_DIM, err_msg
        assert -_LARGEST_DIM <= dy <= _LARGEST_DIM, err_msg

        new_shapely_representation = affinity.translate(
            self._shapely_representation, xoff=dx, yoff=dy
        )

        polygon_x, polygon_y = zip(*new_shapely_representation.exterior.coords)  # unzip
        return Polygon(polygon_x, polygon_y, lcar=self.lcar)  # todo always polygon


class Circle(Polygon):
    """Circle geometry object.

    :param center: Sequency of 2 floats specifying circle center
    :param radius: Circle radius
    :param lcar: Optional overwrite for characteristic length
    """

    name = "Circle"

    def __init__(
        self, center: Sequence[float], radius: float, lcar: Optional[float] = None
    ):
        """Instantiate Circle."""
        err_msg = f"Please ensure radius is between {_SMALLEST_DIM} and {_LARGEST_DIM}"  # make sure radius is reasonable
        assert _SMALLEST_DIM <= radius <= _LARGEST_DIM, err_msg
        self.radius = radius

        center_point = shp.Point(center[0], center[1])
        shapely_representation = shp.Polygon(center_point.buffer(self.radius))
        polygon_x, polygon_y = zip(*shapely_representation.exterior.coords)

        super().__init__(polygon_x, polygon_y, lcar=lcar)

    def inside(self, x, *_, buffer: float = 5e-6) -> str:
        """Create the string expression for a particular circle for dolfin to check if a point is inside.

        x[0] and x[1] are the dolfin expressions for coordinates x, y

        checks if (x-x[0])^2 + (y-x[1])^2 <= (r+buffer)^2

        :param x: Ignored
        :param _: Ignored
        :param buffer: Radius buffer to include edge points - should be small, enough to catch rounding errors
        :return: String of code to compile
        """
        return (
            f"(({self.center[0]} - {_DOLFIN_X}) * ({self.center[0]} - {_DOLFIN_X}) +"
            f" ({self.center[1]} - {_DOLFIN_Y}) * ({self.center[1]} - {_DOLFIN_Y})) <="
            f" (({self.radius + buffer})*({self.radius + buffer}))"
        )  # multiline return like this does concat


class Rectangle(Polygon):
    """Rectangle geometry object.

    :param corner: Bottom left corner of the rectangle: sequence of 2 floats defining the bottom left
    :param width: Width of the rectangle (x-axis)
    :param height: Optional height of the rectangle: if not specified the rectangle becomes a square
    :param lcar: Optional overwrite for characteristic length
    """

    name = "Rectangle"

    def __init__(
        self,
        corner: Sequence[float],
        width: float,
        height: Optional[float] = None,
        lcar: Optional[float] = None,
    ):
        """Instantiate rectangle."""
        # set properties before checking as operations done in setter
        self.width = width
        self.height = height

        err_msg = f"Please ensure dimension between {_SMALLEST_DIM} and {_LARGEST_DIM}"
        assert _SMALLEST_DIM <= self.width <= _LARGEST_DIM, err_msg
        assert _SMALLEST_DIM <= self.height <= _LARGEST_DIM, err_msg

        polygon = [
            [corner[0], corner[1]],
            [corner[0] + self.width, corner[1]],
            [corner[0] + self.width, corner[1] + self.height],
            [corner[0], corner[1] + self.height],
        ]  # open polygon

        self.corner = corner

        # unpack polygon into x, y list and expand into arguments
        super().__init__(*zip(*polygon), lcar=lcar)

    @classmethod
    def from_center(
        cls,
        center: Sequence[float],
        width: float,
        height: Optional[float] = None,
        lcar: Optional[float] = None,
    ) -> "Rectangle":
        """Construct rectangle when the center of the rectangle is specified (alternate constructor).

        :param center: Center coordinate: sequence of 2 floats defining the rectangle center
        :param width: Width of the rectangle (x-axis)
        :param height: Optional height of the rectangle: if not specified the rectangle becomes a square
        :param lcar: Optional overwrite for characteristic length
        :return: A rectangle defined from the center
        """
        height = cls.validate_height(height, width)
        bottom = [center[0] - width / 2, center[1] - height / 2]  # bottom left corner

        return cls(bottom, width, height, lcar=lcar)

    @property
    def height(self) -> float:
        """Get the height of the rectangle.

        If the rectangle is defined as a square then _height is None and therefore return the width instead
        """
        return self.validate_height(self._height, self.width)

    @height.setter
    def height(self, h: Optional[float]) -> None:
        """Set the height of the rectangle.

        :param h: Height, none if square
        :return: None
        """
        self._height = h

    @staticmethod
    def validate_height(height: Optional[float], width: float) -> float:
        """Convert the height to a valid number.

        When the rectangle is defined as a square the height is None

        :param height: Height to validate
        :param width: Width of object
        :return: Returns the height or the width if height is None
        """
        return height if height else width

    def inside(self, x, *_, buffer=1e-6) -> str:
        """Create the string expression for a particular rectangle for dolfin to check if a point is inside.

        x[0] and x[1] are the dolfin expressions for coordinates x, y

        :param x: Ignored
        :param _: Ignored
        :param buffer: Edge buffer to include edge points - should be small, enough to catch rounding errors
        :return: String of code to compile
        """
        return (
            f"({self.corner[0] - buffer} < {_DOLFIN_X}) &&"
            f"({self.corner[1] - buffer} < {_DOLFIN_Y}) &&"
            f"({self.corner[0] + self.width + buffer} > {_DOLFIN_X}) &&"
            f"({self.corner[1] + self.height + buffer} > {_DOLFIN_Y})"
        )


class PreDomain(NamedTuple):
    """Convenience structure for containing information to be converted to a domain."""

    label: int
    primitive: bool
    objects: Union[Polygon, List[Polygon]]
