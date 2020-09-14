# coding=UTF-8
"""Functions for creating and finite element meshes from primitives."""

import dolfin as df
import numpy as np
import pygmsh
from dolfin.cpp.mesh import MeshFunctionSizet

from fish2eod.geometry.primitives import Circle, Polygon
from fish2eod.mesh.model_geometry import ModelGeometry


def create_mesh(model_geometry: ModelGeometry, verbose: bool = True) -> df.Mesh:
    """Create the mesh from the model geometry.

    :param model_geometry: The model geometry to mesh
    :param verbose: Should gmsh output be printed
    :return: The computed mesh
    """
    mesh_geometry = pygmsh.opencascade.Geometry()
    # add each object to the mesh geometry
    surfaces = [
        mesh_add(obj, mesh_geometry) for (_, obj) in model_geometry.flat_geometry
    ]

    if len(surfaces) > 1:  # todo check whole background? and check different lcar
        mesh_geometry.boolean_fragments(surfaces[1:], surfaces[:1])

    created_mesh = pygmsh.generate_mesh(mesh_geometry, prune_z_0=True, verbose=verbose)
    assert (
        created_mesh.cells[-1].type == "triangle"
    ), "Something has changed in meshio: please submit a bug report"
    return create_dolfin_mesh(created_mesh.points, created_mesh.cells[-1].data)


def mesh_add(obj: Polygon, mesh_geometry: pygmsh.opencascade.Geometry):
    """Add an object to the mesh.

    Adds as a circle or polygon depending on the obj type

    :param obj: Circle to add
    :param mesh_geometry: pygmsh Geometry to add the circle to
    :return: Mesh surface
    """
    if isinstance(obj, Circle):
        return add_circle(obj, mesh_geometry)

    return add_poly(obj, mesh_geometry)


def add_circle(obj: Polygon, mesh_geometry: pygmsh.opencascade.Geometry):
    """Add circle objects to the mesh_geometry.

    :param obj: Circle to add
    :param mesh_geometry: pygmsh Geometry to add the circle to
    :return: Mesh surface
    """
    obj = mesh_geometry.add_circle(
        [obj.center[0], obj.center[1], 0], obj.radius, lcar=obj.lcar
    )
    return obj.plane_surface


def add_poly(obj: Polygon, mesh_geometry: pygmsh.opencascade.Geometry):
    """Add polygonal objects to the mesh_geometry.

    :param obj: Polygon (rectangle or generic) to add
    :param mesh_geometry: pygmsh Geometry to add the polygon to
    :return: Mesh surface
    """
    obj = mesh_geometry.add_polygon(list(obj.mesh_representation), lcar=obj.lcar)
    return obj.surface


def create_dolfin_mesh(points: np.ndarray, cells: np.ndarray) -> df.Mesh:
    """Convert the pygmsh mesh to a dolfin representation.

    https://bitbucket.org/fenics-project/dolfin/issues/845/initialize-mesh-from-vertices

    :param points: Mesh vertex coordinates
    :param cells: Mesh topology
    :returns: dolfin mesh
    """
    editor = df.MeshEditor()
    mesh = df.Mesh()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(points.shape[0])
    editor.init_cells(cells.shape[0])
    for k, point in enumerate(points):
        editor.add_vertex(k, point)
    for k, cell in enumerate(cells):
        editor.add_cell(k, cell)
    editor.close()
    return mesh


def create_mesh_function(name: str, mesh: df.Mesh, dimension: int) -> MeshFunctionSizet:
    """Create boilerplate mesh function.

    :param name: Meshfunction name
    :param mesh: Model mesh
    :param dimension: dimension of the function
    :return: Empty MeshFunction
    """
    mf = df.MeshFunction("size_t", mesh, dimension, mesh.domains())
    mf.set_all(0)
    mf.rename(name, name)

    return mf
