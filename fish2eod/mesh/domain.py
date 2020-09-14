# coding=UTF-8
"""Functions for working with finite element domains."""

from functools import reduce
from typing import Iterable, List, Union

import dolfin as df
from dolfin.cpp.mesh import MeshFunctionInt

from fish2eod.geometry.primitives import Circle, Polygon, Rectangle
from fish2eod.mesh.mesh import create_mesh_function
from fish2eod.mesh.model_geometry import ModelGeometry
from fish2eod.geometry.primitives import PreDomain


def mark_domains(mesh: df.Mesh, model_geometry: ModelGeometry) -> MeshFunctionInt:
    """Create a meshfunction marked with the domain ids.

    :param mesh: Created dolfin mesh
    :param model_geometry: ModelGeometry containing the added geometry
    :return: Domain meshfunction
    """
    domains = create_mesh_function("domain", mesh, 2)
    for pre_domain in iterate_domains(model_geometry):
        domain_label, is_primitive, objs = (
            pre_domain.label,
            pre_domain.primitive,
            pre_domain.objects,
        )
        if is_primitive:
            mark_primitives(domain_label, domains, objs)
        else:
            mark_polygon(domain_label, domains, objs)

    return domains


def iterate_domains(model_geometry: ModelGeometry) -> Iterable[PreDomain]:
    """Iterate over objects in a model geometry and inform if the object is primitive.

    :param model_geometry: ModelGeometry to iterate
    :return: Domain label, if the objects are primitive, and list of primitives or a polygon
    """
    # get objects on each defined domain
    for (domain_label, objs,) in model_geometry.geometry_map.items():
        # partition objs into polygons and primitives because they're currently handled separately
        primitives = [o for o in objs if type(o) != Polygon]
        polygons = [o for o in objs if type(o) == Polygon]

        if primitives:
            yield PreDomain(label=domain_label, primitive=True, objects=primitives)

        for polygon in polygons:  # polygons must be marked one by one
            yield PreDomain(label=domain_label, primitive=False, objects=polygon)


def mark_primitives(
    domain_label: int, domains: df.MeshFunction, objs: List[Union[Circle, Rectangle]]
) -> None:
    """Effecient marker for primitives.

    Each primitive has an inside(x, ...) which returns some boolean string expression

    I.e. there will be a list of ["a==b", "c==d", ...]

    Reduce opperation convers that to (a==b) || (c==d) ||

    The [:-3] strips off the " ||"

    :param domain_label: Label (int) of the domain
    :param domains: Domain meshfunction to update
    :param objs: Objects to mark
    :returns: None
    """
    statement = reduce(
        lambda x, y: x + "(" + y + ") || ", [o.inside(None, None) for o in objs], ""
    )
    # strip off last " || " and mark in compiled code
    df.CompiledSubDomain(statement[:-3]).mark(domains, domain_label)


def mark_polygon(domain_label: int, domains: df.MeshFunction, obj: Polygon) -> None:
    """Extremely inefficient marker for polygons.

    Try to create complex shapes out of rectangles instead of resorting to this

    :param domain_label: Label (int) of the domain
    :param domains: Domain meshfunction to update
    :param obj: Polygon to mark (only one at a time)
    :returns: None
    """
    obj.mark(domains, domain_label)  # TODO: these cost a lot!
