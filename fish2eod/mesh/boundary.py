# coding=UTF-8
"""Functions for creating and labeling finite element boundaries."""

from typing import Iterable, Optional, Tuple, Sequence

import dolfin as df
from dolfin.cpp.mesh import MeshFunctionSizet

from fish2eod.helpers.type_helpers import BOUNDARY_MARKER
from fish2eod.mesh.mesh import create_mesh_function
from fish2eod.mesh.model_geometry import ModelGeometry


def boundary_iterator(
    domains: MeshFunctionSizet,
) -> Iterable[Tuple[df.Facet, Tuple[int, ...]]]:
    """Get every facet (edge) with at least 2 things touching it i.e. not an external edge.

    :param domains: The labeled domains
    :return: Iterable of tuples of an edge and its touching domains
    """
    for facet in df.facets(domains.mesh()):  # iterate over edges in the mesh
        # set: therefore ignore duplicates
        touching_domains = {domains[c] for c in df.cells(facet)}
        if len(touching_domains) > 1:
            # if at least 2 partners i.e. not an external edge
            yield facet, tuple(sorted(touching_domains))


def mark_edge_by_rules(
    neighbouring_domains: Sequence[int], boundary_markers: Tuple[BOUNDARY_MARKER]
) -> Optional[int]:
    """Use rules specified in BoundaryCondition to mark complex edges.

    Gets the edge label from the rule if it should be marked, otherwise returns -1

    Markers are applied sequentially: so if a boundary is defined and is then crossed by another. The second boundary
    takes precedence

    :param neighbouring_domains: Domains touching the edges
    :param boundary_markers: BoundaryMarkers to use
    :return: Label of the edge
    """
    label = None
    for marker in boundary_markers:  # like domains rules are applied sequentially
        # todo maybe convert to iterator
        should_mark, new_label = marker(
            neighbouring_domains[0], neighbouring_domains[1]
        )
        if should_mark:
            label = new_label
    return label


class External(df.SubDomain):
    """Label external boundaries."""

    @staticmethod
    def inside(_, *on_boundary):
        """Is the point inside the domain.

        In this case the inside method is only considering if the point is a on a boundary between 2 domains.

        :param _: Ignored
        :param on_boundary: List from dolfin index 0 is used to determine boundary state
        """
        return on_boundary[0]


def mark_boundary(
    model_geometry: ModelGeometry,
    neighbouring_domains: Sequence[int],
    boundary_markers: Tuple[BOUNDARY_MARKER],
) -> int:
    """Mark a boundary given its neighbours and boundary markers.

    :param model_geometry: ModelGeometry containing the geometry
    :param neighbouring_domains: Domains touching the edge
    :param boundary_markers: List
    :return:
    """
    # TODO background will overwrite label example
    if model_geometry.background_label in neighbouring_domains:
        # if the background label is in the neighbouring domains - set the edge id to the other domain's id
        # todo ensure that there are only 2 neighbouring domains
        return neighbouring_domains[1]

    # We need custom labeling - use an appropriate label or treat as background
    label = mark_edge_by_rules(neighbouring_domains, boundary_markers)
    if label:  # if the edge could be labeled by a rule
        return label
    # treat as background
    return model_geometry.background_label


def mark_boundaries(
    domains: MeshFunctionSizet,
    model_geometry: ModelGeometry,
    boundary_markers: Tuple[BOUNDARY_MARKER],
    external_boundary,
) -> Tuple[MeshFunctionSizet, MeshFunctionSizet]:
    """Top level function to mark boundaries.

    :param domains: Model domains
    :param model_geometry: Model geometry
    :param boundary_markers: List of BoundaryMarkers
    :param external_boundary: ID of the external boundary
    :return: Boundary and Outline MeshFunctions
    """
    mesh = domains.mesh()
    boundaries = create_mesh_function("boundary", mesh, 1)
    # todo copy boundaries if possible?
    outline = create_mesh_function("outline", mesh, 1)

    External().mark(boundaries, external_boundary)  # mark external boundaries

    # get each edge and its mating domains
    for edge, neighbouring_domains in boundary_iterator(domains):
        #  all edges at this stage are interesting and belong to an outline
        outline[edge] = 1
        boundaries[edge] = mark_boundary(
            model_geometry, list(neighbouring_domains), boundary_markers
        )

    return boundaries, outline
