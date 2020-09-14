from typing import Union, Sequence, Callable, NamedTuple, Optional, Dict, List, Tuple

import dolfin as df
import numpy as np
from dataclasses import dataclass


class ElectricImageParameters(NamedTuple):
    """Helper named tuple for computing electric images.

    Specify the domain names and the conductivity value for the "null" condition for electric images

    domains is the set of domains to modify
    value is either a single conductance value for all domains or a sequence of numbers (1 per domain) for different
    null conditions
    """

    domains: Sequence[str]
    value: Union[float, Sequence[float]]


class DataSet(NamedTuple):
    """Helper for organizing a solution."""

    topology: Optional[np.ndarray]
    geometry: Optional[np.ndarray]
    data: Union[str, List]
    time_step: str
    name: str
    parameter_state: Dict[str, str]
    domain_map: Dict[str, str]


@dataclass(frozen=True)
class SkinStructure:
    """Store the skin in a helpful structure.

    _inner/_outer refer to the inner and outer skin layers in (x,y) pairs
    _arc_length saves the arc length of the boundary (inner boundary is used).
    """

    left_inner: np.ndarray
    left_outer: np.ndarray
    right_inner: np.ndarray
    right_outer: np.ndarray
    left_arc_length: np.ndarray
    right_arc_length: np.ndarray


@dataclass(frozen=True)
class TDP:
    """Store the transdermal potential (tdp) or image."""

    left_tdp: np.ndarray
    right_tdp: np.ndarray


class ComputatableSideInformation(NamedTuple):
    skin_type: str
    side: str
    coordinates: np.ndarray
    arc_length: np.ndarray


class BoundaryCondition(NamedTuple):
    """Helper named tuple for boundary conditions.

    value is the value of the boundary condition; it can be a number, a string-like expression i.e. "1+x[0]" or a full
    dolfin expression

    label is the boundary label to apply the value to
    """

    value: Union[float, str, df.Expression]
    label: int


EOD_TYPE = Union[float, Sequence[float]]

BOUNDARY_MARKER = Callable[[int, int], Tuple[bool, int]]

COLOR_STYLE_TYPE = Union[str, Sequence, None]

FISH_COORDINATES = Union[Sequence[Sequence[float]], Sequence[float]]
SPATIAL_FUNCTION = Callable[[float, float], float]

PARAMETER_TYPE = Optional[
    Dict[str, Union[float, Sequence[float], ElectricImageParameters]]
]
