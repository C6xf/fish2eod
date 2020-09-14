"""Compute transdermal potential (tdp).

Compute the tdp for the left and right sides. Images are computed when the input has already been subtracted.
"""
from typing import Tuple, Iterable, TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import during from typing
    from fish2eod import BaseFishModel

import numpy as np
from scipy.interpolate import interp1d

from fish2eod.geometry.fish import Fish
from fish2eod.geometry.operations import uniform_spline_interpolation
from fish2eod.helpers.type_helpers import (
    SkinStructure,
    TDP,
    ComputatableSideInformation,
)


def get_skin_coordinates(fish: Fish, skin_type: str, side: str) -> np.ndarray:
    data = getattr(fish.sides[skin_type], side)
    return np.vstack(uniform_spline_interpolation(*data.T, n=100)).T


def get_skin_arc_length(coordinates: np.ndarray) -> np.ndarray:
    coordinate_distnace = np.sqrt(
        np.diff(coordinates[:, 0]) ** 2 + np.diff(coordinates[:, 1]) ** 2
    )
    arc_length = np.concatenate([[0], np.cumsum(coordinate_distnace)])

    return arc_length


def get_side_information(fish: Fish) -> Iterable[ComputatableSideInformation]:
    for side in ["left", "right"]:
        for skin_type in ["body", "outer_body"]:
            coordinates = get_skin_coordinates(fish, skin_type, side)
            yield ComputatableSideInformation(
                skin_type=skin_type,
                side=side,
                coordinates=coordinates,
                arc_length=get_skin_arc_length(coordinates),
            )


def skin_potential(
    model: "BaseFishModel", side_information: ComputatableSideInformation
):
    v = [model.eval(*c) for c in side_information.coordinates]
    voltage_on_skin = interp1d(np.linspace(0, 1, len(v)), v)
    return voltage_on_skin(np.linspace(0, 1, len(v)))


def compute_transdermal_potential(
    model: "BaseFishModel",
) -> Iterable[Tuple[SkinStructure, TDP]]:
    """Compute the electric image.

    TODO refactor this

    :param model: Saved model
    :return: Iterator over skin and electric image as structs
    """

    for ix, fish in enumerate(model.fish_container.fishes):
        side_information = get_side_information(fish)
        data = {
            f"{d.side}-{d.skin_type}": [
                d.coordinates,
                d.arc_length,
                skin_potential(model, d),
            ]
            for d in side_information
        }

        left_tdp = data["left-outer_body"][2] - data["left-body"][2]  # ext - int
        right_tdp = data["right-outer_body"][2] - data["right-body"][2]

        # create data classes with dynamic names to inject fish index into data set
        sub_skin_structure = type(f"SkinStructure_{ix}", (SkinStructure,), {})
        sub_electric_image = type(f"ElectricImage_{ix}", (TDP,), {})
        yield (
            sub_skin_structure(
                left_inner=data["left-body"][0],
                left_outer=data["left-outer_body"][0],
                right_inner=data["right-body"][0],
                right_outer=data["right-outer_body"][0],
                left_arc_length=data["left-body"][1],
                right_arc_length=data["right-body"][1],
            ),
            sub_electric_image(left_tdp=left_tdp, right_tdp=right_tdp),
        )
