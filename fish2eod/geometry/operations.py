# coding=UTF-8
"""Geometry operations for slicing, offsetting, and interpolating lines."""

from functools import partial
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import shapely.geometry as shp
from numpy.linalg import norm
from scipy.interpolate import InterpolatedUnivariateSpline
from shapely import ops


def cut_line_between_fractions(
    line: shp.linestring, start_fraction: float, end_fraction: float
) -> shp.LineString:  # todo refactor
    """Cut a line at 2 fractions and return the region between them.

    :param line: Linestring to cut
    :param start_fraction: Fraction 1 (0 < f1 < f2)
    :param end_fraction: Fraction 2 (f1 < f2 < 1)
    :return: Linestring that was cut out
    """
    assert 0 <= start_fraction <= end_fraction
    assert start_fraction <= end_fraction <= 1

    # point a fraction on the line
    p1 = line.interpolate(start_fraction, normalized=True)
    p2 = line.interpolate(end_fraction, normalized=True)

    # Cut the line: this returns the target cut but also returns several little chunks (<< 1e-6 in length)
    # small buffer includes rounding errors
    cut_lines = ops.split(line, shp.MultiPoint([p1, p2]).buffer(1e-12))

    # Filter out the tiny pieces
    filter_function = partial(filter_line_length, line, start_fraction, end_fraction)
    valid_line = filter(filter_function, cut_lines)

    # There should be a single item in this list
    return list(valid_line)[0]


def filter_line_length(
    original_line: shp.LineString,
    start_fraction: float,
    end_fraction: float,
    line_slice: shp.LineString,
    tol: float = 1e-10,
) -> bool:
    """Determine if a line slice is "very close" to the target line fractions.

    :param original_line: Original line being cut
    :param start_fraction: Fraction 1 (0 < f1 < f2)
    :param end_fraction: Fraction 2 (f1 < f2 < 1)
    :param line_slice: New line to check
    :param tol: Optional parameter to define how close is "close"
    :return: If the line_slice is the desired sliced line
    """
    # Computes the effective fraction of the line slice on the original line
    line_fractions = [
        original_line.project(shp.Point(c), normalized=True) for c in line_slice.coords
    ]

    # Compute the distance between the fractions and computes the error
    err = abs(line_fractions[0] - start_fraction) + abs(
        line_fractions[-1] - end_fraction
    )

    return err < tol


def measure_and_interpolate(
    x: Sequence[float], y: Sequence[float], m: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate coordinates and compute length of each section.

    :param x: X coordinates
    :param y: Y coordinates
    :param m: Minimum number of points to interpolate
    :return: length of each segment and interpolated x, y
    """
    t = np.linspace(0, len(x) - 1, m)
    t = np.unique(np.concatenate((t, np.arange(len(x)))))
    x_i = np.interp(t, np.arange(len(x)), x)
    y_i = np.interp(t, np.arange(len(y)), y)

    distance = np.sqrt(np.diff(x_i) ** 2 + np.diff(y_i) ** 2)  # distance of each step
    return distance, x_i, y_i


def uniform_spline_interpolation(
    x: Sequence[float], y: Sequence[float], n: int, m: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate coordinates with n evenly spaced points along a spline.

    Interpolation will ensure the first and last point on the line are included in the spline

    If there are turns in the spline (like a sine-wave) the number of points (n) needs to be high enough that spatial
    frequency of sampled points >> spatial frequency of spline or you will get cuts at the curves - interpolation is a
    low pass filter

    :param x: x coordinates
    :param y: y coordinates
    :param n: Number of points to interpolate
    :param m: Resolution for computing curve length (default of 10000 should be sufficient)
    :return: Interpolated x, y points for sources
    """
    # Interpolate curve and measure distance between each point
    distance, x_i, y_i = measure_and_interpolate(x, y, m)
    cum_distance = np.cumsum(distance)

    # Each selected point should jump 1/(n-1)
    target_distances = cum_distance[-1] / (n - 1) * np.arange(1, n)

    # [0] and indexes where cum_distance closest to target distance
    nodes = [0] + [
        np.argmin(np.abs(cum_distance - target_distance)) + 1
        for target_distance in target_distances
    ]

    return x_i[nodes], y_i[nodes]


def parallel_curves(
    x: Sequence[float], y: Sequence[float], d: Union[np.ndarray, float] = 1.0
) -> Dict[str, np.ndarray]:
    """TODO def this."""
    # https://github.com/boredStats/parallel-curves-for-python
    dx = np.gradient(x)
    dy = np.gradient(y)

    dx2 = np.gradient(dx)
    dy2 = np.gradient(dy)

    nv = np.ndarray(shape=[len(dy), 2])
    nv[:, 0] = dy
    nv[:, 1] = -dx

    unv = np.zeros(shape=nv.shape)
    norm_nv = norm(nv, axis=1)  # magn(nv, 2)

    unv[:, 0] = nv[:, 0] / norm_nv
    unv[:, 1] = nv[:, 1] / norm_nv

    r0 = (dx ** 2 + dy ** 2) ** 1.5
    r1 = np.abs(dx * dy2 - dy * dx2)
    r1[r1 < 1e-13] = 1e-13
    radius = r0 / r1  # r0 / r1 #todo SANITY CHECK THIS

    overlap = radius < d

    dy3 = np.zeros(shape=dy2.shape)
    dy3[dy2 > 0] = 1
    concavity = 2 * dy3 - 1

    x_inner = x - unv[:, 0] * d
    y_inner = y - unv[:, 1] * d

    x_outer = x + unv[:, 0] * d
    y_outer = y + unv[:, 1] * d

    res = {
        "x_inner": x_inner,
        "y_inner": y_inner,
        "x_outer": x_outer,
        "y_outer": y_outer,
        "R": radius,
        "unv": unv,
        "concavity": concavity,
        "overlap": overlap,
    }
    return res


def extend_line(
    x: Sequence[float], y: Sequence[float], fraction: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Extend (linearly extrapolate) an arbitrary curve at both ends.

    x and y are now of the form

    front_extrapolate, x_original, end_extrapolate

    :param x: x coordinates of the curve
    :param y: y coordinates of the curve
    :param fraction: Fraction of the line distance to extrapolate
    :return: Extended line
    """
    assert np.all(np.abs(x) <= 1e6)
    assert np.all(np.abs(y) <= 1e6)

    # parameterize the curve on t=[0, 1]
    t = np.linspace(0, 1, len(x))

    # k=1 means linear
    interpolator_x = InterpolatedUnivariateSpline(t, x, k=1, ext="extrapolate")
    interpolator_y = InterpolatedUnivariateSpline(t, y, k=1, ext="extrapolate")

    # stitch the the points at -fraction and 1+fraction onto the original line
    x = np.concatenate([[interpolator_x(-fraction)], x, [interpolator_x(1 + fraction)]])
    y = np.concatenate([[interpolator_y(-fraction)], y, [interpolator_y(1 + fraction)]])

    return x, y
