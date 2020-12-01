"""Plotting module for wrapping matplotlib with fish2eod data.

2D plots are "domain-like"
1D plots are "boundary-like"
"""
from inspect import signature
from typing import Tuple, Union, Any, Sequence, Optional, Dict

import matplotlib as mpl
import matplotlib.axes
import matplotlib.colors
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation, LinearTriInterpolator

from fish2eod.helpers.type_helpers import COLOR_STYLE_TYPE
from fish2eod.mesh.model_geometry import ModelGeometry
from fish2eod.models import Model
from fish2eod.xdmf.load import H5Solution


def split_mpl_kwargs(tainted: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Split kwargs into valid mpl kwargs and internal fish2fem kwargs

        Valid keys are determined by inclusion in the Line2D signature which should contain most common ones

        :param tainted: kwarg dictionary containing the mixture of kwargs
        :returns: Matplotlib kwargs and fish2fem kwargs dictionaries
    """
    valid_kwargs = (
        set(signature(mpl.lines.Line2D).parameters.keys())
        .union(signature(mpl.axes.Axes.imshow).parameters.keys())
        .union(signature(mpl.collections.Collection).parameters.keys())
    )  # common line or image parameters

    mpl_kwargs = {k: tainted[k] for k in tainted.keys() if k in valid_kwargs}
    fish2fem_kwargs = {k: tainted[k] for k in tainted.keys() if k not in valid_kwargs}

    return mpl_kwargs, fish2fem_kwargs


def upscale_parameter_name(normal_params: Dict[str, Any]) -> Dict[str, Any]:
    """ Infer valid matplotlib parameters for line collection.

    Line collection parameters have plural names of common parameters (i.e. color -> colors).
    Try every parameter with an additional 's' and see it it fits a valid parameter name

    :param normal_params: Parameters with standard (color, linestyle, ...) names
    :returns: Parameters with pluralized (colors, linestyles, ...) names
    """
    valid_kwargs = set(signature(mpl.collections.Collection).parameters.keys()).union(
        ["colors"]
    )  # for some reason colors is hidden in kwargs
    upscaled = {**normal_params, **{k + "s": v for k, v in normal_params.items()}}

    return {k: upscaled[k] for k in upscaled.keys() if k in valid_kwargs}


def generate_mask(
    solution: H5Solution,
    include_domains: Tuple[str, ...] = ("water", "body"),
    include=True,
    **kwargs
) -> Tuple[int, ...]:
    """Generate a Mask For the Domain name.

    Masked nodes are removed from plot/analysis. When requesting include the masked nodes are flipped

    :param solution: Loaded solution object
    :param include_domains: Tuple of domains to mask
    :param include: Whether or not the domains included or excluded
    :return: Valid matplotlib mask
    """
    if isinstance(
        include_domains, str
    ):  # if the included_domains are passed as a string
        include_domains = (include_domains,)

    *_, domain_function = extract(solution, "domain", **kwargs)
    valid_domains = ~np.isin(
        domain_function, [solution.domain_map[d] for d in include_domains]
    )  # mask is not in

    if not include:  # flip mask if were excluding domains
        valid_domains = ~valid_domains

    return tuple(valid_domains)


def generate_triangles_for_2d(
    solution: H5Solution, variable: str, mask: Optional[Sequence[bool]] = None, **kwargs
) -> Tuple[Triangulation, Any]:
    """Generate the triangle and data structure for 2d data (domain-like).

    :param solution: Loaded solution object
    :param variable: Variable to extract
    :param mask: Valid mask
    :param kwargs: Parameter selection for the solution
    :return: Triangles and data for plotting
    """
    topology, geometry, data = extract(solution, variable, **kwargs)
    return generate_triangles(topology, geometry, mask=mask), data


def generate_triangles_for_1d(
    solution: H5Solution, variable: str, mask: Optional[Sequence[bool]] = None, **kwargs
) -> Tuple[Triangulation, Any]:
    """Generate the triangle and data structure for 1d data (boundary-like).

    :param solution: Loaded solution object
    :param variable: Variable to extract
    :param mask: Valid mask
    :param kwargs: Parameter selection for the solution
    :return: Triangles and data for plotting
    """
    _, geometry, data = extract(solution, variable, **kwargs)
    tri_topology, *_ = extract(solution, "domain", **kwargs)

    return generate_triangles(tri_topology, geometry, mask=mask), data


def generate_triangles(
    topology, geometry, mask: Optional[Sequence[bool]] = None
) -> Triangulation:
    """Convert topology and geometry to triangles - optionally set mask as well.

    :param topology: Topology of the triangles (connectivity)
    :param geometry: Geometry of the triangles (location of the nodes)
    :param mask: Valid matplotlib mask
    :return: Triangulation object for plotting
    """
    tri = Triangulation(geometry[:, 0], geometry[:, 1], topology)
    if mask:
        tri.set_mask(mask)

    return tri


def get_valid_nodes(triangles: Triangulation) -> np.ndarray:
    """Get nodes in the mesh wish are valid (not masked).

    :param triangles: The triangulated mesh
    :return: List of node ids outside of mask
    """
    return np.unique(triangles.get_masked_triangles()).reshape(-1,)


# TODO
def normalize_color(
    triangles: Optional[Triangulation], data, color_style: COLOR_STYLE_TYPE
) -> matplotlib.colors.Normalize:
    r"""Compute a colormap for data based on color style.

    color_style options:

    Falsy (None, false, ...): Standard color map from min(data) -- max(data)
    "full": Centers at 0 with the bounds being min(data), max(data). This does not have even positive and negative
    scales. I.e. for a red-blue map with range -1 to +10 then 0-10 will be white to red and -1 to 0 will be blue to
    white
    "equal" Symmetric color ranges with 0 at center. I.e. scale is from \pm max(abs(data))
    array-like [min, max] sets the colorbar to a preset range

    :param triangles: Triangulated mesh
    :param data: Data to normalized
    :param color_style: What type of colormap
    :return: Normalized map
    """
    if isinstance(color_style, Sequence) and not isinstance(color_style, str):
        return matplotlib.colors.Normalize(vmin=color_style[0], vmax=color_style[1])

    norm = matplotlib.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    if color_style:
        if triangles:
            good_nodes = get_valid_nodes(triangles)
        else:
            good_nodes = np.ones(data.shape, dtype=bool)

        if color_style == "full":
            if np.min(data[good_nodes]) >= 0 or np.max(data[good_nodes]) <= 0:
                raise ValueError("Full colorscheme requires positive and negative data")
            norm = matplotlib.colors.TwoSlopeNorm(
                vmin=np.min(data[good_nodes]), vcenter=0, vmax=np.max(data[good_nodes])
            )
        else:
            if color_style != "equal":
                Warning("asdf")
            r = np.max(np.abs(data[good_nodes]))
            if r == 0:
                r = 1
            norm = matplotlib.colors.TwoSlopeNorm(vmin=-r, vcenter=0, vmax=r)

    return norm


def mesh_plot_2d(
    solution,
    variable: str,
    *,
    color_style: COLOR_STYLE_TYPE = "equal",
    colorbar: bool = True,
    mask=None,
    **kwargs
):
    """Plot surface functions i.e. functions defined on the surface of the mesh or on the nodes.

    :param solution: The solution structure
    :param variable: Name of the variable
    :param color_style: How to symmetrize the colormap (False/None, 'full', 'equal')
    :param colorbar: Should the colorbar be plotted
    :param mask: Valid matplotlib mask see generate_mask
    :param kwargs: Data subset parameters and matplotlib settings
    """

    mpl_kwargs, fish2fem_kwargs = split_mpl_kwargs(kwargs)
    if "topology" in fish2fem_kwargs:
        topology, geometry, data = (
            fish2fem_kwargs.pop("topology"),
            fish2fem_kwargs.pop("geometry"),
            fish2fem_kwargs.pop("data"),
        )
        triangles = generate_triangles(topology, geometry, mask=mask)
    else:
        triangles, data = generate_triangles_for_2d(
            solution, variable, mask=mask, **fish2fem_kwargs
        )

    color_norm = normalize_color(triangles, data, color_style)

    if len(triangles.triangles) == len(
        data
    ):  # defined on the surface #data = facecolor
        im = plt.tripcolor(triangles, facecolors=data, norm=color_norm, **mpl_kwargs)
    else:
        im = plt.tripcolor(
            triangles, data, shading="gouraud", norm=color_norm, **mpl_kwargs
        )

    cbar = None
    if colorbar:
        cbar = plt.colorbar(extend="both")
    plt.gca().set_aspect("equal")

    return im, cbar


def field_norm(e_x: np.ndarray, e_y: np.ndarray) -> np.ndarray:
    """Compute norm of a vector field.

    :param e_x: x component of the field
    :param e_y: y compoennt of the field
    :return: Norm of the field
    """
    return np.sqrt(e_x ** 2 + e_y ** 2)


def gradient(
    solution,
    variable,
    *,
    gradient_norm=True,
    plot=True,
    color_style: COLOR_STYLE_TYPE = "equal",
    mask=None,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute and plot the gradient of a scalar field.

    :param solution: The solution structure
    :param variable: Name of the variable
    :param gradient_norm: Should the norm be computed
    :param plot: Should the data be plotted
    :param color_style: How to symmetrize the colormap (False/None, 'full', 'equal')
    :param mask: Valid matplotlib mask see generate_mask
    :param kwargs:
    :return: Computed field as a scalar or vector field
    """
    mpl_kwargs, fish2fem_kwargs = split_mpl_kwargs(kwargs)
    triangles, data = generate_triangles_for_2d(
        solution, variable, mask=mask, **fish2fem_kwargs
    )
    topology = triangles.triangles
    geometry = list(zip(triangles.x, triangles.y))
    interpolator = LinearTriInterpolator(triangles, -data)
    (e_x, e_y) = interpolator.gradient(triangles.x, triangles.y)
    if gradient_norm:
        norm = field_norm(e_x, e_y)
        if plot:
            mesh_plot_2d(
                None,
                None,
                color_style=color_style,
                mask=mask,
                topology=topology,
                geometry=geometry,
                data=data,
                **kwargs
            )
        return norm
    else:
        plt.quiver(triangles.x, triangles.y, e_x, e_y, **mpl_kwargs)
        return e_x, e_y


def extract_edges(solution: H5Solution, variable: str, **kwargs):
    """Extract edges (boundaries) from the dataset.

    Convenience to select plottable boundaries

    :param solution: fish2eod solution to use
    :param variable: Name of the variable to use
    :param mask: Valid mask
    :param kwargs: Optional parameters from solution
    :return: Triangles and data for the selected data
    """
    edge_topology, geometry, data = extract(solution, variable, **kwargs)
    edge_filter = (data != Model._EXTERNAL_BOUNDARY) * (
        data != ModelGeometry.background_label
    )
    valid_data = data[edge_filter]
    edge_geometry = np.array(
        [[geometry[top[0]], geometry[top[1]]] for top in edge_topology[edge_filter]]
    )

    return edge_geometry, valid_data


def mesh_plot_1d(
    solution, variable, *, color_style: COLOR_STYLE_TYPE = None, colorbar=True, **kwargs
):
    """Create a 1D mesh plot.

    1D plots represent boundary functions

    :param solution: fish2eod solution to plot
    :param variable: Name of the variable to plot
    :param cmap: Name of the colormap
    :param color_style: Which type of symmetry see `compute_norm`
    :param color: Color to use (if not data i.e. outline)
    :return: None
    """
    mpl_kwargs, fish2fem_kwargs = split_mpl_kwargs(kwargs)

    if "color" in kwargs and "cmap" in kwargs:
        raise ValueError("You cannot mix a fixed color and a colormap")

    edge_geometry, data = extract_edges(solution, variable, **fish2fem_kwargs)
    norm = normalize_color(None, data, color_style)

    color = mpl_kwargs.pop("color", None)
    cmap = mpl_kwargs.pop("cmap", None)
    cbar = None
    if color is None:
        m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = m.to_rgba(data)[:, :3]

        if colorbar:
            cbar = plt.colorbar(m)
    else:
        colors = color

    mpl_kwargs["colors"] = colors
    plot_edges(edge_geometry, **mpl_kwargs)

    return plt.gca(), cbar


def plot_outline(solution, **kwargs) -> None:
    """Plot solution outline.

    :param solution: fish2eod solution to plot
    :param kwargs: Optional parameters from the solution
    :return: None
    """
    mesh_plot_1d(solution, "outline", colorbar=False, **kwargs)


def plot_edges(g, **kwargs) -> None:
    """Plot solution edges as lines.

    :param g: Coordinates of the edges
    :param norm: Color normalization
    :param colors: Which colors to use
    :return: None
    """

    mpl_params = upscale_parameter_name(kwargs)
    ln_coll = mpl.collections.LineCollection(g, **mpl_params)
    ax = plt.gca()
    ax.add_collection(ln_coll)
    plt.xlim([g[:, 0].min(), g[:, 0].max()])
    plt.ylim([g[:, 1].min(), g[:, 1].max()])
    ax.set_aspect("equal")


def extract(solution, variable, **kwargs):
    """Extract data from a solution given variable and parameter values.

    Variable and parameters are set in the solution and the data is extracted from the h5 file

    :param solution: fish2eod solution so load
    :param variable: Name of the variable to load
    :param kwargs: Optional parameters to extract from solution
    :return: Extracted data
    """
    dataset = solution[variable]
    for key, val in kwargs.items():
        dataset = dataset[key + "=" + str(val)]
    return dataset.load_data()
