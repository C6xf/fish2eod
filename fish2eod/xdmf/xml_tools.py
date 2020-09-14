"""Common tools for working with XDMF/XML files."""
import inspect
from pathlib import Path
from typing import Callable

from lxml import etree as et


def create_minimal_xdmf():
    """Create the minimal XDMF template.

    Most of these properties are necessary for Paraview compatibility. Some may not be required but they're used in the
    Fenics XDMF files.
    """
    xdmf = et.Element(
        "Xdmf", nsmap={"xi": "http://www.w3.org/2001/XInclude"}, Version="3.0"
    )  # parent Xdmf tag

    domain = et.SubElement(xdmf, "Domain")  # domain contains the grid

    et.SubElement(
        domain,
        "Grid",
        Name="TimeSeries",
        GridType="Collection",
        CollectionType="Temporal",
    )  # this grid specifies that there will be several solutions - this is referenced where discussing the "outer grid"

    return xdmf


def get_outer_grid(xdmf, parent_tag="Domain"):
    """Get the outer grid element from the XDMF structure.

    :param xdmf: The xdmf structure to parse (et.Element)
    :param parent_tag: The name of the domain tag (parent)
    """
    domains = xdmf.findall(parent_tag)
    assert len(domains) == 1  # there should only be one domain

    children = domains[0].getchildren()
    assert len(children) == 1  # there should only be one child which is the outer grid

    return children[0]


def make_manifest_file(model_name: str, save_path: Path, data_folder="DATA") -> None:
    """Create the manifest file.

    The manifest file holds the paths to the 1d, 2d, misc data as well as the path to the h5 file

    :param model_name: Name of the model
    :param save_path: Path of parent folder of the model
    :param data_folder: Optional argument specifying name of the sub-folder containing the data
    :return: None
    """
    manifest = et.Element("Manifest")
    et.SubElement(manifest, "data", type="1d", path=str(Path(data_folder) / "1d.xdmf"))
    et.SubElement(manifest, "data", type="2d", path=str(Path(data_folder) / "2d.xdmf"))
    et.SubElement(
        manifest, "data", type="misc", path=str(Path(data_folder) / "misc.xdmf")
    )
    et.SubElement(manifest, "h5", path=str(Path(data_folder) / f"{model_name}.h5"))

    write_xml(save_path.joinpath("manifest.xml"), manifest)


def assert_all_defined(f: Callable) -> Callable:
    """Wrap f and check that all kwargs of a function are defined at function call and not None.

    :param f: The function to check. Needs to be kwarg only function
    """

    def wrapper(**kwargs):
        """Check for None kwargs at runtime."""
        target_keys = inspect.signature(f).parameters.keys()
        assert len(kwargs) == len(target_keys), "Not enough parameters passed"
        assert all((p is not None for p in kwargs.values())), "Some parameters none"
        return f(**kwargs)

    return wrapper


@assert_all_defined
def create_timestep_grid_misc(*, step_number=None, metadata=None, domain_map=None):
    """Create the grid for a "time-step" in the miscellaneous data-set.

    Time step refers to an individual simulation which could be time or some parameter sweep
    Typical parameters like topology and dimension are not specified due to miscellaneous dataset.

    :param step_number: Current iteration of the model. Paraview will plot in this order
    :param metadata: Dictionary of parameter_name:value to be saved along with the simulation
    :return: Newly created grid (et.Element)
    """
    grid = et.Element("Grid", Name="mesh", GridType="Uniform")
    grid.append(et.Element("Time", Value=str(step_number)))
    grid.append(et.Element("Metadata", **metadata))
    grid.append(et.Element("DomainMap", **domain_map))

    return grid


@assert_all_defined
def create_timestep_grid(
    step_number=None,
    num_topology_elements=None,
    dim=None,
    data_location_topology=None,
    num_geometry_elements=None,
    data_location_geometry=None,
    metadata=None,
    domain_map=None,
):
    """Create the inner grid defining a time-step.

    This function has many inputs which must be specified as keywords. Additionally they must all be specified non-None
    type. The assert_all_defined decorator ensures this.

    :param step_number: Current iteration of the model. Paraview will plot in this order
    :param num_topology_elements: Number of elements in the Topology
    :param dim: Dimension of the data
    :param data_location_topology: h5 location of the Topology
    :param num_geometry_elements: Number of elements in the geometry
    :param data_location_geometry: h5 location of the geometry
    :param metadata: Dictionary of parameter_name:value to be saved along with the simulation
    :return: Inner grid (et.Element)
    """
    grid = et.Element("Grid", Name="mesh", GridType="Uniform")
    grid.append(et.Element("Time", Value=str(step_number)))
    grid.append(et.Element("Metadata", **metadata))
    grid.append(et.Element("DomainMap", **domain_map))

    topology = make_topology(num_topology_elements, dim, data_location_topology)
    geometry = make_geometry(num_geometry_elements, data_location_geometry)

    grid.append(topology)
    grid.append(geometry)

    return grid


def make_topology(num_elements: int, dim: int, location: str):
    """Create the XDMF topology element.

    :param num_elements: Number of elements in the topology (# triangles/lines)
    :param dim: Dimension of the solution
    :param location: Location of the topology in the h5 file
    :return: The topology element (et.Element)
    """
    node_map = {2: "Triangle", 1: "PolyLine"}
    topology = et.Element(
        "Topology",
        NumberOfElements=str(num_elements),
        TopologyType=node_map[dim],
        NodesPerElement=str(dim + 1),
    )

    topology_data = et.Element(
        "DataItem",
        Dimensions=f"{num_elements} {dim + 1}",
        NumberType="UInt",
        Format="HDF",
    )
    topology_data.text = location
    topology.append(topology_data)

    return topology


def make_geometry(num_elements: int, location: str):
    """Create the XDMF geometry element.

    :param num_elements: Number of elements in the geometry (# nodes)
    :param location: Location of the geometry in the h5 file
    :return: The geometry element (et.Element)
    """
    geometry = et.Element("Geometry", GeometryType="XY")

    geometry_data = et.Element("DataItem", Dimensions=f"{num_elements} 2", Format="HDF")
    geometry_data.text = location
    geometry.append(geometry_data)

    return geometry


def write_xml(save_path: Path, xml_obj, xml_declaration=True, doctype=None) -> None:
    """Write an ElementTree object to an xml/xdmf file.

    :param save_path: Path to save the file to
    :param xml_obj: The ElementTree to save
    :param xml_declaration: Include XML declaration
    :param doctype: Optional doctype for XDMF files (None for no doctype)
    :return: None
    """
    with save_path.open(mode="wb") as doc:
        doc.write(
            et.tostring(
                et.ElementTree(xml_obj),
                pretty_print=True,
                xml_declaration=xml_declaration,
                doctype=doctype,
            )
        )
