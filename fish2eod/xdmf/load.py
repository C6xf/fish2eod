"""Load solutions in a filterable form from xdmf/manifests."""
from abc import abstractmethod
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import (
    Union,
    Tuple,
    Dict,
    Iterator,
    Any,
    Optional,
    List,
)

import h5py
import numpy as np
from lxml import etree as et

from fish2eod.helpers.type_helpers import DataSet
from fish2eod.xdmf.xml_tools import get_outer_grid


class Solution:
    def __init__(self):
        self.variables = None
        self.data = None
        self._selected_variable = None

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def load_data(self):
        pass

    def filter_variables(self, variable: str) -> "Solution":
        """Extract the variable from the solution.

        :param variable: Name of the variable
        :return: Filtered solution
        """
        assert variable in self.variables, "Unknown variable"

        new_selected_variable = variable
        new_data = filter(lambda x: x.name == variable, self.data)

        return self.spawn_from_variable(new_selected_variable, new_data)

    @abstractmethod
    def spawn_from_variable(self, new_selected_variable, new_data):
        pass


class H5Solution(Solution):
    """Store the solution or filtered subset.

    A solution is a collection of datasets. This can be filtered on parameter or variable. Once subset further
    processing sub setting is needed until a single solution remains

    :param h5_file: Path to the h5 data set
    :param parameter_levels: Dictionary of parameter : levels of parameter
    :param variables: Possible variables
    :param data: List of DataSet objects
    :param s_param: List of selected_parameters if any else None
    :param s_var: Selected variable if selected else None
    """

    def __init__(
        self,
        h5_file: Path,
        parameter_levels: Dict[str, Tuple[int]],
        domain_map: Dict[str, int],
        variables: Tuple[str],
        data: List[DataSet],
        s_param: Tuple[str, ...] = (),
        s_var: Optional[str] = None,
    ):
        """Instantiate solution."""
        super().__init__()
        self.h5_file = h5_file
        self.parameter_levels = parameter_levels
        self.domain_map = domain_map
        self.variables = variables
        self.data = data

        self._selected_parameter = s_param
        self._selected_variable = s_var

    def __getitem__(self, item: str) -> "H5Solution":
        """Use [] to subset the solution.

        Variables can be selected with ['variable_name']
        Parameters can be selected with ['parameter_name=ix'] where ix is a parameter level

        :param item: The variable/parameter to select
        :return: The subset solution
        """
        if "=" in item:
            return self.filter_parameters(*item.split("="))  # '=' in item
        else:
            return self.filter_variables(item)

    @property
    def parameters(self) -> Tuple[str]:
        """Get the parameter names.

        :return: Tuple of parameter names - no particular order
        """
        return tuple(self.parameter_levels.keys())

    def filter_parameters(self, parameter: str, value: str) -> "H5Solution":
        """Filter the solution where parameter = value.

        :param parameter: Name of the parameter
        :param value: String of selected parameter index
        :return: Filtered solution
        """
        assert parameter in self.parameter_levels.keys(), "Unknown parameter"
        assert int(value) in self.parameter_levels[parameter], "Invalid value"

        new_selected_parameter = self._selected_parameter + (f"{parameter}={value}",)

        # dump selected parameter from available params
        new_parameter_levels = self.parameter_levels.copy()
        new_parameter_levels.pop(parameter)

        new_data = filter(lambda x: x.parameter_state[parameter] == value, self.data)

        return H5Solution(
            self.h5_file,
            new_parameter_levels,
            self.domain_map,
            self.variables,
            list(new_data),
            s_param=new_selected_parameter,
            s_var=self._selected_variable,
        )

    def spawn_from_variable(self, new_selected_variable, new_data):
        return H5Solution(
            self.h5_file,
            self.parameter_levels,
            self.domain_map,
            tuple(),
            list(new_data),
            s_param=self._selected_parameter,
            s_var=new_selected_variable,
        )

    def load_data(self):
        """Load data.

        Checks if loadable i.e. no other parameters or variables to select
        Loads either a namedtuple for misc data or a tuple of arrays for fem dataset

        :return: Loaded data
        """
        assert len(self.parameter_levels) == 0, "Please select all parameters"
        assert len(self.variables) == 0, "Please select a variable"
        assert len(self.data) == 1, "Something has gone horribly wrong"

        data_set = self.data[0]  # we know only 1 value
        if isinstance(data_set.data, str):  # DataSet.data is a str for fem
            return load_fem_data(self.h5_file, data_set)
        else:  # assume list structure
            return load_misc_data(self.h5_file, data_set)


class DataSolution(Solution):
    def __getitem__(self, item: str) -> "DataSolution":
        """Use [] to subset the solution.

        Variables can be selected with ['variable_name']
        Parameters can be selected with ['parameter_name=ix'] where ix is a parameter level

        :param item: The variable/parameter to select
        :return: The subset solution
        """
        if "=" in item:
            raise ValueError("You may not select parameters from DataSolutions")

        return self.filter_variables(item)

    def __init__(
        self,
        domain_map: Dict[str, int],
        variables: Tuple[str],
        data: List[DataSet],
        s_var: str = None,
    ):
        super().__init__()
        self.domain_map = domain_map
        self.variables = variables
        self.data = data
        self._selected_variable = s_var

    def spawn_from_variable(self, new_selected_variable, new_data):
        return DataSolution(
            self.domain_map, tuple(), list(new_data), s_var=new_selected_variable
        )

    def load_data(self):
        if self.data[0].topology is not None:
            return self.data[0].topology, self.data[0].geometry, self.data[0].data
        else:
            return self.data[0].data


def load_misc_data(h5_file: Path, data_set: DataSet) -> Tuple:
    """Load misc data from the h5 file.

    :param h5_file: Path to the h5 file
    :param data_set: DataSet to load
    :return: The misc_data in its named tuple
    """
    # extract [(path1, name1), ,,,] to (path1, ...), (name1, ...)
    paths, name = zip(*data_set.data)
    data_holder = {
        k: v for k, v in zip(name, load_from_h5(h5_file, *paths))
    }  # make a dictionary from zipped name and loaded data

    # convert dict to appropriate named tuple
    return namedtuple(data_set.name, data_holder.keys())(**data_holder)


def load_fem_data(h5_file: Path, data_set: DataSet) -> Tuple[np.ndarray, ...]:
    """Load fem data from the h5 file.

    FEM data is a topology, geometry and an array representation of function
    TODO named tuple

    :param h5_file: Path to the h5 file
    :param data_set: DataSet to load
    :return: The tuple of data topology, geometry, data
    """
    topology_path = data_set.topology
    geometry_path = data_set.geometry
    data_path = data_set.data

    return tuple(load_from_h5(h5_file, topology_path, geometry_path, data_path))


def load_from_h5(h5_file: Path, *data_paths: str) -> Iterator[np.ndarray]:
    """Load data from the h5_file with a sequence of paths.

    Nx1 data is converted to an array of length N and unsignedintegers are converted to signed integers

    :param h5_file: Path to the h5 file
    :param data_paths: Arbitrary number of data_paths to load
    :return: Iterator of arrays for each data_path
    """
    with h5py.File(str(h5_file), "r", swmr=True) as f:
        for data_path in data_paths:
            data = f[data_path][()]
            yield data


def variables_from_data(path: Path) -> Tuple[str, ...]:
    """Extract all the variables from a dataset.

    :param path: Path to the manifest
    :return: Variable names
    """
    variable_names = set()
    for data in extract_all_data(path):
        variable_names.add(data.name)

    return tuple(variable_names)


def parameter_levels_from_data(path: Path) -> Dict[str, Tuple[int]]:
    """Extract all parameter levels (and parameters) from a dataset.

    :param path: Path to the manifest
    :return: Parameters and corresponding levels
    """
    p = defaultdict(set)
    for data in extract_all_data(path):
        for parameter, level in data.parameter_state.items():
            p[parameter].add(int(level))
    return {
        k: tuple(sorted(v)) for k, v in p.items()
    }  # set removes duplicate but not sorted


def domain_map_from_data(path: Path) -> Dict[str, int]:
    """Extract the domain map from the data.

    TODO duplication in constant map across time steps

    :param path: Path to the manifest
    :return: Map of domains name to number
    """
    data = next(extract_all_data(path))
    return {k: int(v) for k, v in data.domain_map.items()}


def extract_all_data(path: Path) -> Iterator[DataSet]:
    """Load data from each xdmf file.

    :param path: Path to manifest
    :return: Iterator of DataSets
    """
    manifest_file = find_manifest(path)
    xdmf_files, h5_file = parse_manifest(manifest_file)

    for xdmf in xdmf_files:
        yield from load_xdmf(xdmf)


def misc_attribute_to_data_set(grid) -> Iterator[Tuple[List, str]]:
    """Return data_set compatible structure from misc attributes.

    The goal is for each (super) attribute to have a structure
    ([(sub_attr1, path1), ...], name_of_super_attribute, ...) where each super attribute behaves like a normal attribute
    and the path with be a list of subpaths and subnames

    :param grid: Grid to convert (et.Element)
    :return: Structured misc data
    """
    return [
        (
            list(extract_path_from_grid(attr, "Attribute")),
            attr.attrib["Name"].split(".")[-1],
        )
        for attr in grid.findall("Attribute")
    ]  # (sub_attribute paths, name) to reconstruct namedtuple


def load_xdmf(xdmf_file: Path) -> Iterator[DataSet]:
    """Load an xdmf file.

    :param xdmf_file: The path to the xdmf file
    :return: Sequence of DataSet objects
    """
    # Load the file and get the outer_grid for iterating simulations
    xdmf_data = et.parse(str(xdmf_file))
    outer_grid = get_outer_grid(xdmf_data)

    # for all inner grids
    for grid in outer_grid.iterchildren():
        time_step = extract_unique_from_grid(grid, "Time")
        parameter_state = extract_unique_from_grid(grid, "Metadata")
        domain_map = extract_unique_from_grid(grid, "DomainMap")
        try:  # this is for 1d/2d datasets with geometry/topolgy
            topology, _ = next(extract_path_from_grid(grid, "Topology"))
            geometry, _ = next(extract_path_from_grid(grid, "Geometry"))
            data_set = extract_path_from_grid(grid, "Attribute")
        except StopIteration:  # no topology or geometry in misc dataset
            topology = geometry = None
            data_set = misc_attribute_to_data_set(grid)

        yield from (
            DataSet(
                time_step=time_step["Value"],
                topology=topology,
                geometry=geometry,
                data=data_path,
                name=data_name,
                parameter_state=parameter_state,
                domain_map=domain_map,
            )
            for data_path, data_name in data_set
        )  # TODO I think this is nicer than for ... yield


def extract_unique_from_grid(grid, term: str) -> Dict[str, Any]:
    """Extract a unique term from the grid.

    A unique term like <Time /> or <Metadata /> only is entered once per grid

    :param grid: The grid (et.Element) to search
    :param term: The term to search for
    :return: The attributes of the found element
    """
    term = grid.findall(term)
    assert len(term) == 1, f"Term {term} is not unique"
    return term[0].attrib


def extract_path_from_grid(grid, parent: str) -> Iterator[Tuple[str, Optional[str]]]:
    """Extract h5 data paths from the grid.

    :param grid: The grid to search (et.Element)
    :param parent: The parent to get paths of (Attribute for example for data)
    :return: Sequence of child paths and the name of the dataset or None if not available
    """
    parent_xml = grid.findall(parent)
    for p in parent_xml:
        children = p.getchildren()
        assert len(children) == 1
        name = p.attrib["Name"] if "Name" in p.attrib else None
        yield children[0].text.split(":")[1], name


def parse_manifest(manifest: Path) -> Tuple[Iterator[Path], Path]:
    """Parse the xdmf/h5 files from the manifest.

    :param manifest: Manifest path
    :return: The xdmf paths and the h5 file
    """
    manifest_xml = et.parse(str(manifest))

    xdmf_files = manifest_xml.getroot().findall("data")
    xdmf_paths = tuple((manifest.parent / xdmf.attrib["path"] for xdmf in xdmf_files))
    h5 = manifest.parent / manifest_xml.getroot().find("h5").attrib["path"]

    return xdmf_paths, h5


def load_from_file(path: Union[str, Path]) -> H5Solution:
    """User-facing api to load the data.

    :param path: Path to manifest.xml or the model folder
    :return: The Solution object
    """
    manifest_file = find_manifest(path)
    xdmf_files, h5_file = parse_manifest(manifest_file)
    return H5Solution(
        h5_file,
        parameter_levels_from_data(path),
        domain_map_from_data(path),
        variables_from_data(path),
        list(extract_all_data(path)),
    )


def find_manifest(path: Union[str, Path], search_name="manifest.xml") -> Path:
    """Find the manifest file in the given path.

    Since the path can be specified as either the path to the manifest or the containing directory this will check
    if the specified path is the manifest or if the folder given contains the manifest.

    :param path: Path to manifest or search path
    :param search_name: Name of file to search for
    :return: Path to the manifest
    """
    path = Path(path)

    assert path.exists(), "Manifest or model path does not exist"

    if path.is_dir():  # assuming manifest
        path = path.joinpath(search_name)
        assert (
            path.exists()
        ), f"Manifest {search_name} does not exist please point direct to manifest"

    return path
