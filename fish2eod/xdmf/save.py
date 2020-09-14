"""Save solutions to an expandable structure.

Saves 1D data I.e. function defined on boundaries and 2D data I.e. function defined on domains/cells and misc data
(see later)

There is one manifest file which stores the the locations of the different data types which are stored in separate
XDMF(ish) files for paraview compatibility (not strictly XDMF as they contain additional tags but could be considered a
superset of XDMF).

<Manifest>
    <data type='1d' path="PATH_TO_1D_DATA.xdmf"/>
    <data type='2d' path="PATH_TO_2D_DATA.xdmf"/>
    <data type='misc' path="PATH_TO_misc.xdmf"/>
    <h5 path="PATH_TO_h5_file.h5"/>
</Manifest>

1D dataset contains boundary functions

The XDMF tag contains a domain containing a grid which specifies a "set of parameters" with each parameter being stored
in a nested grid tag. Within a simulation the topology (what connects to what) and the geometry (where was is) and the
is stored in topology and geometry tags with the function defined on the Attribute tag

<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
      <Grid Name="mesh" GridType="Uniform">
        <Time Value="0"/>
        <Metadata TimeName="ParaViewTime" step="0" location="0" amp="0"/>
        <Topology NumberOfElements="173" TopologyType="PolyLine" NodesPerElement="2">
          <DataItem Dimensions="173 2" NumberType="UInt" Format="HDF">MODEL.h5:/Mesh/2/mesh/edges</DataItem>
        </Topology>
        <Geometry GeometryType="XY">
          <DataItem Dimensions="62 2" Format="HDF">MODEL.h5:/Mesh/1/mesh/geometry</DataItem>
        </Geometry>
        <Attribute Name="boundary" AttributeType="Scalar" Center="Cell">
          <DataItem Dimensions="173 1" Format="HDF">MODEL.h5:/VisualisationVector/3</DataItem>
        </Attribute>
    </Grid>
  </domain>
</Xdmf>

2D dataset contains domain/cell functions

The XDMF tag contains a domain containing a grid which specifies a "set of parameters" with each parameter being stored
in a nested grid tag. Within a simulation the topology (what connects to what) and the geometry (where was is) and the
is stored in topology and geometry tags with the function defined on the Attribute tag

<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">
  <Domain>
      <Grid Name="mesh" GridType="Uniform">
        <Time Value="0"/>
        <Metadata TimeName="ParaViewTime" step="0" location="0" amp="0"/>
        <Topology NumberOfElements="114" TopologyType="Triangle" NodesPerElement="3">
          <DataItem Dimensions="114 3" NumberType="UInt" Format="HDF">MODEL.h5:/Mesh/0/mesh/topology</DataItem>
        </Topology>
        <Geometry GeometryType="XY">
          <DataItem Dimensions="62 2" Format="HDF">MODEL.h5:/Mesh/1/mesh/geometry</DataItem>
        </Geometry>
        <Attribute Name="V" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="62 1" Format="HDF">MODEL.h5:/VisualisationVector/5</DataItem>
        </Attribute>
        <Attribute Name="sigma" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="62 1" Format="HDF">MODEL.h5:/VisualisationVector/6</DataItem>
        </Attribute>
        <Attribute Name="domain" AttributeType="Scalar" Center="Cell">
          <DataItem Dimensions="114 1" Format="HDF">MODEL.h5:/VisualisationVector/7</DataItem>
        </Attribute>
      </Grid>
  </domain>
</Xdmf>
"""
from pathlib import Path
from typing import Union

import dataclasses
import h5py
import numpy as np
from lxml import etree as et

from fish2eod.helpers.dolfin_tools import get_dimension, get_data
from fish2eod.models import Model
from fish2eod.xdmf.xml_tools import (
    create_minimal_xdmf,
    make_manifest_file,
    create_timestep_grid_misc,
    create_timestep_grid,
    get_outer_grid,
    write_xml,
)


class Saver:
    """Saves relevant data from a model into paraview-compatible set of xdmf files.

    :param model_name: Name of the model
    :param save_path: Path to save the model to (model will be saves to save_path/model_name)
    """

    def __init__(self, model_name: str, save_path: Union[Path, str]):
        """Instantiate Saver."""
        self.model_name = model_name
        self.save_path = Path(save_path).joinpath(model_name).joinpath("DATA")

        self.save_path.mkdir(parents=True, exist_ok=True)

        self.data_1d, self.data_2d, self.data_misc = [
            create_minimal_xdmf() for _ in range(3)
        ]
        self.current_grid_1d = self.current_grid_2d = self.current_grid_misc = None

        self.step = -1  # start at -1 so first +=1 makes step = 0

        make_manifest_file(self.model_name, self.save_path.parent)

    def create_next_grid(
        self,
        num_topology_elements_1d=None,
        num_topology_elements_2d=None,
        num_geometry_elements=None,
        metadata=None,
        domain_map=None,
    ):
        """Create grid for a the next step for the 1d and 2d xdmf files.

        Creates the grid and assigns them as a pointer in current_grid_1d and current_grid_2d attributes

        # TODO refactor the duplication out of this function. Any loop right now will be awkward

        :param num_geometry_elements: Number of elements in the geometry (number of nodes)
        :param num_topology_elements_1d: Number of elements in the 1d topology (lines)
        :param num_topology_elements_2d: Number of elements in the 2d topology (triangles)
        :param metadata: Metadata dictionary
        """
        self.step += 1
        self.current_grid_1d = create_timestep_grid(
            step_number=str(self.step),
            metadata=metadata,
            domain_map=domain_map,
            num_topology_elements=num_topology_elements_1d,
            dim=1,
            data_location_topology=f"{self.model_name}.h5:{self.topology_path_1d}",
            num_geometry_elements=num_geometry_elements,
            data_location_geometry=f"{self.model_name}.h5:{self.geometry_path}",
        )
        data_1d_outer_grid = get_outer_grid(self.data_1d)
        data_1d_outer_grid.append(self.current_grid_1d)

        self.current_grid_2d = create_timestep_grid(
            step_number=str(self.step),
            metadata=metadata,
            domain_map=domain_map,
            num_topology_elements=num_topology_elements_2d,
            dim=2,
            data_location_topology=f"{self.model_name}.h5:{self.topology_path_2d}",
            num_geometry_elements=num_geometry_elements,
            data_location_geometry=f"{self.model_name}.h5:{self.geometry_path}",
        )
        data_2d_outer_grid = get_outer_grid(self.data_2d)
        data_2d_outer_grid.append(self.current_grid_2d)

        self.current_grid_misc = create_timestep_grid_misc(
            step_number=str(self.step), metadata=metadata, domain_map=domain_map
        )
        data_misc_outer_grid = get_outer_grid(self.data_misc)
        data_misc_outer_grid.append(self.current_grid_misc)

    @property
    def geometry_path(self) -> str:
        """H5 path for the geometry arrays (/Geometry/step)."""
        return f"/Geometry/{self.step}"

    @property
    def topology_path_1d(self) -> str:
        """H5 path for the 1d topology arrays (/Topology/1d/step)."""
        return f"/Topology/1d/{self.step}"

    @property
    def topology_path_2d(self) -> str:
        """H5 path for the 2d topology arrays (/Topology/2d/step)."""
        return f"/Topology/2d/{self.step}"

    def write_geometry(self, geometry: np.ndarray) -> None:
        """Write the geometry (coordinates) to the h5 file.

        :param geometry: Array of x,y coordinates for the geometry
        """
        self.write_h5(self.geometry_path, geometry)

    def write_topology(self, topology: np.ndarray) -> None:
        """Write the topology to the dimension appropriate h5 file path.

        :param topology: Array of topology
        """
        if topology.shape[1] == 2:
            self.write_h5(self.topology_path_1d, topology)
        elif topology.shape[1] == 3:
            self.write_h5(self.topology_path_2d, topology)
        else:
            raise ValueError("Unknown topology")

    def add_misc_data(self, d) -> None:
        """Add the miscellaneous data to the xdmf and the h5 file.

        Misc data is treated separately because for each data structure there is an outer Attribute which holds a
        collection of data sets from d (namedtuple)

        :param d: Miscelaneous data container, must be a namedtuple
        :return: None
        """
        name = d.__class__.__name__  # name is used to later reconstruct the namedtuple
        d = d.__dict__  # named tuple has this

        # add index for multiple fish
        data_group = et.Element("Attribute", Name=name, AttributeType="DataGroup")
        for data_name, data in d.items():
            attribute = et.Element("Attribute", Name=data_name)
            data_group.append(attribute)

            data_item = et.Element("DataItem")
            data_item.text = (
                f"{self.model_name}.h5:/Data/{name}_{data_name}/{self.step}"
            )
            attribute.append(data_item)

            self.write_h5(f"/Data/{name}_{data_name}/{self.step}", data)
        self.current_grid_misc.append(data_group)

    def add_fenics_data(self, f) -> None:
        """Add the dataset to the h5 file and the reference it in the XDMF file.

        If the data `f` is a tuple of misc data its passed to add_misc_data otherwise its passed to add_fencis_data

        :param f: Data to save Meshfunction/function
        """
        data = get_data(f)
        n_elements = data.shape[0]  # number of nodes
        dim = get_dimension(f)

        # define Paraview plotting properties
        center = "Cell" if dim == 1 else "Node"
        if f.name() == "domain":
            center = "Cell"  # domain is a special case where 2d has a cell type

        # which grid to use
        ref = self.current_grid_1d if dim == 1 else self.current_grid_2d

        # create new xdmf entry
        attribute = et.Element(
            "Attribute", Name=f.name(), AttributeType="Scalar", Center=center,
        )
        data_item = et.Element("DataItem", Dimensions=f"{n_elements} 1", Format="HDF")
        data_item.text = f"{self.model_name}.h5:/Data/{f.name()}/{self.step}"
        attribute.append(data_item)
        ref.append(attribute)

        self.write_h5(f"/Data/{f.name()}/{self.step}", data)

    def add_data(self, f) -> None:
        """Add the dataset to the h5 file and the reference it in the XDMF file.

        If the data `f` is a tuple of misc data its passed to add_misc_data otherwise its passed to add_fencis_data

        :param f: Data to save Meshfunction/function/tuple
        """
        if dataclasses.is_dataclass(f):
            self.add_misc_data(f)
        else:
            self.add_fenics_data(f)

    def save(self):
        """Save the XDMF files.

        Because XDMF are so small - this overwrites the entire XDMF file with the new data: which is the old data + new
        """
        for file_name, data_set in zip(
            ["1d.xdmf", "2d.xdmf", "misc.xdmf"],
            [self.data_1d, self.data_2d, self.data_misc],
        ):
            write_xml(
                self.save_path / file_name,
                data_set,
                xml_declaration=False,
                doctype='<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
            )

    def write_h5(self, path: str, d: np.ndarray):
        """Write to the h5 file.

        :param path: Internal h5 path to write to
        :param d: Dataset to write
        """
        with h5py.File(self.save_path / f"{self.model_name}.h5", "a") as f:
            f.require_group("Geometry")
            f.require_group("Topology/1d")
            f.require_group("Topology/2d")
            f.require_group("Data")

            dset = f.create_dataset(path, data=d)
            dset.attrs["partition"] = [0]  # from dolfin h5 data

    def save_model(self, model: Model, metadata=None):
        """Save the specified model to an XDMF representation.

        :param model: The model to save
        :param metadata: Metadata (parameter states) to save along with the data
        """
        # set metadata = {} if None and ensure all values are str type they'll start as int
        metadata = {} if not metadata else metadata
        metadata = {k: str(v) for k, v in metadata.items()}
        domain_map = {k: str(v) for k, v in model.model_geometry.domain_names.items()}

        data = model.generate_save_data()

        geometry = model.topology_coordinates
        topology_2d = model.topology_2d
        topology_1d = model.topology_1d

        self.create_next_grid(
            num_topology_elements_1d=topology_1d.shape[0],
            num_topology_elements_2d=topology_2d.shape[0],
            num_geometry_elements=geometry.shape[0],
            metadata=metadata,
            domain_map=domain_map,
        )

        self.write_geometry(geometry)
        self.write_topology(topology_1d)
        self.write_topology(topology_2d)

        [self.add_data(f) for f in data]  # add all data
        self.save()
