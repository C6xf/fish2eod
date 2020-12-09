# coding=UTF-8
"""Templates for solving finite element problems.

Model is the base class which defines an interface for creating geometry, meshing, labeling and solving without
a full implementation

QESModel adds quasi-electrostatic physics to the Model class

BaseFishModel is the template class for electric fish simulations by specifying convenience api calls for adding
geometry, and voltage/current sources while also defining and handling the fish objects
"""
from abc import ABC, abstractmethod
from itertools import repeat
from typing import Optional, Sequence, Tuple, Iterable

import dataclasses
import dolfin as df
import numpy as np
from dolfin.cpp.mesh import MeshFunctionSizet
from scipy.interpolate import CubicSpline

from fish2eod.analysis.transdermal import compute_transdermal_potential
from fish2eod.geometry.fish import get_eod, FishContainer
from fish2eod.geometry.primitives import Circle, Rectangle
from fish2eod.helpers.dolfin_tools import (
    Equation,
    get_data,
    get_dimension,
)
from fish2eod.helpers.type_helpers import (
    BoundaryCondition,
    FISH_COORDINATES,
    ElectricImageParameters,
    DataSet,
    BOUNDARY_MARKER,
    EOD_TYPE,
)
from fish2eod.mesh.boundary import mark_boundaries
from fish2eod.mesh.domain import mark_domains
from fish2eod.mesh.mesh import create_mesh
from fish2eod.mesh.model_geometry import ModelGeometry, QESGeometry
from fish2eod.properties import Property
from fish2eod.xdmf.load import DataSolution


def fenics_representation(mixed_representation: BoundaryCondition) -> df.Expression:
    """Get a uniform representation of a boundary condition in a df.Expression.

    :param mixed_representation: A generic BoundaryCondition which is defined as a number, string, or df.Expression
    :return: A unified df.Expression for the boundary condition
    """
    if isinstance(mixed_representation.value, (int, float, str)):
        return df.Expression(str(mixed_representation.value), degree=2)

    return mixed_representation.value


class Model(ABC):
    """Model interface for meshing, properties and plotting.

    Models are physics and geometry agnostic and knows nothing about these things - this is an internal class and
    would only be subclassed to develop new physics modules.
    """

    _verbose = 0  # 0 = off, 1 = stage details, 2=dolfin messages, 3=gmsh messages
    _EXTERNAL_BOUNDARY = 9999
    _VERBOSE_OFF = 0
    _VERBOSE_STAGE = 1
    _VERBOSE_DOLFIN = 2
    _VERBOSE_GMSH = 3

    @abstractmethod
    def generate_save_data(self) -> Tuple:
        """Generate save data for the base model.

        Subsequent physics specialization should super() and concat their data to save so all relevant is saved

        :return: solution, domains, boundaries, and outline
        """
        return self._fem_solution, self.domains, self.boundaries, self.outline

    @property
    def _fem_solution(self) -> df.TrialFunction:
        """Get copy of the model solution."""
        sol_copy = self.COMPILED_EQUATIONS.u.copy(deepcopy=True)
        sol_copy.rename("solution", "solution")
        return sol_copy

    def eval(self, x, y) -> float:
        return self._fem_solution(x, y)

    @property
    def domain_map(self):
        return {
            name: int(label) for name, label in self.model_geometry.domain_names.items()
        }

    def structure_to_dataset(self, structure):
        if dataclasses.is_dataclass(structure):  # custom fish-like data
            data_topology = None
            data_geometry = None
            name = structure.__class__.__name__
            data = structure

        else:  # fenics-like data
            data = get_data(structure)
            dim = get_dimension(structure)

            data_topology = self.topology_1d if dim == 1 else self.topology_2d
            data_geometry = self.topology_coordinates
            name = structure.name()

        return DataSet(
            time_step="0",
            topology=data_topology,
            geometry=data_geometry,
            data=data,
            name=name,
            parameter_state={},
            domain_map=self.domain_map,
        )

    @property
    def solution(self):
        all_data = [
            self.structure_to_dataset(structure)
            for structure in self.generate_save_data()
        ]
        var_names = tuple(d.name for d in all_data)
        return DataSolution(
            domain_map=self.domain_map, variables=var_names, data=all_data
        )

    def __init__(self):
        """Initialize the FEM model."""
        self.verboseness = self._verbose  # TODO ?
        self.SAVE_DATA = []
        self.COMPILED_EQUATIONS = dict()
        self.function_space_v = None

        self.model_geometry = ModelGeometry()
        self.mesh: Optional[df.Mesh] = None
        self.domains: Optional[MeshFunctionSizet] = None
        self.boundaries: Optional[MeshFunctionSizet] = None
        self.outline: Optional[MeshFunctionSizet] = None

        self.COMPILED_EQUATIONS: Optional[Equation] = None
        self.function_space_v = None

    @property
    def topology_2d(self) -> np.ndarray:
        """ Get mesh cell topology (triangles)

        :return: 2D topology
        """
        return self.mesh.cells()

    @property
    def topology_1d(self) -> np.ndarray:
        """ Get mesh edge topology (lines)

        :return: 1D topology
        """
        return np.array(list(map(lambda x: x.entities(0), df.edges(self.mesh))))

    @property
    def topology_coordinates(self) -> np.ndarray:
        """ Get mesh node coordinates

        :return: Mesh nodes
        """
        return self.mesh.coordinates()

    @abstractmethod
    def create_geometry(self, **kwargs):
        """ Create the model geometry

        Must be overwritten in subclasses to define geometry
        """
        return

    @property
    def verboseness(self):
        """Get the verboseness level of the model."""
        return self._verbose

    @verboseness.setter
    def verboseness(self, v):
        self._verbose = v
        df.set_log_active(self._verbose >= self._VERBOSE_DOLFIN)

    @abstractmethod
    def build_equations(self, **model_parameters):
        """Build the equations to solve.

        Must be overwritten in subclasses to define physics
        :param model_parameters: kwargs catch all for model parameters
        """
        return

    @abstractmethod
    def solve(self, **model_parameters):
        """Solve the system.

        Must be overwritten in subclasses to define how to solve
        :param model_parameters: kwargs catch all for model parameters
        """
        return

    def bc_to_dirichlet_bc(self, bc: BoundaryCondition) -> df.DirichletBC:
        """Convert fish2eod boundary conditions to their fenics counterpart.

        :returns: Boundary condition in fenics df.DirichletBC form
        """
        return df.DirichletBC(
            self.function_space_v, fenics_representation(bc), self.boundaries, bc.label
        )

    def compile(self, recompute_mesh=True, **model_parameters) -> None:
        """Create model mesh, label domains/boundaries and build model equations.

        Needs to be called before solving the models

        :param model_parameters: Model parameter dictionary.
        :param recompute_mesh: kwargs catch all for model parameters
        """
        # forces the geometry to update. If the mesh doesnt need to be recomputed the non-geometric parameters
        # (i.e. e.g. sigma) will update. The "geometry" will be recreated but since the physical parts haven't changed
        # the mesh can remain the same.
        self.model_geometry.clear()
        self.create_geometry(**model_parameters)

        # remove existing geometry and recreate if no mesh or needs recompute
        if recompute_mesh or not self.mesh:
            self.mesh = create_mesh(
                self.model_geometry, verbose=self.verboseness >= self._VERBOSE_GMSH
            )

            self.domains = mark_domains(self.mesh, self.model_geometry)

            self.boundaries, self.outline = mark_boundaries(
                self.domains,
                self.model_geometry,
                self.get_boundary_markers(),
                external_boundary=self._EXTERNAL_BOUNDARY,
            )

        self.build_equations(**model_parameters)

    def get_property(self, property_name: str) -> Property:
        """Get the property object of a named property.

        :param property_name: Name of the property (typically sigma)
        :returns: The property object
        """
        return Property(self.domains, self.model_geometry.parameters[property_name])

    def get_boundary_markers(self) -> Tuple[BOUNDARY_MARKER, ...]:
        """Get the boundary markers for a model.

        A marker is a list of two function with the same signature b1,b2,x,y where b1 is the domain id of the first
        domain, and b2 is the domain id of the second domain. x, and y are the coordinates of the midpoint of the
        boundary. Using whatever criteria you wish, a true or false is returned if that boundary follows the criteria.
        The second function with the same signature b1,b2,x,y returns the boundary ID you want to assign the target
        boundary.

        :return: List of the boundary markers
        """
        Warning("Not Implemented no custom boundaries")
        return tuple()

    # def refine(self, _):  # todo implement
    #     """Refine mesh should be overwritten in developed model class as default behaviour is to do nothing.
    #
    #     :param _: Ignored
    #     """
    #     Warning("Refinement called with no algorithm: no refinement taking place")

    def update_parameter(
        self,
        domain_label: int,
        parameter_name: str,
        parameter_value: float,
        **model_parameters,
    ) -> None:
        """Update non-geometric parameters such as conductivity without recomputing mesh.

        :param domain_label: Label of domain to update parameter on
        :param parameter_name: Name of parameter: for example: sigma
        :param parameter_value: New value of parameter
        :param model_parameters: Catchall for model parameters
        """
        self.model_geometry.parameters[parameter_name][domain_label] = parameter_value
        self.build_equations(**model_parameters)

    def plot_domains(self, color='viridis'):
        """Plot labeled domains with color corresponding to their label.

        :parameter color: Name of the colormap to use
        """
        if self.domains is None:
            raise ValueError("models Not Compiled")

        p = df.plot(self.domains)
        p.set_cmap(color)

    def plot_solution(self, **kwargs):
        df.plot(self._fem_solution, **kwargs)

    def plot_mesh(self, color="grey"):
        """Draw mesh.

        :param color: Color of edges, defaults to 'grey'
        """
        if self.mesh is None:
            raise ValueError("No Geometry")

        df.plot(self.mesh, color=color)

    def plot_geometry(self, color: str = 'Dark2', legend: bool = False):
        """Draw geometry.

        :param color: Colormap of edges
        :param legend: Whether to label edges in legend
        """
        if self.model_geometry is None:
            raise ValueError("No Geometry")

        self.model_geometry.draw(color=color, legend=legend)


class QESModel(Model, ABC):
    """FEM framework for solving quasi electrostatic models.

    Extends model QES physics work and the ability to add Neumann and Dirichelet conditions

    :param allow_overlaps: Should the geometry fail if two objects overlap (not including background)
    """

    def __init__(self, allow_overlaps: bool = True):
        """Instantiate QESModel."""
        super().__init__()
        self.model_geometry = QESGeometry(allow_overlaps=allow_overlaps)

    def solve(self, **model_parameters):
        """Solve the system.

        :param model_parameters: kwargs of the model parameters
        """
        a, u, rhs = (
            self.COMPILED_EQUATIONS.a,
            self.COMPILED_EQUATIONS.u,
            self.COMPILED_EQUATIONS.rhs,
        )
        # convert all bcs to their fenics form
        dirichelet_conditions = map(
            self.bc_to_dirichlet_bc, self.get_dirichlet_conditions(**model_parameters)
        )

        # solve a==rhs with test function u /w dirichelet bcs
        df.solve(a == rhs, u, list(dirichelet_conditions))

    def get_neumann_conditions(self, **model_parameters) -> Tuple[BoundaryCondition]:
        """Return the Neumann conditions (current sources).

        By default there are none. This should be overridden in your model subclass to implement if wanted

        :param model_parameters: Catchall kwarg for model parameters
        """
        Warning("No Sources")
        return tuple()

    def get_dirichlet_conditions(self, **model_parameters) -> Tuple[BoundaryCondition]:
        """Return the Dirichlet conditions (voltage sources).

        By default the external boundaries are grounded. This should be overridden in your model subclass to implement
        if wanted

        :param model_parameters: Catchall kwarg for model parameters
        """
        Warning("No Boundary Conditions: defaulting to grounded exterior")
        return (BoundaryCondition(value=0, label=self._EXTERNAL_BOUNDARY),)

    def generate_save_data(self) -> Tuple:
        """Get save data for the model.

        :return: None
        """
        sigma_function = df.interpolate(
            self.get_property("sigma"), self.function_space_v
        )
        sigma_function.rename("sigma", "sigma")
        return super().generate_save_data() + (sigma_function,)

    def build_equations(self, **model_parameters):
        """Create equations in weak form to solve with included sources (neumann conditions).

        :param model_parameters: Catchall kwarg for model parameters
        """
        dx = df.Measure("dx", subdomain_data=self.domains)
        ds = df.Measure("dS", subdomain_data=self.boundaries)

        self.function_space_v = df.FunctionSpace(self.mesh, "CG", 2)
        u = df.TrialFunction(self.function_space_v)
        v = df.TestFunction(self.function_space_v)

        sigma = self.get_property("sigma")  # conductance

        a = df.inner(sigma * df.grad(u), df.grad(v)) * dx
        u = df.Function(self.function_space_v)

        # laplace equation so rhs is 0 but needs to be written out
        rhs = df.Constant(0) * v * dx

        source_terms = (
            bc_to_rhs_term(bc, v, ds)
            for bc in self.get_neumann_conditions(**model_parameters)
        )
        rhs += sum(source_terms)
        self.COMPILED_EQUATIONS = Equation(a=a, u=u, rhs=rhs)


def bc_to_rhs_term(
    bc: BoundaryCondition, v: df.TestFunction, ds: df.Measure
) -> df.Form:
    """Convert a boundary condition (Neumann) into a source term on a trial function space."""
    source = fenics_representation(bc)
    return source("+") * v("+") * ds(bc.label)


class BaseFishModel(QESModel):
    """Main class that should be used for all fish simulations.

    Defines fish, tank, and ground electrode) along with properties need to solve them. This should be subclassed to
    implement custom geometries or sources.
    """

    WATER_CONDUCTIVITY = 0.023 / 100  # S/m
    GROUND_CONDUCTIVITY = 7.27e4 / 100  # WATER_CONDUCTIVITY
    BODY_CONDUCTIVITY = 0.356 / 100  # S/m
    ORGAN_CONDUCTIVITY = 0.92717 / 100

    def __init__(self):
        """Instantiate BaseFishModel with default parameters."""
        super().__init__(allow_overlaps=True)

        self.WATER_NAME = "water"
        self.SKIN_NAME = "skin"
        self.BODY_NAME = "body"
        self.ORGAN_NAME = "organ"
        self.GROUND_NAME = "ground"

        self.GROUND_RADIUS = 0.5
        self.GROUND_LOCATION = [-20, 20]

        self.TANK_SIZE = 70

        self.fish_container = FishContainer()

        self.active_solution = None

    # def refine(self, markers):
    #     self.fish.refine(markers)

    def setup_fish(
        self,
        fish_x: FISH_COORDINATES,
        fish_y: FISH_COORDINATES,
        species: str = "Apteronotus",
    ):
        """Create the fish given coordinates and species.

        :param fish_x: X coordinates of the fish(es) spine
        :param fish_y: Y coordinates of the fish(es) spine
        :param species: Species of fish to use
        """
        self.fish_container.init_fish(fish_x, fish_y, species)

    def update_for_image(self, null_properties, model_parameters):
        null_domains = null_properties.domains
        null_conductance = null_properties.value

        # allow single value to be set for all domains
        if isinstance(null_conductance, (int, float)):
            null_conductance = [null_conductance] * len(null_domains)

        original_conds = {}
        for domain, cond in zip(null_domains, null_conductance):
            domain_id = self.model_geometry[domain]
            original_conds[domain_id] = self.model_geometry.parameters["sigma"][
                domain_id
            ]
            self.update_parameter(domain_id, "sigma", cond, **model_parameters)

        return original_conds

    def solve(self, **model_parameters):
        """Solve the fish model.

        :param model_parameters: all model parameters
        """
        super().solve(**model_parameters)

        # update parameters forces a recompile so store out here
        active_solution = self._fem_solution

        null_properties: ElectricImageParameters = model_parameters.get("image", None)
        if null_properties:
            original_conds = self.update_for_image(null_properties, model_parameters)
            super().solve(parameters=model_parameters)

            # compute perturbation
            diff_solution = active_solution.copy(deepcopy=True)
            diff_solution.vector()[:] -= self._fem_solution.vector()[:]
            diff_solution.rename("diff", "diff")

            # reset to default conductivity in case of sweep
            for domain, original_cond in original_conds.items():
                self.update_parameter(
                    domain, "sigma", original_cond, **model_parameters
                )

            self.COMPILED_EQUATIONS = Equation(
                a=self.COMPILED_EQUATIONS.a,
                rhs=self.COMPILED_EQUATIONS.rhs,
                u=diff_solution,
            )

            self.active_solution = active_solution

    def generate_save_data(self) -> Tuple:
        """Add model solution and fish geometry to save data.

        :return: Fish specific data: skin geometry, transdermal potential (optional image data) and all parent data
        """
        save_geometry, tdp = tuple(zip(*compute_transdermal_potential(self)))
        if self.active_solution:  # defined only if perturbation computed
            active_sol = self.active_solution.copy(deepcopy=True)
            active_sol.rename("active", "active")
            return save_geometry + tdp + (active_sol,) + super().generate_save_data()
        else:
            return save_geometry + tdp + super().generate_save_data()

    def get_dirichlet_conditions(self, **kwargs) -> Tuple[BoundaryCondition, ...]:
        """Create the dirichlet conditions on the ground electrode and get user conditions.

        :param kwargs: Model parameters to feed into add_voltage_sources
        """
        extra_bc = self.add_voltage_sources(**kwargs)
        return extra_bc + (
            BoundaryCondition(value=0, label=self.model_geometry[self.GROUND_NAME]),
        )

    def add_geometry(self, **kwargs):
        """User extendable function to add to the geometry.

        Overwrite this method and implement something of the form

        self.model_geometry.add_domain("Domain name", obj1, obj2, ...)

        You can add additional arguments to represent model parameters which you can use in the object definition

        :param kwargs: Catchall for remaining parameters
        """

    def add_boundary_assignement_rules(self) -> Tuple[BOUNDARY_MARKER]:
        """ Add rules for assigning boundaries

        If defining new boundaries with custom labels this method should be overwritten

        :return: List of BoundaryBarkers
        """
        return tuple()

    def add_voltage_sources(self, **kwargs) -> Tuple[BoundaryCondition, ...]:
        """User extendable function to add a voltage source (constant voltage).

        Overwrite this method and implement something of the form

        return [BoundaryCondition(value=0, label=self.model_geometry['some domain']),
                BoundaryCondition(value=3, label=self.model_geometry['some other domain']),
                ...]

        You can add additional arguments to represent model parameters which you can use in the condition definition

        :param kwargs: Catchall for remaining parameters
        """
        return ()

    def add_current_sources(self, **kwargs) -> Tuple[BoundaryCondition, ...]:
        """User extendable function to add a current source (constant current).

        Overwrite this method and implement something of the form

        return [BoundaryCondition(value=0, label='some_domain'),
                BoundaryCondition(value=3, label='some_other_domain'),
                ...]

        You can add additional arguments to represent model parameters which you can use in the condition definition

        :param kwargs: Catchall for remaining parameters
        """
        return ()

    def create_geometry(
        self,
        fish_x: Sequence[float],
        fish_y: Sequence[float],
        species: str = "APTERONOTUS",
        **kwargs,
    ):
        """Create the model geometry by making a fish, tank and ground and adding any additional user geometry.

        :param fish_x: x coordinates (head->tail) for the fish
        :param fish_y: y coordinates (head->tail) for the fish
        :param species: Name of the species
        :param kwargs: Catchall for the remaining parameters
        """
        self.setup_fish(fish_x, fish_y, species)

        ground = Circle(self.GROUND_LOCATION, self.GROUND_RADIUS)
        tank = Rectangle.from_center([0, 0], self.TANK_SIZE)

        self.model_geometry.add_domain(
            self.WATER_NAME, tank, sigma=self.WATER_CONDUCTIVITY,
        )
        self.model_geometry.add_domain(
            self.GROUND_NAME, ground, sigma=self.GROUND_CONDUCTIVITY
        )
        self.add_geometry(**kwargs)

        # iterate over (possible) multiple fish and add them
        for ix, (outer_body, body, organ, skin_cond) in enumerate(
            zip(
                self.fish_container.outer_body,
                self.fish_container.body,
                self.fish_container.organ,
                self.fish_container.skin_conductance,
            )
        ):
            self.model_geometry.add_domain(
                f"{self.SKIN_NAME}_{ix}", outer_body, sigma=skin_cond
            )
            self.model_geometry.add_domain(
                f"{self.BODY_NAME}_{ix}", body, sigma=self.BODY_CONDUCTIVITY
            )
            self.model_geometry.add_domain(
                f"{self.ORGAN_NAME}_{ix}", organ, sigma=self.ORGAN_CONDUCTIVITY
            )

    def get_eod_functions(self, eod_phase: EOD_TYPE) -> Iterable[CubicSpline]:
        """ Convert the EOD phase into a set of EOD functions to evaluate

        :param eod_phase: The EOD phase for the fish(es)
        :return: List of evaluable EOD functions
        """
        try:
            iter(eod_phase)
        except TypeError:
            eod_phase = repeat(eod_phase)

        for phase, data_base in zip(eod_phase, self.fish_container.data_base):
            yield get_eod(phase, data_base)

    def get_neumann_conditions(self, eod_phase: EOD_TYPE = 0.24, **model_parameters):
        """Generate the Neumann conditions for the model.

        By default this is the EOD from the organ but can be extended with BCs from add_current_sourced

        :param eod_phase: EOD phase (0<=phase<=1)
        :param model_parameters: Catchall for the remaining parameters
        """

        extra_sources = self.add_current_sources(**model_parameters)

        # This looks like only the left boundary is a source but since both organ boundaries share the label and they're
        # "parallel": the shared parametrization is sufficient

        bc_eod_pair = zip(
            self.fish_container.eod_boundary_condition,
            self.get_eod_functions(eod_phase),
        )

        return extra_sources + tuple(
            (
                bc(eod, self.model_geometry.domain_names[f"{self.ORGAN_NAME}_{ix}"])
                for ix, (bc, eod) in enumerate(bc_eod_pair)
            )
        )

    def get_boundary_markers(self) -> Tuple[BOUNDARY_MARKER, ...]:
        return self.add_boundary_assignement_rules() + (
            self.organ_rule,
            self.inner_skin_rule,
        )

    def organ_rule(self, b1, b2) -> Tuple[bool, int]:
        """ Mark the boundary for organ.

        :param b1: Name of lower domain
        :param b2: Name of upper domain
        :return: If the boundary should be marked and the label
        """
        for ix in range(len(self.fish_container)):
            body_condition = b1 == self.model_geometry[f"{self.BODY_NAME}_{ix}"]
            organ_condition = b2 == self.model_geometry[f"{self.ORGAN_NAME}_{ix}"]
            if body_condition and organ_condition:
                return True, self.model_geometry[f"{self.ORGAN_NAME}_{ix}"]
        return False, -1

    def inner_skin_rule(self, b1, b2) -> Tuple[bool, int]:
        """ Mark the boundary for inner skin.

        :param b1: Name of lower domain
        :param b2: Name of upper domain
        :return: If the boundary should be marked and the label
        """
        for ix in range(len(self.fish_container)):
            skin_condition = b1 == self.model_geometry[f"{self.SKIN_NAME}_{ix}"]
            body_condition = b2 == self.model_geometry[f"{self.BODY_NAME}_{ix}"]
            if skin_condition and body_condition:
                return True, self.model_geometry[f"{self.BODY_NAME}_{ix}"]
        return False, -1
