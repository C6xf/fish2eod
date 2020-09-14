# coding=UTF-8
"""Solve a model repeatedly for multiple versions of parameters (parametric sweeps).

ParameterSet is a generic set of parameters for multiple parameters swept in serial (i.e. (x0, y0), (x1, y1), ...)
EODPhase is a parameterSet for sweeping the eod phase (0<=p<=1). No other parameters can be added to this
FishPosition is a ParameterSet for sweeping the position of the fish

ParameterSweep takes one or more parameter sets and generates the model parameters for all combinations of the sets
IterativeSolver iterates over parameter sweep and generates, solves and saves the model
"""

from collections import ChainMap
from functools import reduce
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence, Tuple, Union, Type

from tqdm import tqdm

from fish2eod.models import Model
from fish2eod.xdmf.save import Saver


class ParameterSet:
    """Set of parameters to be swept together. For example an x/y coordinate pair need to be kept together.

    Validates parameters and will fail if invalid shapes. A given parameter may be of any type as long as internal
    methods can handle. Exceptions are

    Fish coordinates must be numeric arrays [x1, x2, ...] for example
    EODPhase must be a number s.t. 0 <= p <= 1 [p1, p2, ...] for example

    For a generic parameter or parameter shared across all fish the format is

    [
        p1, p2, p3, ...
    ]

    For a parameter given individually for each fish the format is

    [

        [       #p1
            p1_fish1, p1_fish2, ...
        ] ,

        [       #p2
            p2_fish1, p2_fish2, ...
        ] , ...

    ]

    :param parameters: dictionary of parameter name and list of its parameters
    :param name: Name of the parameter set
    :param rebuild_mesh: Does the geometry change when parameters do? Yes for geometry parameters no if conductivity
    """

    def __init__(self, name: str, rebuild_mesh=False, **parameters: Sequence):
        """Create the ParameterSet."""
        self.validate(parameters)
        self._parameters = parameters
        self.name = name
        self.rebuild_mesh = rebuild_mesh

    def add_parameter_set(self, new_parameter_set: "ParameterSet") -> None:
        """Join two existing parameter sets. Only the parent's name is kept.

        :param new_parameter_set: The parameter set to join
        :return: None
        """
        assert len(self) == len(
            new_parameter_set
        ), "New parameter set must be of the same length"

        self._parameters = {
            **self._parameters,
            **new_parameter_set._parameters,
        }  # join the new parameters with the old

    @property
    def parameters(self) -> Tuple[str, ...]:
        """Get the names of the parameters inside the ParameterSet.

        return: Tuple of the names (order is irrelevant)
        """
        return tuple(self._parameters.keys())

    def __iter__(self) -> Iterator[Tuple[Dict[str, int], Dict[str, float]]]:
        """Iterate over the parameter set.

        :returns: (iterator) Dict of name and current step (0, 1, 2, ...) and the (parameter name:value) dict
        """
        for ix in range(len(self)):
            parameters_at_step = {
                name: self._parameters[name][ix] for name in self.parameters
            }
            yield {self.name: ix}, parameters_at_step

    def __len__(self) -> int:
        """Length of the parameter sweep (#steps)."""
        return len(self._parameters[self.parameters[0]])

    def validate(self, parameters: Dict[str, Sequence]) -> None:
        """Validate shape of parameters

        :param parameters: Dictionary (name: value) of parameters to sweep
        :return: None
        :raises: Assertion error if parameters are invalid shape
        """
        assert len(parameters) > 0, "At least one parameter must be specified"
        it = iter(parameters.values())
        target_len = len(next(it))
        assert all(
            len(lst) == target_len for lst in it
        ), "All parameters must have same number of values"

        multidimensional_length = []
        single_dimensional_length = []
        # todo refactor
        for name, p in parameters.items():  # ensure each parameter is a sequence
            assert isinstance(p, Iterable), "Parameters must not be list-like"
            assert len(p) > 0, "At least one parameter value must be specified"
            single_dimensional_length.append(len(p))
            try:  # check if multidimensional (multiple fish) will fail with type error otherwise
                assert all(
                    [len(p[0]) == len(x) for x in p]
                ), f'Parameter "{name}" is specified for different numbers of fish'
                multidimensional_length.append(len(p[0]))
            except TypeError:
                pass  # not multidimensional ignore
        assert (
            len(set(multidimensional_length)) <= 1
        ), "Some parameters specified for different numbers of fish"
        assert (
            len(set(single_dimensional_length)) == 1
        ), "Some parameters specified have different length"

    def to_sweep(self) -> "ParameterSweep":
        """Convert parameter set to a sweep if only parameter set.

        Convenience method to convert this parameter set into a sweep to avoid an extra boilerplate line

        :return: The parameter sweep
        """
        return ParameterSweep(self)


class EODPhase(ParameterSet):
    """Convenience class for sweeping EOD Phase.

    :param phases: Set of phases to use
    :param set_name: Name of the parameter set (Default is EODPhase)
    """

    def __init__(self, phases: Sequence, set_name="EODPhase"):
        """Instantiate EODPhase."""
        super().__init__(set_name, rebuild_mesh=False, eod_phase=phases)

    def add_parameter_set(self, new_parameter_set: "ParameterSet") -> None:
        """Overwrite add_parameter_set to disable it.

        At the "time-scale" of EOD phase it doesn't make sense to sweep any other parameters so disable functionality
        """
        raise NotImplementedError("Cannot add parameters to eod phase")

    def validate(self, parameters: Dict[str, Sequence]):
        super().validate(parameters)
        assert len(parameters) == 1, "Only one parameter may be speficied for EODPhase"


class FishPosition(ParameterSet):
    """Convenience class for sweeping fish positions.

    :param fish_x: Set of x positions to use
    :param fish_y: Set of y positions to use
    :param set_name: Name of the parameter set (Default is FishPosition)
    """

    def __init__(
        self,
        fish_x: Sequence[Sequence],
        fish_y: Sequence[Sequence],
        set_name="FishPosition",
    ):
        """Instantiate FishPosition."""
        super().__init__(set_name, rebuild_mesh=True, fish_x=fish_x, fish_y=fish_y)

    def validate(self, parameters: Dict[str, Sequence]):
        super().validate(parameters)
        assert len(parameters) == 2, "Only one parameter may be speficied for EODPhase"
        fish_x = parameters["fish_x"]
        fish_y = parameters["fish_y"]

        try:  # is multi fish? # TODO refactor this
            iter(fish_x[0][0])
            for step in range(len(fish_x)):
                for ix, (x, y) in enumerate(zip(fish_x[step], fish_y[step])):
                    assert len(x) == len(
                        y
                    ), f"Unequal number of x and y components at step {step} for fish {ix}"
        except TypeError:  # single fish
            for step, (x, y) in enumerate(zip(fish_x, fish_y)):
                assert len(x) == len(
                    y
                ), f"Unequal number of x and y components at step {step}"


class ParameterSweep:
    """Generate all combinations of parameter sets.

    :param parameter_set: An arbitrary (>0) number of parameter sets
    """

    def __init__(self, *parameter_sets: ParameterSet):
        """Instantiate ParameterSweep."""
        assert len(parameter_sets) > 0, "At least one parameter set must be specified"
        self.ordered_parameter_sets = sort_parameter_sets(parameter_sets)

    def __len__(self) -> int:
        """Length of the ParameterSweep len(ps1)*len(ps2)*..."""
        return reduce(
            lambda count, ps: count * len(ps), self.ordered_parameter_sets, 1
        )  # product of length of each param set

    def __iter__(self) -> Iterable[Tuple[bool, Dict[str, int], Dict[str, float]]]:
        """Iterate over combinations (combinatorial product) of parameter sweeps.

        :returns: if the model should be remeshed, the parameter dict, and the parameter level dict
        """
        canary_value = None
        for sim_info in product(*self.ordered_parameter_sets):
            remesh = self.should_remesh(canary_value, sim_info[self.remesh_index])
            if remesh:
                canary_value = sim_info[self.remesh_index]

            parameter_level, simulation_set = zip(*sim_info)

            # join the multiple parameter set dicts into one
            parameters = dict(ChainMap(*simulation_set))
            parameter_level = dict(ChainMap(*parameter_level))

            yield remesh, parameters, parameter_level

    def should_remesh(self, canary_value, new_value) -> bool:
        """Determine if the next simulation should be remeshed

        Check the canary value against the new value. If it's changed then remesh the simulation

        :param canary_value: The value of the remesh index the last time the simulation was remeshed
        :param new_value: The value of the remesh index for the next simulation
        :return: If the simulation should be remeshed
        """
        return (self.remesh_index >= 0) and (canary_value != new_value)

    @property
    def remesh_index(self) -> int:
        """Get the "canary" index for remeshing.

        Considering the ordered ParameterSets' remesh flags i.e. [True, True, True, False]. When the sets are run
        through sequentially the right-most remesh_flag containing parameter set will change indicating that the mesh
        needs to be recomputed

        :returns: The index of the canary or -1 if there is no remesh
        """
        n_sets = len(self.ordered_parameter_sets)
        remesh_flags = [ps.rebuild_mesh for ps in self.ordered_parameter_sets]

        if not any(remesh_flags):
            return -1

        return n_sets - 1 - remesh_flags[::-1].index(True)  # rightmost True


def sort_parameter_sets(
    parameter_sets: Sequence[ParameterSet],
) -> Sequence[ParameterSet]:
    """Get the parameter sets in their efficient order.

    Ideally all of the non-remesh parameters would be run sequentially only re-meshing when needed the order should
    be

    Order is remeshable, non-remeshable

    :returns: Parameter sets in their efficient order
    """
    return sorted(
        parameter_sets, key=lambda x: x.rebuild_mesh, reverse=True
    )  # reverse so [True,...,False,...]


class IterativeSolver:
    """Wrapper to take a model and run it for multiple versions parameter.

    Given a parameter sweep (set unique parameters to run through) the IterativeSolver will reset and re-run the model
    for each parameter set only remeshing when geometric features are changed for optimal performance.

    :param model: Class of model to create: NOT AN INSTANCE
    :param parameter_sweep: parametric sweep to run
    :param fixed_parameters: kwarg parameters that are fixed (fish_x / fish_y for example if not sweeping)
    """

    def __init__(
        self,
        name: str,
        save_path: Union[Path, str],
        model: Type[Model],
        parameter_sweep: ParameterSweep,
        **fixed_parameters,
    ):
        """Initialize IterativeSolver without running it."""
        self.parameters = parameter_sweep
        self.model = model()
        self.fixed_parameters = fixed_parameters
        self.saver = Saver(name, save_path)

    def run(self) -> None:
        """Iterate over parameter sweep and solve model with the given parameters."""
        for do_mesh, parameters, parameter_level in tqdm(self.parameters):
            self.run_step(parameters, do_mesh, parameter_level)

    def run_step(
        self, p: Dict[str, float], do_mesh: bool, parameter_level: Dict[str, int]
    ) -> None:
        """Solve model for a particular parameter step(set).

        :param p: Parameter set to solve the model for
        :param do_mesh: Should the model have its mesh computed
        :param parameter_level: Map of parameter and which version (1,2,3,...) it is
        """
        # combine the fixed parameters and the parameters at current step
        # in the event of conflict (double defined) the sweeping version will overwrite
        parameters = {**self.fixed_parameters, **p}

        # compile, run, and save the model step
        self.model.compile(**parameters, recompute_mesh=do_mesh)
        self.model.solve(**parameters)
        self.saver.save_model(self.model, metadata=parameter_level)
