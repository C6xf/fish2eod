from itertools import product
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from fish2eod import (
    plotting,
    BaseFishModel,
    Circle,
    ParameterSet,
    ParameterSweep,
    IterativeSolver,
    load_from_file,
)

"""
Boilerplate to solve the model
"""


class PreyClass(BaseFishModel):
    def add_geometry(self, prey_x, prey_y, **kwargs):
        prey = Circle((prey_x, prey_y), 0.5)

        self.model_geometry.add_domain("prey", prey, sigma=1)


d = TemporaryDirectory()

prey_x = ParameterSet("prey_x", prey_x=[3, 6], rebuild_mesh=True)
prey_y = ParameterSet("prey_y", prey_y=[2, 4], rebuild_mesh=True)

parameter_sweep = ParameterSweep(prey_x, prey_y)
it = IterativeSolver(
    "temp", d.name, PreyClass, parameter_sweep, fish_x=[-15, 15], fish_y=[0, 0]
)
it.run()

"""
Extract the solution object

Plot the "FEM solution"
"""

loaded_solution = load_from_file(d.name + "/temp")
prey_x_values = loaded_solution.parameter_levels["prey_x"]
prey_y_values = loaded_solution.parameter_levels["prey_y"]
for prey_x, prey_y in product(prey_x_values, prey_y_values):
    plt.figure()

    # pass the solution parameters in an additional dictionary.
    # all other arguments can be used as normal
    plotting.mesh_plot_2d(loaded_solution, "solution", prey_x=prey_x, prey_y=prey_y)
    plotting.plot_outline(loaded_solution, prey_x=prey_x, prey_y=prey_y)
