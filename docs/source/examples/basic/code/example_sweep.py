from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from fish2eod import (
    BaseFishModel,
    Circle,
    ElectricImageParameters,
    plotting,
    IterativeSolver,
    ParameterSet,
    ParameterSweep,
    load_from_file,
)


class PreyClass(BaseFishModel):
    def add_geometry(self, prey_cond, **kwargs):
        prey = Circle((3, 2), 0.5)
        self.model_geometry.add_domain("prey", prey, sigma=prey_cond)


parameter_set = ParameterSet("sigma", prey_cond=[0.00023 / 10, 0.00023, 0.00023 * 10])
parameter_sweep = ParameterSweep(parameter_set)

d = TemporaryDirectory()  # VERY IMPORTANT - do not do this!
# you want to use a proper save directiory. This directory created as is WILL BE DELETED when python closes.

EIP = ElectricImageParameters(domains=("prey",), value=BaseFishModel.WATER_CONDUCTIVITY)
solver = IterativeSolver(
    "Example",
    d.name,
    PreyClass,
    parameter_sweep,
    fish_x=[-15, 15],
    fish_y=[0, 0],
    image=EIP,
)
solver.run()

loaded_solution = load_from_file(d.name + "/Example")
for ix in range(3):
    plt.figure()
    plotting.mesh_plot_2d(loaded_solution, "solution", sigma=ix)
    plotting.plot_outline(loaded_solution, sigma=ix)
