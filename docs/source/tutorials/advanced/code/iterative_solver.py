from tempfile import TemporaryDirectory

from fish2eod import (
    ParameterSet,
    ParameterSweep,
    BaseFishModel,
    IterativeSolver,
    Circle,
)


class PreyClass(BaseFishModel):
    """Class to show parameter sweeps of a prey"""

    def add_geometry(self, prey_x, prey_y, **kwargs):
        prey = Circle([prey_x, prey_y], 0.5)
        self.model_geometry.add_domain("prey_domain", prey, sigma=1)


"""
Define ParameterSets for the prey_x and prey_y
"""
prey_x = ParameterSet("px", prey_x=[5, 10], rebuild_mesh=True)
prey_y = ParameterSet("py", prey_y=[3, 6], rebuild_mesh=True)
ps = ParameterSweep(prey_x, prey_y)
model_name = "prey_example"
save_path = TemporaryDirectory().name  # give a real path when doing this

solver = IterativeSolver(
    model_name, save_path, PreyClass, ps, fish_x=[0, 20], fish_y=[0, 20]
)
# solver.run()
