import matplotlib.pyplot as plt

from fish2eod import BaseFishModel, Circle, ElectricImageParameters, plotting


class PreyClass(BaseFishModel):
    def add_geometry(self, **kwargs):
        prey = Circle((3, 2), 0.5)

        self.model_geometry.add_domain("prey", prey, sigma=1)


model = PreyClass()
EIP = ElectricImageParameters(domains=("prey",), value=model.WATER_CONDUCTIVITY)
parameters = {"fish_x": [-15, 15], "fish_y": [0, 0], "image": EIP}
model.compile(**parameters)
model.solve(**parameters)

plotting.mesh_plot_2d(model.solution, "solution")
plotting.plot_outline(model.solution)
plt.xlim([-20, 20])
plt.ylim([-10, 10])
