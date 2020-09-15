from fish2eod import BaseFishModel, Rectangle, ElectricImageParameters, plotting


class PreyClass(BaseFishModel):
    SHUTTLE_CONDUCTANCE = 1e-6

    def add_geometry(self, **kwargs):
        shuttle = Rectangle((-5, 2), 15, 2)
        hole = Rectangle((4, 2), 2, 2)

        self.model_geometry.add_domain(
            "shuttle", shuttle, sigma=self.SHUTTLE_CONDUCTANCE
        )
        self.model_geometry.add_domain("hole", hole, sigma=self.SHUTTLE_CONDUCTANCE)


model = PreyClass()
EIP = ElectricImageParameters(domains=("hole",), value=model.WATER_CONDUCTIVITY)
parameters = {"fish_x": [-15, 15], "fish_y": [0, 0], "image": EIP}
model.compile(**parameters)
model.solve(**parameters)

plotting.plot_outline(model.solution)
plotting.mesh_plot_2d(model.solution, "solution")
