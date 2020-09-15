"""Example with Dirichelet conditions (voltage sources)."""
from fish2eod import BaseFishModel, Circle, BoundaryCondition, plotting


class ExampleVoltageSource(BaseFishModel):
    """Example model with an additional voltage source."""

    def add_geometry(self, **kwargs):
        """Add a circular object to carry the voltage."""
        source = Circle([0, 3], 1)
        self.model_geometry.add_domain("fg", source, sigma=1)

    def add_voltage_sources(self, **kwargs):
        """Add a voltage source to the circle"""
        # We can pull the domain id of the "fg" domain from the model geometry
        # value=0.1 means that the dirichelet condition is 0.1V. i.e. a 100mV source
        return (BoundaryCondition(value=0.1, label=self.model_geometry["fg"]),)


"""
Compile and solve the model

Plot the result with a color bar
"""
model = ExampleVoltageSource()
model.compile(fish_x=[0, 20], fish_y=[0, 0])
model.solve(fish_x=[0, 20], fish_y=[0, 0])

plotting.plot_outline(model.solution, color="k")
plotting.mesh_plot_2d(model.solution, "solution", colorbar=True)
