"""Example with Neumann conditions (current sources)."""
from fish2eod import BaseFishModel, Circle, BoundaryCondition, plotting


class ExampleCurrentSource(BaseFishModel):
    """Example model with an additional voltage source."""

    def add_geometry(self, **kwargs):
        """Add a circular object to carry the current."""
        source = Circle([0, 3], 1)
        self.model_geometry.add_domain("fg", source, sigma=1)

    def add_current_sources(self, **kwargs):
        """Add a voltage source to the circle"""
        # We can pull the domain id of the "fg" domain from the model geometry
        # value=-3e-6 means that the Neurmann condition is as -3uA/cm^2 source
        return (BoundaryCondition(value=-3e-6, label=self.model_geometry["fg"]),)


"""
Compile and solve the model

Plot the result with a color bar
"""
model = ExampleCurrentSource()
model.compile(fish_x=[0, 20], fish_y=[0, 0])
model.solve(fish_x=[0, 20], fish_y=[0, 0])

plotting.plot_outline(model.solution, color="k")
plotting.mesh_plot_2d(model.solution, "solution", colorbar=True)
