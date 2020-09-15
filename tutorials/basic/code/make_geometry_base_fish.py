"""Example model showing how to work with model geometry in the BaseFishModel."""
import matplotlib.pyplot as plt

from fish2eod import BaseFishModel, Circle


class ExampleWithGeometry(BaseFishModel):
    """Example model with geometry"""

    def add_geometry(self, **kwargs):
        rod = Circle([5, 5], 2)
        self.model_geometry.add_domain("rod_domain", rod, sigma=1e6)


"""
Create and compile the model
"""
model = ExampleWithGeometry()
model.compile(fish_x=[0, 20], fish_y=[0, 0])  # ignore this for now.

"""
Plot the outline of the model.
"""
plt.figure()
model.plot_geometry(legend=True)
plt.title("Geometry")
