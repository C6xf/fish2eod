import matplotlib.pyplot as plt

from fish2eod import plotting, BaseFishModel, Circle

"""
Boilerplate to solve the model
"""


class PreyClass(BaseFishModel):
    def add_geometry(self, **kwargs):
        prey = Circle((3, 2), 0.5)

        self.model_geometry.add_domain("prey", prey, sigma=1)


model = PreyClass()
parameters = {"fish_x": [-15, 15], "fish_y": [0, 0]}
model.compile(**parameters)
model.solve(**parameters)

"""
Plot showing effect of color_style and flags

The colorbar flag turns the colorbar on or off
The mask flag sets the mask (requires valid mask)
"""

model_solution = model.solution  # pull solution
plt.figure()
plotting.mesh_plot_2d(
    model_solution, "solution", colorbar=True, mask=None, color_style="Full"
)  # plot the result named solution
plt.figure()
plotting.mesh_plot_2d(
    model_solution, "solution", colorbar=True, mask=None, color_style="Equal"
)  # plot the result named solution
plt.figure()
plotting.mesh_plot_2d(
    model_solution, "solution", colorbar=True, mask=None, color_style=None
)  # plot the result named solution
plt.figure()
plotting.mesh_plot_2d(
    model_solution, "solution", colorbar=True, mask=None, color_style=[-0.005, 0.005]
)  # plot the result named solution


"""
Plots also take standard matplotlib properties

Here we change the colormap and the alpha chanel
"""
plt.figure()
plotting.mesh_plot_2d(model_solution, "solution", cmap="hot", alpha=0.5)
