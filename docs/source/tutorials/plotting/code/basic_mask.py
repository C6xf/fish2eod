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
Extract the solution object

Plot the "FEM solution"
Mask only the water (outside fish)
"""

model_solution = model.solution  # pull solution
mask = plotting.generate_mask(model.solution, include_domains=("water",))
plotting.mesh_plot_2d(
    model_solution, "solution", mask=mask
)  # plot the result named solution

# For convenience you can also exclude a domain by setting include=False
plt.figure()
mask = plotting.generate_mask(model.solution, include_domains=("water",), include=False)
plotting.mesh_plot_2d(
    model_solution, "solution", mask=mask
)  # plot the result named solution
