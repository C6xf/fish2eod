"""Example where we solve the BaseFishModel."""
from fish2eod import BaseFishModel, plotting

"""
Define model parameters.

Here the parameters set a fish with a head at 0cm and tail at 10cm aligned with the y=0 line.
"""
parameters = {"fish_x": [0, 10], "fish_y": [0, 0]}

"""
Create the model and compile it.
After compilation the model is solved.

Plot the result and the outline (geometry).
"""
model = BaseFishModel()
model.compile(**parameters)
model.solve(**parameters)

plotting.mesh_plot_2d(model.solution, "solution")
plotting.plot_outline(model.solution)
