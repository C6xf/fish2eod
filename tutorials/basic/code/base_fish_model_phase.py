"""Example where we show how to set eod phase."""

from fish2eod import BaseFishModel, plotting

"""
Define model parameters.

Here we draw a normal 20cm fish but set its phase to 0.5 in the EOD cycle
"""
parameters = {"fish_x": [0, 20], "fish_y": [0, 0], "phase": 0.5}

"""
Create the model and compile it

Plot the subdomains afterwards
"""
model = BaseFishModel()
model.compile(**parameters)
model.solve(**parameters)

plotting.mesh_plot_2d(model.solution, "solution")
plotting.plot_outline(model.solution)
