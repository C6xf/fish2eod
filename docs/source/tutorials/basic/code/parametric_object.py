import matplotlib.pyplot as plt

from fish2eod import BaseFishModel, Circle


class ParametricObject(BaseFishModel):
    def add_geometry(self, object_x, object_y, **kwargs):
        obj = Circle([object_x, object_y], 3)

        self.model_geometry.add_domain("object_domain", obj, sigma=1)


model = ParametricObject()

"""
Create the model with parameters

fish_x = [0, 10]
fish_y = [0,0]

This defines a fish spine running with two points on it - (0,0) and (10, 0).

Additionally we will define

object_x = 2
object_y = 10

which creates an object at (2, 10) per our model implementation
"""
parameters = {"fish_x": [0, 10], "fish_y": [0, 0], "object_x": 2, "object_y": 10}
model.compile(**parameters)  # shorthand for extracting each term from the dictionary
model.plot_geometry(color="k")

"""
We can re-do this but now with a different object location of (10, 10) by updating our parameters
"""
plt.figure()
parameters = {"fish_x": [0, 10], "fish_y": [0, 0], "object_x": 10, "object_y": 10}
# additionally since fish_x, and fish_y are not changing
# we could have just done parameters['object_x'] = 10

model.compile(**parameters)
model.plot_geometry(color="k")
