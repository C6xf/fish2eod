"""Example where we show how to set arbitrary coordinates."""
import numpy as np

from fish2eod import BaseFishModel


def compute_coordinates():
    """Compute rotated coordinates for a fish at 45cm fish at  45 degrees with an exponential curve."""

    # x from -30 to 15
    x = np.linspace(-30, 15, 30)

    # y has a subtle exponential bend
    y = np.exp((x + 30) / 20) - 1 - 30
    xy = np.array(list(zip(x, y)))  # join x and y

    rot = np.array(
        [
            [np.cos(np.pi / 4), np.sin(np.pi / 4)],
            [-np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ]
    )  # 45deg rotation matrix

    # apply rotation
    xy = np.dot(xy + [30, 30], rot) - [30, 30]

    # extract x and y coordinates
    x = xy[:, 0]
    y = xy[:, 1]

    return x, y


"""
Define model parameters.

Here the parameters set a fish with a head at -30cm and tail at 15cm then rotated 45degrees
"""
x_new, y_new = compute_coordinates()
parameters = {"fish_x": x_new, "fish_y": y_new}

"""
Create the model and compile it

Plot the subdomains afterwards
"""
model = BaseFishModel()
model.compile(**parameters)
model.plot_domains()
