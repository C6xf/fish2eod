"""Example with how to create a polygon.

Note to users - please avoid polygons where possible as they are quite expensive.
"""
from fish2eod import Polygon

x = [
    0.95105652,
    0.22451399,
    0.0,
    -0.22451399,
    -0.95105652,
    -0.36327126,
    -0.58778525,
    0.0,
    0.58778525,
    0.36327126,
]
y = [
    0.30901699,
    0.30901699,
    1.0,
    0.30901699,
    0.30901699,
    -0.11803399,
    -0.80901699,
    -0.38196601,
    -0.80901699,
    -0.118034,
]

polygon = Polygon(x, y)
polygon.draw()  # this is for display purposes and not necessary for the general use
