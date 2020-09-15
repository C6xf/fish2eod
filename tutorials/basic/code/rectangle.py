"""Example with how to create a rectangle."""
import matplotlib.pyplot as plt

from fish2eod import Rectangle

"""
Create a stock-behaviour rectangle.
"""
w = 5  # 5cm wide
h = 3  # 3cm high
p = [-2, 3]  # location of bottom-left (x,y) = (-2cm, 3cm)
rectangle = Rectangle(p, w, h)
plt.figure()
plt.title("Rectangles and Squares")
rectangle.draw()  # this is for display purposes and not necessary for the general use

"""
For simplicity if you want a square i.e. w=h then you need only provide the length.
"""
square = Rectangle(p, 2)
square.draw(color="r")  # color is used to specify the line color of the shape

"""
Additionally it's sometimes natural to define a square from its center

We have the optional 'constructor' from_center to do this.
Here we create an identical rectangle to the first one - except instead of `p` representing the bottom left corner we
have p representing the center of the rectangle.

Similarly to a "normal" rectangle the height can be omitted to create a square w=h.
"""
plt.figure()
plt.title("Rectangle Defined at the Center")
center = [0, 0]
center_rectangle = Rectangle.from_center(p, w, h)
center_rectangle.draw()
