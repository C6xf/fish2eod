"""Example with how to create a circle."""
import matplotlib.pyplot as plt

from fish2eod import Circle

r = 1  # 1cm radius
p = [0, 5]  # location of center (x,y) = (0cm, 5cm)
circle = Circle(p, r)
circle.draw()  # this is for display purposes and not necessary for the general use
plt.title("Circles")
