"""Example model where we show domains."""
import matplotlib.pyplot as plt

from fish2eod import BaseFishModel

"""
Create and compile the model.

Additionally plot the domains at different scales do see the detail
"""
model = BaseFishModel()
model.compile(fish_x=[-30, 30], fish_y=[0, 0])  # ignore this for now

plt.figure()
model.plot_domains()

plt.figure()
model.plot_domains()
plt.xlim([-30.1, -29.6])
plt.ylim([-0.1, 0.1])

plt.figure()
model.plot_domains()
plt.xlim([-25, -22])
plt.ylim([-2, 2])
