"""Example highlighting finite element mesh and domains."""
import matplotlib.pyplot as plt

from fish2eod import Circle, Rectangle, QESModel


class MeshClass(QESModel):
    """Simple model defining two rectangles and a circle."""

    def create_geometry(self, **_):
        """Create rectangles and a circle for display."""
        background = Rectangle.from_center([0, 0], 10, 10)
        r1 = Rectangle([0, 0], 1, 2)
        r2 = Rectangle([-3, -3], 2, 1)
        c = Circle([3, 3], 1)

        self.model_geometry.add_domain("bg", background, sigma=1)
        self.model_geometry.add_domain("f1", r1, sigma=1)
        self.model_geometry.add_domain("f2", r2, c, sigma=1)


"""
Create and compile the model
"""
model = MeshClass()
model.compile()

"""
Plot domains and geometry outline
"""
plt.figure()
model.plot_domains()
model.plot_geometry(color="k")

"""
Plot mesh and outline
"""
plt.figure()
model.plot_mesh(color=[0.7, 0.7, 0.7])
model.plot_geometry(color="k")
