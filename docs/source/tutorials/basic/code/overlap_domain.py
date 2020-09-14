"""Example model where we show overlapping domains."""
from fish2eod import BaseFishModel, Circle, Rectangle


class StackingClass(BaseFishModel):
    """Example class highlighting domain order."""

    def add_geometry(self, **kwargs):
        circle = Circle([5, 10], 5)
        square = Rectangle([5, 10], 10, 10)

        # the circle is added before the square meaning the square will obscure the circle
        self.model_geometry.add_domain("bottom", circle, sigma=1)
        self.model_geometry.add_domain("top", square, sigma=2)


"""
Returning to default behaviour.

Since the square was added second it overwrites any domain that it touches. Such as the quarter circle it overlaps.
"""
model = StackingClass()
model.compile(fish_x=[0, 10], fish_y=[0, 0])
model.plot_domains()
model.plot_geometry(color="k")
