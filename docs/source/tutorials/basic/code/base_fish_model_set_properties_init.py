"""Example where we introduce subclassing and default behaviour."""
from fish2eod import BaseFishModel


class DefaultBehaviour(BaseFishModel):
    """Example of subclassing. ExampleClass will have identical behaviour to BaseFishModel."""

    pass


"""
Create the model and then update properties.

Updating properties must be done before the model is compiled.
"""
model = DefaultBehaviour()

model.GROUND_RADIUS = 0.5  # cm
model.GROUND_LOCATION = [-20, 20]  # cm
model.MIRROR_GROUND = False  # if the ground is mirrored about the line y=0

model.TANK_SIZE = 70  # 70cm x 70cm

model.WATER_CONDUCTIVITY = 0.00023  # S/m
model.GROUND_CONDUCTIVITY = 727  # S/m
model.BODY_CONDUCTIVITY = 0.00356  # S/m
model.ORGAN_CONDUCTIVITY = 0.00927  # S/m
