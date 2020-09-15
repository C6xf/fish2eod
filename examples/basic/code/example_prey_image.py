import matplotlib.pyplot as plt

from fish2eod import (
    BaseFishModel,
    Circle,
    ElectricImageParameters,
    compute_transdermal_potential,
)


class PreyClass(BaseFishModel):
    def add_geometry(self, **kwargs):
        prey = Circle((3, 2), 0.5)

        self.model_geometry.add_domain("prey", prey, sigma=1)


model = PreyClass()
EIP = ElectricImageParameters(domains=("prey",), value=model.WATER_CONDUCTIVITY)
parameters = {"fish_x": [-15, 15], "fish_y": [0, 0], "image": EIP}
model.compile(**parameters)
model.solve(**parameters)

organ, e_image = list(compute_transdermal_potential(model))[0]

plt.plot(organ.left_arc_length, e_image.left_tdp)
plt.plot(organ.right_arc_length, e_image.right_tdp)

plt.legend(["Left Image", "Right Image"])
