from fish2eod import BaseFishModel


class ExampleClass(BaseFishModel):
    pass


parameters = {"fish_x": [-15, 15], "fish_y": [0, 0]}
model = ExampleClass()
model.compile(parameters=parameters)
model.plot_geometry()
